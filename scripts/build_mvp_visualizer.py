#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cache_mvp_global_character_centered_profiles import (  # noqa: E402
    DEFAULT_SAFE_TAGS,
    favorites_weighted_row_weight,
    descriptor_support,
    load_character_descriptors,
    normalize,
    read_assignable_basis,
    read_basis,
    read_json,
    rounded,
    top_roles,
    write_json,
)
from analyze_sv1_diffusivity import lowdin_from_global_gram, nnls_pg  # noqa: E402
from role_edge_exclusions import DEFAULT_EXCLUSIONS_PATH, filter_excluded_role_edges, load_role_edge_exclusions  # noqa: E402
from seiyuu_local_nmf_lane_svd import load_bangumi_collects  # noqa: E402
from seiyuu_local_nmf_lane_svd import load_or_create_embeddings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the MVP seiyuu role-profile visualizer payload.")
    parser.add_argument("--basis", type=Path, default=Path("run/production_personality_basis/production_personality_basis_kept.tsv"))
    parser.add_argument("--role-edges", type=Path, default=Path("data/role_edges_current_seiyuu_expanded.json"))
    parser.add_argument("--role-edge-exclusions", type=Path, default=DEFAULT_EXCLUSIONS_PATH)
    parser.add_argument("--merged-tags", type=Path, default=Path("data/external/merged/all_characters_llm_vndb_personality_tags.json"))
    parser.add_argument("--safe-tags", nargs="*", type=Path, default=DEFAULT_SAFE_TAGS)
    parser.add_argument("--safe-enrichment", type=Path, default=Path("data/external/safe_enrichment/character_safe_enrichment.jsonl"))
    parser.add_argument("--bangumi-raw-dir", type=Path, default=Path("data/external/safe_enrichment/raw/bangumi"))
    parser.add_argument("--canonicalization", type=Path, default=Path("models/global_descriptor_canonicalization/descriptor_canonicalization.json"))
    parser.add_argument("--embedding-cache-dir", type=Path, default=Path("run/production_personality_basis/embeddings"))
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--output-dir", type=Path, default=Path("site/mvp_visualizer"))
    parser.add_argument("--top-descriptors", type=int, default=5)
    parser.add_argument("--round-digits", type=int, default=6)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def slug(value: str) -> str:
    output = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return output or "profile"


def clean_generated_profiles(profiles_dir: Path) -> None:
    for stale_path in profiles_dir.glob("*.json"):
        stale_path.unlink()
    for mode_key in WEIGHT_MODES:
        mode_dir = profiles_dir / mode_key
        if not mode_dir.exists():
            continue
        for stale_path in mode_dir.glob("*.json"):
            stale_path.unlink()


def norm_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def search_aliases(name: str, native_name: str = "") -> list[str]:
    aliases = {norm_name(name), norm_name(native_name)}
    parts = [part for part in norm_name(name).split() if part]
    if len(parts) >= 2:
        aliases.add(" ".join(reversed(parts)))
    return sorted(alias for alias in aliases if alias)


def bangumi_source_by_character(safe_tags: list[Path]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for path in safe_tags:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                character_id = payload.get("anilist_character_id")
                if character_id is None:
                    continue
                source_blocks = payload.get("source_blocks") or []
                bangumi_blocks = [block for block in source_blocks if str(block.get("source") or "").lower() == "bangumi"]
                if not bangumi_blocks:
                    continue
                block = bangumi_blocks[0]
                rows[int(character_id)] = {
                    "bangumi_url": block.get("url") or "",
                    "bangumi_source_key": block.get("source_key") or "",
                    "bangumi_license": block.get("license") or "",
                    "bangumi_favorites": int(
                        payload.get("bangumi_favorites")
                        or payload.get("bangumi_favourites")
                        or block.get("favorites")
                        or block.get("favourites")
                        or 0
                    ),
                }
    return rows


def descriptor_rows(
    descriptors: list[str],
    scores: np.ndarray,
    supported_indices: set[int],
    support_by_descriptor: dict[str, dict[str, Any]],
    *,
    positive: bool,
    limit: int,
    digits: int,
) -> list[dict[str, Any]]:
    if positive:
        candidate_indices = [index for index in supported_indices if scores[index] > 0.0]
        candidate_indices.sort(key=lambda index: float(scores[index]), reverse=True)
    else:
        candidate_indices = [index for index in range(len(descriptors)) if scores[index] < 0.0]
        candidate_indices.sort(key=lambda index: float(scores[index]))
    output = []
    for index in candidate_indices[:limit]:
        descriptor = descriptors[index]
        support = support_by_descriptor.get(descriptor, {})
        output.append(
            {
                "descriptor": descriptor,
                "score": rounded(float(scores[index]), digits),
                "characters": int(support.get("characters") or 0),
                "weighted_support": rounded(float(support.get("weighted_support") or 0.0), digits),
                "anilist_favorites": int(support.get("favourites") or 0),
            }
        )
    return output


def fit_quality(coefficients: np.ndarray, active: list[int], atoms: np.ndarray, target: np.ndarray) -> float:
    if not active:
        return 0.0
    approximation = coefficients @ atoms[active]
    return float((approximation @ target) / max(float(np.linalg.norm(approximation)), 1.0e-12))


def supported_descriptor_fit(
    descriptors: list[str],
    atoms: np.ndarray,
    target: np.ndarray,
    support_rows: list[dict[str, Any]],
    descriptor_index: dict[str, int],
    *,
    fit_target: float = 0.80,
    max_terms: int = 80,
    limit: int = 5,
    digits: int = 6,
) -> dict[str, Any]:
    support_indices = [
        descriptor_index[row["descriptor"]]
        for row in support_rows
        if row["descriptor"] in descriptor_index
    ][:max_terms]
    support_by_descriptor = {row["descriptor"]: row for row in support_rows}

    active: list[int] = []
    positive_coefficients = np.zeros(0, dtype=np.float64)
    positive_fit = 0.0
    for descriptor_id in support_indices:
        active.append(descriptor_id)
        positive_coefficients = nnls_pg(atoms[active], target, max_iter=3000)
        positive_fit = fit_quality(positive_coefficients, active, atoms, target)
        if positive_fit >= fit_target:
            break

    signed_active: list[int] = []
    signed_coefficients = np.zeros(0, dtype=np.float64)
    signed_fit = 0.0
    for descriptor_id in support_indices:
        signed_active.append(descriptor_id)
        signed_coefficients = np.linalg.lstsq(atoms[signed_active].T, target, rcond=None)[0]
        signed_fit = fit_quality(signed_coefficients, signed_active, atoms, target)
        if abs(signed_fit) >= fit_target:
            break

    def row_for(descriptor_id: int, coefficient: float, share: float) -> dict[str, Any]:
        descriptor = descriptors[descriptor_id]
        support = support_by_descriptor.get(descriptor, {})
        return {
            "descriptor": descriptor,
            "coefficient": rounded(float(coefficient), digits),
            "share": rounded(float(share), digits),
            "characters": int(support.get("characters") or 0),
            "weighted_support": rounded(float(support.get("weighted_support") or 0.0), digits),
            "anilist_favorites": int(support.get("favourites") or 0),
        }

    positive_rows = []
    positive_total = max(float(np.sum(positive_coefficients)), 1.0e-12)
    for offset in np.argsort(positive_coefficients)[::-1]:
        coefficient = float(positive_coefficients[int(offset)])
        if coefficient <= 1.0e-9:
            continue
        positive_rows.append(row_for(active[int(offset)], coefficient, coefficient / positive_total))

    signed_positive = []
    signed_negative = []
    signed_total = max(float(np.sum(np.abs(signed_coefficients))), 1.0e-12)
    for offset in np.argsort(np.abs(signed_coefficients))[::-1]:
        coefficient = float(signed_coefficients[int(offset)])
        if abs(coefficient) <= 1.0e-9:
            continue
        row = row_for(signed_active[int(offset)], coefficient, abs(coefficient) / signed_total)
        if coefficient >= 0 and len(signed_positive) < limit:
            signed_positive.append(row)
        elif coefficient < 0 and len(signed_negative) < limit:
            signed_negative.append(row)
        if len(signed_positive) >= limit and len(signed_negative) >= limit:
            break

    return {
        "positive_fit_percent": rounded(float(positive_fit * 100.0), digits),
        "signed_fit_percent": rounded(float(abs(signed_fit) * 100.0), digits),
        "positive_only": positive_rows[:limit],
        "more": signed_positive,
        "less": signed_negative,
    }


def unit_row_weight(character: dict[str, Any]) -> float:
    return 1.0


WEIGHT_MODES = {
    "unit": {
        "label": "Unweighted",
        "description": "Every supported character contributes equally.",
        "weight_fn": unit_row_weight,
        "row_weight": "1",
    },
    "favorites_weighted": {
        "label": "Popularity-weighted",
        "description": "Characters are weighted by public favorites, with shared-role downweighting.",
        "weight_fn": favorites_weighted_row_weight,
        "row_weight": "sqrt(log1p(AniList + Bangumi favorites)+1) / sqrt(role_edge_count)",
    },
}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir = args.output_dir / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    clean_generated_profiles(profiles_dir)

    descriptors = read_basis(args.basis)
    descriptor_index = {descriptor: index for index, descriptor in enumerate(descriptors)}
    basis_set = set(descriptors)
    assignable_set = read_assignable_basis(args.basis)
    raw_to_canonical = (read_json(args.canonicalization).get("raw_to_canonical") or {}) if args.canonicalization.exists() else {}

    character_descriptors, character_meta = load_character_descriptors(
        args.merged_tags,
        args.safe_tags,
        basis_set,
        assignable_set,
        raw_to_canonical,
    )
    bangumi_by_character = bangumi_source_by_character(args.safe_tags)
    bangumi_collects_by_character = (
        load_bangumi_collects(args.safe_enrichment, args.bangumi_raw_dir)
        if args.safe_enrichment.exists() and args.bangumi_raw_dir.exists()
        else {}
    )

    embeddings = load_or_create_embeddings(descriptors, args.embedding_cache_dir, args.embedding_model)
    _, _, descriptor_atoms, _ = lowdin_from_global_gram(embeddings.astype(np.float64))

    roles_payload = read_json(args.role_edges)
    roles, excluded_roles = filter_excluded_role_edges(
        roles_payload.get("roles") or [],
        load_role_edge_exclusions(args.role_edge_exclusions),
    )
    role_edge_count_by_character: dict[int, int] = defaultdict(int)
    for role in roles:
        role_edge_count_by_character[int(role["character"]["character_id"])] += 1

    seiyuu_by_id: dict[int, dict[str, Any]] = {}
    characters_by_seiyuu: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    unique_characters: dict[int, dict[str, Any]] = {}

    for role in roles:
        seiyuu = role["seiyuu"]
        character = role["character"]
        seiyuu_id = int(seiyuu["seiyuu_id"])
        character_id = int(character["character_id"])
        seiyuu_by_id[seiyuu_id] = seiyuu

        row_descriptors = sorted(descriptor for descriptor in character_descriptors.get(character_id, set()) if descriptor in descriptor_index)
        if not row_descriptors:
            continue

        meta = character_meta.get(character_id) or {}
        bangumi = bangumi_by_character.get(character_id) or {}
        bangumi_collects = bangumi_collects_by_character.get(character_id) or {}
        anilist_favorites = int(character.get("favourites") or meta.get("favourites") or 0)
        bangumi_favorites = int(
            bangumi_collects.get("bangumi_collects")
            or bangumi.get("bangumi_favorites")
            or 0
        )
        row = {
            "character_id": character_id,
            "name": character.get("name") or meta.get("name") or "",
            "anime": character.get("first_anime") or meta.get("first_anime") or "",
            "anilist_favorites": anilist_favorites,
            "bangumi_favorites": bangumi_favorites,
            "aggregate_favorites": anilist_favorites + bangumi_favorites,
            "image": character.get("image") or meta.get("image") or "",
            "site_url": character.get("site_url") or meta.get("site_url") or "",
            "bangumi_url": bangumi_collects.get("bangumi_url") or bangumi.get("bangumi_url") or "",
            "bangumi_license": bangumi.get("bangumi_license") or "",
            "bangumi_comments": int(bangumi_collects.get("bangumi_comments") or 0),
            "role_edge_count": role_edge_count_by_character[character_id],
            "descriptors": row_descriptors,
        }
        # Preserve compatibility with shared row-weight helpers while using the
        # aggregate public signal requested for the production toggle.
        row["favourites"] = row["aggregate_favorites"]
        existing = characters_by_seiyuu[seiyuu_id].get(character_id)
        if existing is None or row["anilist_favorites"] > int(existing.get("anilist_favorites") or 0):
            characters_by_seiyuu[seiyuu_id][character_id] = row
            unique_characters[character_id] = row

    def character_vector(character: dict[str, Any]) -> np.ndarray:
        indices = [descriptor_index[descriptor] for descriptor in character.get("descriptors") or []]
        if not indices:
            return np.zeros(descriptor_atoms.shape[1], dtype=np.float64)
        return np.sum(descriptor_atoms[indices], axis=0)

    character_vectors: dict[int, np.ndarray] = {
        character_id: character_vector(character)
        for character_id, character in unique_characters.items()
    }

    mode_indexes: dict[str, list[dict[str, Any]]] = {}
    ranking_paths: dict[str, str] = {}
    ranking_counts: dict[str, int] = {}
    for mode_key, mode in WEIGHT_MODES.items():
        weight_fn = mode["weight_fn"]
        mode_profiles_dir = profiles_dir / mode_key
        mode_profiles_dir.mkdir(parents=True, exist_ok=True)

        global_sum = np.zeros(descriptor_atoms.shape[1], dtype=np.float64)
        for character_id, character in unique_characters.items():
            global_sum += weight_fn(character) * character_vectors[character_id]
        global_character_vector = normalize(global_sum)

        def centered_character_vector(character: dict[str, Any]) -> np.ndarray:
            return normalize(normalize(character_vector(character)) - global_character_vector)

        index_profiles = []
        ranking_rows = []
        for seiyuu_id, character_map in sorted(
            characters_by_seiyuu.items(),
            key=lambda item: item[1] and next(iter(item[1].values())).get("name", ""),
        ):
            characters = list(character_map.values())
            if not characters:
                continue
            weighted_sum = np.zeros(descriptor_atoms.shape[1], dtype=np.float64)
            for character in characters:
                weighted_sum += weight_fn(character) * character_vector(character)
            uncentered_vector = normalize(weighted_sum)
            centered_vector = normalize(uncentered_vector - global_character_vector)
            descriptor_scores = descriptor_atoms @ centered_vector
            support_rows = descriptor_support(characters, digits=args.round_digits, weight_fn=weight_fn)
            support_by_descriptor = {row["descriptor"]: row for row in support_rows}
            supported_indices = {
                descriptor_index[descriptor]
                for character in characters
                for descriptor in character.get("descriptors") or []
                if descriptor in descriptor_index
            }

            positive_descriptors = descriptor_rows(
                descriptors,
                descriptor_scores,
                supported_indices,
                support_by_descriptor,
                positive=True,
                limit=args.top_descriptors,
                digits=args.round_digits,
            )
            negative_descriptors = descriptor_rows(
                descriptors,
                descriptor_scores,
                supported_indices,
                support_by_descriptor,
                positive=False,
                limit=args.top_descriptors,
                digits=args.round_digits,
            )
            summary_fit = supported_descriptor_fit(
                descriptors,
                descriptor_atoms,
                centered_vector,
                support_rows,
                descriptor_index,
                limit=args.top_descriptors,
                digits=args.round_digits,
            )

            character_rows = []
            for character in characters:
                vector = centered_character_vector(character)
                raw_projection = float(vector @ centered_vector)
                projection = min(1.0, max(0.0, raw_projection))
                descriptor_projection_scores = descriptor_atoms @ vector
                character_rows.append(
                    {
                        "character_id": character["character_id"],
                        "name": character["name"],
                        "anime": character["anime"],
                        "image": character["image"],
                        "site_url": character["site_url"],
                        "bangumi_url": character["bangumi_url"],
                        "projection": rounded(projection, args.round_digits),
                        "raw_projection": rounded(raw_projection, args.round_digits),
                        "anilist_favorites": character["anilist_favorites"],
                        "bangumi_favorites": character["bangumi_favorites"],
                        "aggregate_favorites": character["aggregate_favorites"],
                        "role_edge_count": character["role_edge_count"],
                        "descriptors": character["descriptors"][:24],
                        "descriptor_scores": np.round(descriptor_projection_scores, args.round_digits).tolist(),
                    }
                )
            character_rows.sort(
                key=lambda row: (
                    float(row["projection"]),
                    int(row["aggregate_favorites"]),
                    row["name"],
                ),
                reverse=True,
            )

            seiyuu = seiyuu_by_id[seiyuu_id]
            profile_slug = f"{slug(seiyuu.get('name') or str(seiyuu_id))}_{seiyuu_id}"
            profile = {
                "seiyuu_id": seiyuu_id,
                "name": seiyuu.get("name") or "",
                "native_name": seiyuu.get("native_name") or "",
                "image": seiyuu.get("image") or "",
                "site_url": seiyuu.get("site_url") or "",
                "role_count": int(seiyuu.get("role_count") or 0),
                "character_count": int(seiyuu.get("character_count") or len(characters)),
                "supported_character_count": len(characters),
                "first_year": seiyuu.get("first_year"),
                "weight_mode": mode_key,
                "weight_mode_label": mode["label"],
                "positive_descriptors": positive_descriptors,
                "negative_descriptors": negative_descriptors,
                "summary_fit": summary_fit,
                "descriptor_support": support_rows[:40],
                "notable_roles": top_roles(characters),
                "characters": character_rows,
                "favorite_note": "AniList and Bangumi favorites are from cached public-source records. Bangumi favorites are included when present; otherwise 0.",
            }
            write_json(mode_profiles_dir / f"{profile_slug}.json", profile)
            index_profiles.append(
                {
                    "seiyuu_id": seiyuu_id,
                    "name": profile["name"],
                    "native_name": profile["native_name"],
                    "image": profile["image"],
                    "site_url": profile["site_url"],
                    "role_count": profile["role_count"],
                    "supported_character_count": profile["supported_character_count"],
                    "first_year": profile["first_year"],
                    "aliases": search_aliases(profile["name"], profile["native_name"]),
                    "profile_path": f"mvp_visualizer/profiles/{mode_key}/{profile_slug}.json",
                    "positive_descriptors": summary_fit.get("more", positive_descriptors)[:3],
                }
            )
            ranking_rows.append(
                {
                    "seiyuu_id": seiyuu_id,
                    "name": profile["name"],
                    "native_name": profile["native_name"],
                    "image": profile["image"],
                    "site_url": profile["site_url"],
                    "role_count": profile["role_count"],
                    "character_count": profile["supported_character_count"],
                    "first_year": profile["first_year"],
                    "descriptor_scores": np.round(descriptor_scores, args.round_digits).tolist(),
                    "notable_roles": top_roles(characters),
                }
            )

        index_profiles.sort(key=lambda row: (row["supported_character_count"], row["role_count"], row["name"]), reverse=True)
        mode_indexes[mode_key] = index_profiles
        ranking_rows.sort(key=lambda row: (row["character_count"], row["role_count"], row["name"]), reverse=True)
        ranking_path = args.output_dir / f"rankings_{mode_key}.json"
        ranking_paths[mode_key] = f"mvp_visualizer/{ranking_path.name}"
        ranking_counts[mode_key] = len(ranking_rows)
        write_json(
            ranking_path,
            {
                "generated_at": utc_now(),
                "source": "build_mvp_visualizer.py",
                "mode": {
                    "key": mode_key,
                    "label": mode["label"],
                    "description": mode["description"],
                    "row_weight": mode["row_weight"],
                    "global_centering": "normalize(sum(weight(character) * character_descriptor_vector))",
                    "profile_vector": "normalize(normalize(sum(weight(character) * character_descriptor_vector)) - global_character_vector)",
                },
                "descriptors": descriptors,
                "seiyuu": ranking_rows,
            },
        )

    default_profiles = mode_indexes["unit"]
    profiles_by_id: dict[int, dict[str, Any]] = {}
    for row in default_profiles:
        profiles_by_id[int(row["seiyuu_id"])] = {
            **row,
            "profile_paths": {"unit": row["profile_path"]},
            "positive_descriptors_by_mode": {"unit": row.get("positive_descriptors") or []},
        }
    for mode_key, rows in mode_indexes.items():
        if mode_key == "unit":
            continue
        for row in rows:
            merged = profiles_by_id.setdefault(
                int(row["seiyuu_id"]),
                {
                    **row,
                    "profile_paths": {},
                    "positive_descriptors_by_mode": {},
                },
            )
            merged["profile_paths"][mode_key] = row["profile_path"]
            merged["positive_descriptors_by_mode"][mode_key] = row.get("positive_descriptors") or []
    index_profiles = sorted(
        profiles_by_id.values(),
        key=lambda row: (row["supported_character_count"], row["role_count"], row["name"]),
        reverse=True,
    )
    payload = {
        "generated_at": utc_now(),
        "source": "build_mvp_visualizer.py",
        "model": {
            "basis": str(args.basis),
            "role_edges": str(args.role_edges),
            "role_edge_exclusions": str(args.role_edge_exclusions),
            "excluded_role_edge_count": len(excluded_roles),
            "safe_enrichment": str(args.safe_enrichment),
            "bangumi_raw_dir": str(args.bangumi_raw_dir),
            "descriptor_count": len(descriptors),
            "orthogonal_rank": int(descriptor_atoms.shape[1]),
            "modes": {
                key: {
                    "label": value["label"],
                    "description": value["description"],
                    "row_weight": value["row_weight"],
                    "ranking_path": ranking_paths[key],
                    "ranking_count": ranking_counts[key],
                }
                for key, value in WEIGHT_MODES.items()
            },
            "default_mode": "unit",
            "profile_vector": "normalize(normalize(mode_weighted_seiyuu_character_sum) - normalize(mode_weighted_global_character_sum))",
            "character_projection": (
                "max(0, dot(normalize(normalize(character_descriptor_vector) - mode_global_character_vector), "
                "normalized_profile_vector))"
            ),
        },
        "profiles": index_profiles,
        "samples": index_profiles[:24],
    }
    write_json(args.output_dir / "index.json", payload)
    print(f"wrote {args.output_dir / 'index.json'} with {len(index_profiles)} profiles")


if __name__ == "__main__":
    main()
