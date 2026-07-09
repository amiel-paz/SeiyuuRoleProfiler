#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cache_mvp_global_character_centered_profiles import (
    DEFAULT_SAFE_TAGS,
    favorites_weighted_row_weight,
    load_character_descriptors,
    normalize,
    read_assignable_basis,
    read_basis,
    read_json,
    rounded,
    write_json,
)
from analyze_sv1_diffusivity import lowdin_from_global_gram
from build_mvp_visualizer import bangumi_source_by_character
from role_edge_exclusions import DEFAULT_EXCLUSIONS_PATH, filter_excluded_role_edges, load_role_edge_exclusions
from seiyuu_local_nmf_lane_svd import load_bangumi_collects, load_or_create_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cache per-seiyuu role-profile stabilization curves. At each six-month checkpoint, "
            "compare the cumulative character-vector sum to the cumulative sum six months earlier."
        )
    )
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
    parser.add_argument("--output", type=Path, default=Path("run/profile_stabilization_curves.json"))
    parser.add_argument("--tail-periods", type=int, default=2, help="Six-month periods to append after the latest known role year.")
    parser.add_argument("--round-digits", type=int, default=6)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def unit_row_weight(_: dict[str, Any]) -> float:
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
        "description": (
            "Each character is weighted by sqrt(log1p(AniList + Bangumi favorites) + 1) "
            "and downweighted by sqrt(shared role edges)."
        ),
        "weight_fn": favorites_weighted_row_weight,
        "row_weight": "sqrt(log1p(aggregate_favorites)+1)/sqrt(role_edge_count)",
    },
}


def month_index(year: int, month: int = 1) -> int:
    return year * 12 + (month - 1)


def month_index_to_date(value: int) -> str:
    year = value // 12
    month = value % 12 + 1
    return f"{year:04d}-{month:02d}-01"


def role_year(role: dict[str, Any]) -> int | None:
    years = []
    if role.get("first_year"):
        years.append(int(role["first_year"]))
    for anime in role.get("anime") or []:
        if anime.get("year"):
            years.append(int(anime["year"]))
    return min(years) if years else None


def aggregate_bangumi_favorites(safe_tags: list[Path], safe_enrichment: Path, bangumi_raw_dir: Path) -> dict[int, int]:
    source_by_character = bangumi_source_by_character(safe_tags)
    collects_by_character = (
        load_bangumi_collects(safe_enrichment, bangumi_raw_dir)
        if safe_enrichment.exists() and bangumi_raw_dir.exists()
        else {}
    )
    output: dict[int, int] = {}
    for character_id in set(source_by_character).union(collects_by_character):
        source = source_by_character.get(character_id) or {}
        collects = collects_by_character.get(character_id) or {}
        output[int(character_id)] = int(
            collects.get("bangumi_collects")
            or source.get("bangumi_favorites")
            or 0
        )
    return output


def cumulative_vector(
    characters: list[dict[str, Any]],
    vectors: dict[int, np.ndarray],
    weight_fn,
) -> tuple[np.ndarray, float]:
    total = None
    norm_before_normalize = 0.0
    for character in characters:
        vector = vectors[int(character["character_id"])]
        if total is None:
            total = np.zeros_like(vector, dtype=np.float64)
        total += weight_fn(character) * vector
    if total is None:
        return np.zeros(0, dtype=np.float64), 0.0
    norm_before_normalize = float(np.linalg.norm(total))
    return normalize(total), norm_before_normalize


def settled_after(points: list[dict[str, Any]], threshold: float) -> int | None:
    distances = [
        (index, point)
        for index, point in enumerate(points)
        if point.get("cosine_distance_from_previous_6mo") is not None
    ]
    for index, point in distances:
        future = [
            item.get("cosine_distance_from_previous_6mo")
            for item in points[index:]
            if item.get("cosine_distance_from_previous_6mo") is not None
        ]
        if future and max(float(value) for value in future) <= threshold:
            return int(point["months_since_first_role"])
    return None


def build_curve(
    characters: list[dict[str, Any]],
    vectors: dict[int, np.ndarray],
    start_month: int,
    end_month: int,
    weight_fn,
    round_digits: int,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for checkpoint in range(start_month, end_month + 1, 6):
        previous = checkpoint - 6
        current_characters = [row for row in characters if int(row["first_role_month"]) <= checkpoint]
        previous_characters = [row for row in characters if int(row["first_role_month"]) <= previous]
        current_vector, current_norm = cumulative_vector(current_characters, vectors, weight_fn)
        previous_vector, previous_norm = cumulative_vector(previous_characters, vectors, weight_fn)
        overlap = None
        distance = None
        if current_vector.size and previous_vector.size and current_norm > 0.0 and previous_norm > 0.0:
            overlap_value = float(np.clip(current_vector @ previous_vector, -1.0, 1.0))
            overlap = rounded(overlap_value, round_digits)
            distance = rounded(1.0 - overlap_value, round_digits)
        new_characters = [
            row
            for row in characters
            if previous < int(row["first_role_month"]) <= checkpoint
        ]
        points.append(
            {
                "date": month_index_to_date(checkpoint),
                "months_since_first_role": checkpoint - start_month,
                "cumulative_supported_characters": len(current_characters),
                "new_supported_characters": len(new_characters),
                "cumulative_vector_norm_before_normalize": rounded(current_norm, round_digits),
                "overlap_with_previous_6mo": overlap,
                "cosine_distance_from_previous_6mo": distance,
            }
        )
    return points


def main() -> None:
    args = parse_args()
    descriptors = read_basis(args.basis)
    descriptor_index = {descriptor: index for index, descriptor in enumerate(descriptors)}
    basis_set = set(descriptors)
    assignable_set = read_assignable_basis(args.basis)
    canonicalization = read_json(args.canonicalization) if args.canonicalization.exists() else {}
    raw_to_canonical = canonicalization.get("raw_to_canonical") or {}

    character_descriptors, character_meta = load_character_descriptors(
        args.merged_tags,
        args.safe_tags,
        basis_set,
        assignable_set,
        raw_to_canonical,
    )
    bangumi_favorites_by_character = aggregate_bangumi_favorites(
        args.safe_tags,
        args.safe_enrichment,
        args.bangumi_raw_dir,
    )

    embeddings = load_or_create_embeddings(descriptors, args.embedding_cache_dir, args.embedding_model)
    _, _, descriptor_atoms, _ = lowdin_from_global_gram(embeddings.astype(np.float64))

    roles_payload = read_json(args.role_edges)
    roles, excluded_roles = filter_excluded_role_edges(
        roles_payload.get("roles") or [],
        load_role_edge_exclusions(args.role_edge_exclusions),
    )

    role_edge_count_by_character: dict[int, int] = defaultdict(int)
    first_role_by_seiyuu: dict[int, int] = {}
    latest_role_by_seiyuu: dict[int, int] = {}
    seiyuu_by_id: dict[int, dict[str, Any]] = {}
    for role in roles:
        year = role_year(role)
        if year is None:
            continue
        seiyuu = role["seiyuu"]
        seiyuu_id = int(seiyuu["seiyuu_id"])
        character_id = int(role["character"]["character_id"])
        role_edge_count_by_character[character_id] += 1
        seiyuu_by_id[seiyuu_id] = seiyuu
        first_role_by_seiyuu[seiyuu_id] = min(first_role_by_seiyuu.get(seiyuu_id, year), year)
        latest_role_by_seiyuu[seiyuu_id] = max(latest_role_by_seiyuu.get(seiyuu_id, year), year)

    characters_by_seiyuu: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    unique_supported_characters: dict[int, dict[str, Any]] = {}
    for role in roles:
        year = role_year(role)
        if year is None:
            continue
        seiyuu = role["seiyuu"]
        character = role["character"]
        seiyuu_id = int(seiyuu["seiyuu_id"])
        character_id = int(character["character_id"])
        row_descriptors = sorted(descriptor for descriptor in character_descriptors.get(character_id, set()) if descriptor in descriptor_index)
        if not row_descriptors:
            continue

        meta = character_meta.get(character_id) or {}
        anilist_favorites = int(character.get("favourites") or meta.get("favourites") or 0)
        bangumi_favorites = int(bangumi_favorites_by_character.get(character_id) or 0)
        row = {
            "character_id": character_id,
            "name": character.get("name") or meta.get("name") or "",
            "anime": character.get("first_anime") or meta.get("first_anime") or "",
            "anilist_favorites": anilist_favorites,
            "bangumi_favorites": bangumi_favorites,
            "aggregate_favorites": anilist_favorites + bangumi_favorites,
            "favourites": anilist_favorites + bangumi_favorites,
            "image": character.get("image") or meta.get("image") or "",
            "site_url": character.get("site_url") or meta.get("site_url") or "",
            "role_edge_count": role_edge_count_by_character[character_id],
            "descriptors": row_descriptors,
            "first_role_year": year,
            "first_role_month": month_index(year),
        }
        existing = characters_by_seiyuu[seiyuu_id].get(character_id)
        if existing is None:
            characters_by_seiyuu[seiyuu_id][character_id] = row
            unique_supported_characters[character_id] = row
            continue
        existing["first_role_year"] = min(int(existing["first_role_year"]), year)
        existing["first_role_month"] = min(int(existing["first_role_month"]), month_index(year))
        if row["aggregate_favorites"] > int(existing.get("aggregate_favorites") or 0):
            for key in ("anilist_favorites", "bangumi_favorites", "aggregate_favorites", "favourites", "image", "site_url", "anime"):
                existing[key] = row[key]

    character_vectors: dict[int, np.ndarray] = {}
    for character_id, character in unique_supported_characters.items():
        indices = [descriptor_index[descriptor] for descriptor in character.get("descriptors") or []]
        if indices:
            character_vectors[character_id] = np.sum(descriptor_atoms[indices], axis=0)
        else:
            character_vectors[character_id] = np.zeros(descriptor_atoms.shape[1], dtype=np.float64)

    profiles: list[dict[str, Any]] = []
    for seiyuu_id, character_map in characters_by_seiyuu.items():
        seiyuu = seiyuu_by_id.get(seiyuu_id) or {}
        first_year = first_role_by_seiyuu.get(seiyuu_id)
        latest_year = latest_role_by_seiyuu.get(seiyuu_id)
        if first_year is None or latest_year is None:
            continue
        start_month = month_index(first_year)
        end_month = month_index(latest_year) + max(0, args.tail_periods) * 6
        characters = sorted(character_map.values(), key=lambda row: (int(row["first_role_month"]), row["name"]))
        mode_curves: dict[str, Any] = {}
        for mode_key, mode in WEIGHT_MODES.items():
            points = build_curve(
                characters,
                character_vectors,
                start_month,
                end_month,
                mode["weight_fn"],
                args.round_digits,
            )
            valid_distances = [
                float(point["cosine_distance_from_previous_6mo"])
                for point in points
                if point.get("cosine_distance_from_previous_6mo") is not None
            ]
            mode_curves[mode_key] = {
                "points": points,
                "summary": {
                    "max_cosine_distance": rounded(max(valid_distances), args.round_digits) if valid_distances else None,
                    "mean_cosine_distance": rounded(float(np.mean(valid_distances)), args.round_digits) if valid_distances else None,
                    "settled_after_months_distance_lte_0_10": settled_after(points, 0.10),
                    "settled_after_months_distance_lte_0_05": settled_after(points, 0.05),
                },
            }
        profiles.append(
            {
                "seiyuu_id": seiyuu_id,
                "name": seiyuu.get("name") or "",
                "native_name": seiyuu.get("native_name") or "",
                "image": seiyuu.get("image") or "",
                "site_url": seiyuu.get("site_url") or "",
                "role_count": int(seiyuu.get("role_count") or 0),
                "character_count": int(seiyuu.get("character_count") or len(character_map)),
                "supported_character_count": len(character_map),
                "first_role_year": first_year,
                "first_supported_role_year": min(int(row["first_role_year"]) for row in characters),
                "latest_role_year": latest_year,
                "curves": mode_curves,
            }
        )

    profiles.sort(key=lambda row: (row["supported_character_count"], row["role_count"], row["name"]), reverse=True)
    payload = {
        "generated_at": utc_now(),
        "source": "cache_profile_stabilization_curves.py",
        "model": {
            "description": (
                "For every seiyuu, build cumulative role-profile vectors at six-month checkpoints. "
                "Each character vector is the sum of supported production personality descriptor atoms "
                "from the same B @ G @ X/Löwdin orthogonalized basis used by the MVP visualizer. "
                "The curve reports cosine overlap and cosine distance between the cumulative normalized "
                "vector at each checkpoint and the cumulative normalized vector six months earlier."
            ),
            "date_precision": "year",
            "role_event_date_model": "Each role edge is placed at YYYY-01-01 because the current role cache stores years, not exact dates.",
            "tail_periods": args.tail_periods,
            "basis": str(args.basis),
            "role_edges": str(args.role_edges),
            "role_edge_exclusions": str(args.role_edge_exclusions),
            "excluded_role_edge_count": len(excluded_roles),
            "descriptor_count": len(descriptors),
            "orthogonal_rank": int(descriptor_atoms.shape[1]),
            "modes": {
                key: {
                    "label": value["label"],
                    "description": value["description"],
                    "row_weight": value["row_weight"],
                }
                for key, value in WEIGHT_MODES.items()
            },
            "settling_summary": "settled_after_months is the first checkpoint after which all future six-month cosine distances stay under the threshold.",
        },
        "profiles": profiles,
    }
    write_json(args.output, payload)
    print(f"wrote {args.output} with {len(profiles)} seiyuu")


if __name__ == "__main__":
    main()
