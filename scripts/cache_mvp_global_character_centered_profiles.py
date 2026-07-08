#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
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

from analyze_sv1_diffusivity import candidate_descriptor_values, lowdin_from_global_gram  # noqa: E402
from role_edge_exclusions import DEFAULT_EXCLUSIONS_PATH, filter_excluded_role_edges, load_role_edge_exclusions  # noqa: E402
from seiyuu_local_nmf_lane_svd import load_or_create_embeddings  # noqa: E402


DEFAULT_SAFE_TAGS = [
    Path("data/external/safe_enrichment_llm/character_tags.jsonl"),
    Path(
        "run/gpu_llm_tagging/a100_batch_transformers_prod_20260701/"
        "batch_transformers_prod_complete/character_tags_deduped_aggressive.jsonl"
    ),
    Path("run/gpu_llm_tagging/returned_latest/character_tags.jsonl"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cache the current MVP seiyuu profile representation: production personality descriptors, "
            "B @ G @ X smoothing, favorites-weighted character weighting, and global character centering."
        )
    )
    parser.add_argument("--basis", type=Path, default=Path("run/production_personality_basis/production_personality_basis_kept.tsv"))
    parser.add_argument("--role-edges", type=Path, default=Path("data/role_edges_current_seiyuu_expanded.json"))
    parser.add_argument("--role-edge-exclusions", type=Path, default=DEFAULT_EXCLUSIONS_PATH)
    parser.add_argument("--merged-tags", type=Path, default=Path("data/external/merged/all_characters_llm_vndb_personality_tags.json"))
    parser.add_argument("--safe-tags", nargs="*", type=Path, default=DEFAULT_SAFE_TAGS)
    parser.add_argument("--canonicalization", type=Path, default=Path("models/global_descriptor_canonicalization/descriptor_canonicalization.json"))
    parser.add_argument("--embedding-cache-dir", type=Path, default=Path("run/production_personality_basis/embeddings"))
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--output", type=Path, default=Path("run/mvp_global_character_centered_profiles.json"))
    parser.add_argument("--min-characters", type=int, default=1)
    parser.add_argument("--round-digits", type=int, default=6)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    tmp.replace(path)


def normalize_tag(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"[_/]+", " ", value)
    value = re.sub(r"[^a-z0-9\- ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def read_basis(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row["descriptor"] for row in csv.DictReader(handle, dialect="excel-tab")]


def read_assignable_basis(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["descriptor"]
            for row in csv.DictReader(handle, dialect="excel-tab")
            if int(row.get("character_count") or 0) > 0
        }


def candidate_descriptors(
    raw_value: str,
    basis_set: set[str],
    raw_to_canonical: dict[str, str],
    assignable_set: set[str] | None = None,
) -> list[str]:
    values = []
    for candidate in candidate_descriptor_values(raw_value):
        values.append(candidate)
        values.append(raw_to_canonical.get(candidate, candidate))
    normalized = normalize_tag(raw_value)
    if normalized:
        values.append(normalized)
        values.append(raw_to_canonical.get(normalized, normalized))
    for part in re.split(r"\b(?:and|or|but)\b|[,;/]", normalized):
        part = part.strip()
        if part:
            values.append(part)
            values.append(raw_to_canonical.get(part, part))

    output = []
    for value in values:
        value = normalize_tag(value)
        if value in basis_set and (assignable_set is None or value in assignable_set) and value not in output:
            output.append(value)
    return output


def tag_value(entry: dict) -> str:
    return str(entry.get("tag") or "").strip()


def add_tag_descriptors(
    character_descriptors: dict[int, set[str]],
    character_meta: dict[int, dict],
    character_id: int,
    payload: dict,
    *,
    basis_set: set[str],
    assignable_set: set[str] | None,
    raw_to_canonical: dict[str, str],
) -> None:
    character_meta.setdefault(character_id, payload)
    tags = payload.get("llm_tags") or payload.get("tags") or {}
    for category in ("personality", "traits"):
        for tag in tags.get(category) or []:
            for descriptor in candidate_descriptors(tag_value(tag), basis_set, raw_to_canonical, assignable_set):
                character_descriptors[character_id].add(descriptor)


def load_character_descriptors(
    merged_tags: Path,
    safe_tags: list[Path],
    basis_set: set[str],
    assignable_set: set[str] | None,
    raw_to_canonical: dict[str, str],
) -> tuple[dict[int, set[str]], dict[int, dict]]:
    character_descriptors: dict[int, set[str]] = defaultdict(set)
    character_meta: dict[int, dict] = {}

    merged_payload = read_json(merged_tags)
    for character in merged_payload.get("characters") or []:
        character_id = int(character["anilist_character_id"])
        add_tag_descriptors(
            character_descriptors,
            character_meta,
            character_id,
            character,
            basis_set=basis_set,
            assignable_set=assignable_set,
            raw_to_canonical=raw_to_canonical,
        )

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
                add_tag_descriptors(
                    character_descriptors,
                    character_meta,
                    int(character_id),
                    payload,
                    basis_set=basis_set,
                    assignable_set=assignable_set,
                    raw_to_canonical=raw_to_canonical,
                )

    return character_descriptors, character_meta


def favorites_weighted_row_weight(character: dict) -> float:
    favourites = max(
        float(
            character.get("aggregate_favorites")
            or character.get("aggregate_favourites")
            or character.get("favourites")
            or 0.0
        ),
        0.0,
    )
    role_edges = max(int(character.get("role_edge_count") or 1), 1)
    return math.sqrt(math.log1p(favourites) + 1.0) / math.sqrt(role_edges)


def normalize(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1.0e-12)


def rounded(value: float, digits: int) -> float:
    return round(float(value), digits)


def top_roles(characters: list[dict], limit: int = 5) -> list[dict]:
    rows = sorted(
        characters,
        key=lambda row: (
            int(row.get("aggregate_favorites") or row.get("favourites") or 0),
            row.get("name") or "",
        ),
        reverse=True,
    )
    return [
        {
            "character_id": row["character_id"],
            "name": row.get("name") or "",
            "anime": row.get("anime") or "",
            "favourites": int(row.get("aggregate_favorites") or row.get("favourites") or 0),
            "image": row.get("image") or "",
            "site_url": row.get("site_url") or "",
        }
        for row in rows[:limit]
    ]


def descriptor_support(characters: list[dict], *, digits: int, weight_fn=favorites_weighted_row_weight) -> list[dict]:
    rows: dict[str, dict] = {}
    for character in characters:
        weight = weight_fn(character)
        for descriptor in character.get("descriptors") or []:
            row = rows.setdefault(
                descriptor,
                {"descriptor": descriptor, "characters": 0, "weighted_support": 0.0, "favourites": 0},
            )
            row["characters"] += 1
            row["weighted_support"] += weight
            row["favourites"] += int(character.get("aggregate_favorites") or character.get("favourites") or 0)
    output = [
        {
            "descriptor": row["descriptor"],
            "characters": int(row["characters"]),
            "weighted_support": rounded(row["weighted_support"], digits),
            "favourites": int(row["favourites"]),
        }
        for row in rows.values()
    ]
    output.sort(key=lambda row: (row["weighted_support"], row["characters"], row["favourites"], row["descriptor"]), reverse=True)
    return output


def main() -> None:
    args = parse_args()
    descriptors = read_basis(args.basis)
    descriptor_index = {descriptor: index for index, descriptor in enumerate(descriptors)}
    basis_set = set(descriptors)
    assignable_set = read_assignable_basis(args.basis)
    canonicalization = read_json(args.canonicalization)
    raw_to_canonical = canonicalization.get("raw_to_canonical") or {}

    character_descriptors, character_meta = load_character_descriptors(
        args.merged_tags,
        args.safe_tags,
        basis_set,
        assignable_set,
        raw_to_canonical,
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

    seiyuu_by_id: dict[int, dict] = {}
    characters_by_seiyuu: dict[int, dict[int, dict]] = defaultdict(dict)
    unique_characters: dict[int, dict] = {}

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
        favourites = int(character.get("favourites") or meta.get("favourites") or 0)
        row = {
            "character_id": character_id,
            "name": character.get("name") or meta.get("name") or "",
            "anime": character.get("first_anime") or meta.get("first_anime") or "",
            "favourites": favourites,
            "image": character.get("image") or meta.get("image") or "",
            "site_url": character.get("site_url") or meta.get("site_url") or "",
            "role_edge_count": role_edge_count_by_character[character_id],
            "descriptors": row_descriptors,
        }
        existing = characters_by_seiyuu[seiyuu_id].get(character_id)
        if existing is None or favourites > int(existing.get("favourites") or 0):
            characters_by_seiyuu[seiyuu_id][character_id] = row
            unique_characters[character_id] = row

    def character_vector(character: dict) -> np.ndarray:
        indices = [descriptor_index[descriptor] for descriptor in character.get("descriptors") or []]
        if not indices:
            return np.zeros(descriptor_atoms.shape[1], dtype=np.float64)
        return np.sum(descriptor_atoms[indices], axis=0)

    global_sum = np.zeros(descriptor_atoms.shape[1], dtype=np.float64)
    for character in unique_characters.values():
        global_sum += favorites_weighted_row_weight(character) * character_vector(character)
    global_character_vector = normalize(global_sum)

    profiles = []
    for seiyuu_id, character_map in characters_by_seiyuu.items():
        characters = list(character_map.values())
        if len(characters) < args.min_characters:
            continue
        weighted_sum = np.zeros(descriptor_atoms.shape[1], dtype=np.float64)
        for character in characters:
            weighted_sum += favorites_weighted_row_weight(character) * character_vector(character)
        uncentered_vector = normalize(weighted_sum)
        centered_vector = normalize(uncentered_vector - global_character_vector)
        descriptor_scores = descriptor_atoms @ centered_vector
        seiyuu = seiyuu_by_id[seiyuu_id]
        profiles.append(
            {
                "seiyuu_id": seiyuu_id,
                "name": seiyuu.get("name") or "",
                "native_name": seiyuu.get("native_name") or "",
                "image": seiyuu.get("image") or "",
                "site_url": seiyuu.get("site_url") or "",
                "role_count": int(seiyuu.get("role_count") or 0),
                "character_count": int(seiyuu.get("character_count") or len(characters)),
                "supported_character_count": len(characters),
                "first_year": seiyuu.get("first_year"),
                "uncentered_norm": rounded(float(np.linalg.norm(weighted_sum)), args.round_digits),
                "centered_delta_norm": rounded(float(np.linalg.norm(uncentered_vector - global_character_vector)), args.round_digits),
                "descriptor_scores": np.round(descriptor_scores, args.round_digits).tolist(),
                "descriptor_support": descriptor_support(characters, digits=args.round_digits),
                "notable_roles": top_roles(characters),
            }
        )

    profiles.sort(key=lambda row: (row["supported_character_count"], row["role_count"], row["name"]), reverse=True)
    payload = {
        "generated_at": utc_now(),
        "source": "cache_mvp_global_character_centered_profiles.py",
        "model": {
            "description": (
                "MVP Scheme 1 with global character centering: for each character, sum supported production "
                "descriptor atoms after B @ G @ X. Weight rows by sqrt(log1p(AniList favourites)+1) and "
                "inverse sqrt(shared role edges). Sum rows per seiyuu and normalize. Build the global "
                "background by the same weighted sum over unique supported characters, normalize it, subtract "
                "it from each seiyuu vector, then renormalize. Descriptor scores are dot products against "
                "the same orthogonalized descriptor atoms."
            ),
            "basis": str(args.basis),
            "role_edges": str(args.role_edges),
            "role_edge_exclusions": str(args.role_edge_exclusions),
            "excluded_role_edge_count": len(excluded_roles),
            "merged_tags": str(args.merged_tags),
            "safe_tags": [str(path) for path in args.safe_tags],
            "canonicalization": str(args.canonicalization),
            "embedding_model": args.embedding_model,
            "descriptor_count": len(descriptors),
            "orthogonal_rank": int(descriptor_atoms.shape[1]),
            "global_supported_character_count": len(unique_characters),
            "global_weighted_sum_norm": rounded(float(np.linalg.norm(global_sum)), args.round_digits),
            "row_weight": "sqrt(log1p(AniList favourites)+1) / sqrt(role_edge_count)",
            "global_centering": "subtract_normalized_global_unique_character_weighted_sum",
            "renormalize_after_centering": True,
        },
        "descriptors": descriptors,
        "profiles": profiles,
    }
    write_json(args.output, payload)
    print(f"wrote {args.output} with {len(profiles)} profiles, {len(descriptors)} descriptors")


if __name__ == "__main__":
    main()
