#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cache globally character-centered seiyuu profile diagnostics. The global "
            "background is the normalized weighted sum of unique character rows in the "
            "same B @ G @ X descriptor space. Each seiyuu profile vector subtracts that "
            "background and is renormalized before descriptor fitting."
        )
    )
    parser.add_argument("--profiles", type=Path, default=Path("site/profiles.json"))
    parser.add_argument("--profile-root", type=Path, default=Path("site"))
    parser.add_argument("--output", type=Path, default=Path("site/seiyuu_global_centered_profiles.json"))
    parser.add_argument("--builder", type=Path, default=Path("scripts/build_redundant_svd_site.py"))
    parser.add_argument("--map-builder", type=Path, default=Path("scripts/build_seiyuu_descriptor_map.py"))
    parser.add_argument("--fit-target", type=float, default=0.85)
    parser.add_argument("--min-fit-terms", type=int, default=1)
    parser.add_argument("--max-fit-terms", type=int, default=200)
    parser.add_argument("--round-digits", type=int, default=6)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1.0e-12)


def rounded(value: float, digits: int) -> float:
    return round(float(value), digits)


def character_key(character: dict) -> str:
    character_id = character.get("character_id")
    if character_id is not None:
        return str(character_id)
    return f"{character.get('name') or ''}|{character.get('anime') or ''}"


def descriptor_support_rows(builder, character_labels: list[dict], *, digits: int) -> list[dict]:
    rows = []
    for descriptor, values in builder.descriptor_weighted_support(character_labels).items():
        rows.append(
            {
                "descriptor": descriptor,
                "support": int(values.get("support") or 0),
                "weighted_support": rounded(float(values.get("weighted_support") or 0.0), digits),
                "combined_favourites": int(values.get("combined_favourites") or 0),
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["weighted_support"]),
            int(row["support"]),
            int(row["combined_favourites"]),
            row["descriptor"],
        ),
        reverse=True,
    )
    return rows


def character_projection_rows(
    matrix: np.ndarray,
    character_labels: list[dict],
    target: np.ndarray,
    *,
    digits: int,
) -> list[dict]:
    rows = []
    for weighted_row, character in zip(matrix, character_labels, strict=True):
        row_weight = float(character.get("row_weight") or 1.0)
        raw_row = weighted_row / max(row_weight, 1.0e-12)
        raw_unit = normalize(raw_row)
        rows.append(
            {
                "character_id": character.get("character_id"),
                "name": character.get("name") or "",
                "anime": character.get("anime") or "",
                "image": character.get("image") or "",
                "site_url": character.get("site_url") or "",
                "combined_favourites": int(character.get("combined_favourites") or character.get("favourites") or 0),
                "role_edge_count": int(character.get("role_edge_count") or 1),
                "row_weight": rounded(row_weight, digits),
                "unweighted_row_norm": rounded(float(np.linalg.norm(raw_row)), digits),
                "projection": rounded(float(raw_unit @ target), digits),
                "descriptors": character.get("descriptors") or [],
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["projection"]),
            int(row["combined_favourites"]),
            row["name"],
        ),
        reverse=True,
    )
    return rows


def supported_fit(
    builder,
    target: np.ndarray,
    descriptor_atoms: np.ndarray,
    descriptors: list[str],
    descriptor_index: dict[str, int],
    character_labels: list[dict],
    *,
    fit_target: float,
    min_fit_terms: int,
    max_fit_terms: int,
) -> dict:
    support = builder.descriptor_weighted_support(character_labels)
    supported = [descriptor for descriptor in descriptors if descriptor in support]
    supported_atoms = descriptor_atoms[[descriptor_index[descriptor] for descriptor in supported]]
    return builder.decode_axis_by_weighted_support(
        target,
        supported_atoms,
        supported,
        character_labels,
        stop_fit=fit_target,
        min_terms=min_fit_terms,
        max_terms=max_fit_terms,
    )


def main() -> None:
    args = parse_args()
    builder = load_module("redundant_svd_builder", args.builder)
    map_builder = load_module("seiyuu_descriptor_map_builder", args.map_builder)
    profiles_payload = read_json(args.profiles)
    descriptors, descriptor_index, descriptor_atoms = map_builder.descriptor_basis(builder, profiles_payload)

    params = profiles_payload.get("parameters") or {}
    shared_role_weight = params.get("shared_role_weight") or "inverse_sqrt"
    row_weight = params.get("row_weight") or "none"

    loaded_profiles = []
    unique_characters = {}
    for profile in profiles_payload.get("profiles", []):
        profile_path = args.profile_root / (profile.get("profile_path") or "")
        if not profile_path.exists():
            continue
        profile_payload = read_json(profile_path)
        characters = profile_payload.get("major_lane", {}).get("characters") or []
        loaded_profiles.append((profile, characters))
        for character in characters:
            unique_characters.setdefault(character_key(character), character)

    global_matrix, global_labels = builder.profile_matrix(
        list(unique_characters.values()),
        descriptor_index,
        descriptor_atoms,
        shared_role_weight,
        row_weight,
        False,
    )
    global_sum = np.sum(global_matrix, axis=0)
    global_vector = normalize(global_sum)

    profile_rows = []
    for profile, characters in loaded_profiles:
        matrix, character_labels = builder.profile_matrix(
            characters,
            descriptor_index,
            descriptor_atoms,
            shared_role_weight,
            row_weight,
            False,
        )
        if matrix.shape[0] == 0:
            continue

        raw_sum = np.sum(matrix, axis=0)
        uncentered_target = normalize(raw_sum)
        centered_raw = uncentered_target - global_vector
        centered_target = normalize(centered_raw)
        fit = supported_fit(
            builder,
            centered_target,
            descriptor_atoms,
            descriptors,
            descriptor_index,
            character_labels,
            fit_target=args.fit_target,
            min_fit_terms=args.min_fit_terms,
            max_fit_terms=args.max_fit_terms,
        )

        profile_rows.append(
            {
                "seiyuu_id": profile.get("seiyuu_id"),
                "name": profile.get("name") or "",
                "native_name": profile.get("native_name") or "",
                "image": profile.get("image") or "",
                "site_url": profile.get("site_url") or "",
                "role_count": int(profile.get("role_count") or 0),
                "character_count": int(profile.get("character_count") or len(character_labels)),
                "supported_character_count": len(character_labels),
                "first_year": profile.get("first_year"),
                "global_centered_norm_before_renormalization": rounded(float(np.linalg.norm(centered_raw)), args.round_digits),
                "descriptor_fit": fit,
                "descriptor_support": descriptor_support_rows(builder, character_labels, digits=args.round_digits),
                "character_projections": character_projection_rows(
                    matrix,
                    character_labels,
                    centered_target,
                    digits=args.round_digits,
                ),
            }
        )

    profile_rows.sort(key=lambda row: (row["character_count"], row["role_count"], row["name"]), reverse=True)
    payload = {
        "source": "cache_global_character_centered_profiles.py",
        "model": {
            "description": (
                "For each seiyuu, first form the normalized weighted sum vector from unnormalized B @ G @ X "
                "character rows. Then subtract the normalized global weighted sum over unique characters "
                "and renormalize. Positive descriptor labels are fitted only from descriptors supported "
                "by that seiyuu's characters, added by conservative weighted support."
            ),
            "descriptor_count": len(descriptors),
            "orthogonal_rank": int(descriptor_atoms.shape[1]),
            "global_character_count": len(unique_characters),
            "global_supported_character_count": len(global_labels),
            "global_weighted_sum_norm": rounded(float(np.linalg.norm(global_sum)), args.round_digits),
            "shared_role_weight": shared_role_weight,
            "row_weight": row_weight,
            "normalize_character_rows_before_sum": False,
            "global_centering": "subtract_normalized_unique_character_weighted_sum",
            "renormalize_after_global_centering": True,
            "character_projection_row_normalization": True,
            "descriptor_fit_order": "weighted_support",
            "descriptor_fit_target": args.fit_target,
            "min_fit_terms": args.min_fit_terms,
            "max_fit_terms": args.max_fit_terms,
        },
        "descriptors": descriptors,
        "profiles": profile_rows,
    }
    write_json(args.output, payload)
    print(f"wrote {args.output} with {len(profile_rows)} profiles")


if __name__ == "__main__":
    main()
