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
            "Cache the current seiyuu profiling diagnostic: a normalized weighted sum "
            "of unnormalized B @ G @ X character rows, normalized character-row "
            "projections onto that sum, and support-ordered positive descriptor fits."
        )
    )
    parser.add_argument("--profiles", type=Path, default=Path("site/profiles.json"))
    parser.add_argument("--profile-root", type=Path, default=Path("site"))
    parser.add_argument("--output", type=Path, default=Path("site/seiyuu_weighted_sum_profiles.json"))
    parser.add_argument("--builder", type=Path, default=Path("scripts/build_redundant_svd_site.py"))
    parser.add_argument("--map-builder", type=Path, default=Path("scripts/build_seiyuu_descriptor_map.py"))
    parser.add_argument("--fit-target", type=float, default=0.85)
    parser.add_argument("--min-fit-terms", type=int, default=1)
    parser.add_argument("--max-fit-terms", type=int, default=30)
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


def rounded(value: float, digits: int) -> float:
    return round(float(value), digits)


def weighted_sum_vector(matrix: np.ndarray) -> np.ndarray:
    vector = np.sum(matrix, axis=0)
    return vector / max(float(np.linalg.norm(vector)), 1.0e-12)


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
        raw_norm = float(np.linalg.norm(raw_row))
        raw_unit = raw_row / max(raw_norm, 1.0e-12)
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
                "unweighted_row_norm": rounded(raw_norm, digits),
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


def descriptor_support_rows(builder, character_labels: list[dict], *, digits: int) -> list[dict]:
    support = builder.descriptor_weighted_support(character_labels)
    rows = []
    for descriptor, values in support.items():
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

    profile_rows = []
    for profile in profiles_payload.get("profiles", []):
        profile_path = args.profile_root / (profile.get("profile_path") or "")
        if not profile_path.exists():
            continue
        payload = read_json(profile_path)
        characters = payload.get("major_lane", {}).get("characters") or []
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

        target = weighted_sum_vector(matrix)
        fit = supported_fit(
            builder,
            target,
            descriptor_atoms,
            descriptors,
            descriptor_index,
            character_labels,
            fit_target=args.fit_target,
            min_fit_terms=args.min_fit_terms,
            max_fit_terms=args.max_fit_terms,
        )
        support_rows = descriptor_support_rows(builder, character_labels, digits=args.round_digits)

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
                "descriptor_fit": fit,
                "descriptor_support": support_rows,
                "character_projections": character_projection_rows(
                    matrix,
                    character_labels,
                    target,
                    digits=args.round_digits,
                ),
            }
        )

    profile_rows.sort(key=lambda row: (row["character_count"], row["role_count"], row["name"]), reverse=True)
    payload = {
        "source": "cache_weighted_sum_profile_diagnostics.py",
        "model": {
            "description": (
                "For each seiyuu, form the normalized weighted sum vector from unnormalized B @ G @ X "
                "character rows. Character projections are dot products from individually normalized "
                "unnormalized character rows to that vector. Descriptor labels are positive fits using "
                "supported descriptors added by conservative weighted support until the target fit is reached."
            ),
            "descriptor_count": len(descriptors),
            "orthogonal_rank": int(descriptor_atoms.shape[1]),
            "shared_role_weight": shared_role_weight,
            "row_weight": row_weight,
            "normalize_character_rows_before_sum": False,
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
