#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build seiyuu-level SV1 descriptor map payload.")
    parser.add_argument("--profiles", type=Path, default=Path("site/profiles.json"))
    parser.add_argument("--profile-root", type=Path, default=Path("site"))
    parser.add_argument("--output", type=Path, default=Path("site/seiyuu_descriptor_map.json"))
    parser.add_argument("--builder", type=Path, default=Path("scripts/build_redundant_svd_site.py"))
    parser.add_argument("--round-digits", type=int, default=5)
    parser.add_argument("--max-notable-roles", type=int, default=5)
    parser.add_argument(
        "--representation",
        choices=["sv1", "midpoint"],
        default="sv1",
        help="Use the first SVD descriptor vector, or the normalized midpoint of character rows.",
    )
    parser.add_argument(
        "--normalize-character-rows",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Normalize each character descriptor row before seiyuu SVD. Defaults off for this comparison map.",
    )
    return parser.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_builder(path: Path):
    spec = importlib.util.spec_from_file_location("redundant_svd_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load builder module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def descriptor_basis(builder, profiles_payload: dict) -> tuple[list[str], dict[str, int], np.ndarray]:
    params = profiles_payload.get("parameters") or {}
    descriptor_union = Path(params.get("descriptor_union") or "run/adjectival_personality_union/adjectival_personality_union.json")
    contextual_path = Path(params.get("contextual_personality_scores") or "models/contextual_personality_anchor_scores/contextual_personality_anchor_scores.json")
    embedding_path = Path(params.get("embedding_npz") or "models/adjectival_personality_nmf/adjectival_personality_embeddings_baai_bge-small-en-v1.5.npz")
    descriptor_shape = params.get("descriptor_shape") or "single_word_or_hyphenated"
    contextual_filter = params.get("contextual_personality_filter") or {}
    min_score = float(contextual_filter.get("min_score") if contextual_filter.get("min_score") is not None else 0.005)
    min_count = int(contextual_filter.get("min_descriptor_character_count") or 2)

    union_payload = read_json(descriptor_union)
    contextual_payload = read_json(contextual_path) if contextual_path.exists() else {}
    contextual_scores = {
        str(descriptor): row
        for descriptor, row in (contextual_payload.get("scores_by_descriptor") or {}).items()
    }
    descriptor_rows = union_payload["descriptors"]
    base_mask = [
        builder.descriptor_shape_ok(row["tag"], descriptor_shape)
        and builder.descriptor_context_ok(row["tag"], contextual_scores, min_score, min_count)
        for row in descriptor_rows
    ]
    descriptor_mask = builder.remove_unhyphenated_duplicates(descriptor_rows, base_mask)
    descriptors = [row["tag"] for row, keep in zip(descriptor_rows, descriptor_mask, strict=True) if keep]
    descriptor_index = {descriptor: index for index, descriptor in enumerate(descriptors)}

    embeddings = np.load(embedding_path)["embeddings"].astype(np.float64)
    embeddings = embeddings[np.asarray(descriptor_mask, dtype=bool)]
    embeddings = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1.0e-12)
    gram = embeddings @ embeddings.T
    eigenvalues, eigenvectors = np.linalg.eigh((gram + gram.T) * 0.5)
    keep = eigenvalues > gram.shape[0] * np.finfo(np.float64).eps * max(float(eigenvalues.max()), 1.0) * 100.0
    descriptor_atoms = eigenvectors[:, keep] * np.sqrt(eigenvalues[keep])
    descriptor_atoms = descriptor_atoms / np.maximum(np.linalg.norm(descriptor_atoms, axis=1, keepdims=True), 1.0e-12)
    return descriptors, descriptor_index, descriptor_atoms


def notable_roles(characters: list[dict], max_roles: int) -> list[dict]:
    ranked = sorted(
        characters,
        key=lambda row: (
            int(row.get("combined_favourites") or row.get("favourites") or 0),
            int(row.get("favourites") or 0),
            row.get("name") or "",
        ),
        reverse=True,
    )
    return [
        {
            "character_id": row.get("character_id"),
            "name": row.get("name") or "",
            "anime": row.get("anime") or "",
            "image": row.get("image") or "",
            "site_url": row.get("site_url") or "",
            "favourites": int(row.get("combined_favourites") or row.get("favourites") or 0),
        }
        for row in ranked[:max_roles]
    ]


def main() -> None:
    args = parse_args()
    builder = load_builder(args.builder)
    profiles_payload = read_json(args.profiles)
    descriptors, descriptor_index, descriptor_atoms = descriptor_basis(builder, profiles_payload)

    params = profiles_payload.get("parameters") or {}
    shared_role_weight = params.get("shared_role_weight") or "inverse_sqrt"
    row_weight = params.get("row_weight") or "none"
    normalize_character_rows = bool(args.normalize_character_rows)

    rows = []
    for profile in profiles_payload.get("profiles", []):
        payload_path = args.profile_root / (profile.get("profile_path") or "")
        if not payload_path.exists():
            continue
        payload = read_json(payload_path)
        characters = payload.get("major_lane", {}).get("characters") or []
        matrix, character_labels = builder.profile_matrix(
            characters,
            descriptor_index,
            descriptor_atoms,
            shared_role_weight,
            row_weight,
            normalize_character_rows,
        )
        if matrix.shape[0] == 0:
            continue
        singular_values = np.zeros(0, dtype=np.float64)
        total_mass = 0.0
        if args.representation == "sv1":
            left, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
            _, vector = builder.orient_sv1(left[:, 0].copy(), vt[0].copy(), character_labels)
            singular_mass = singular_values * singular_values
            total_mass = float(np.sum(singular_mass))
        else:
            vector = np.sum(matrix, axis=0)
            vector = vector / max(float(np.linalg.norm(vector)), 1.0e-12)
        descriptor_scores = descriptor_atoms @ vector
        rows.append(
            {
                "seiyuu_id": profile.get("seiyuu_id"),
                "name": profile.get("name") or "",
                "native_name": profile.get("native_name") or "",
                "image": profile.get("image") or "",
                "site_url": profile.get("site_url") or "",
                "role_count": int(profile.get("role_count") or 0),
                "character_count": int(profile.get("character_count") or len(character_labels)),
                "first_year": profile.get("first_year"),
                "sv1_mass_percent": round(float(singular_mass[0] / total_mass * 100.0), 6) if total_mass > 0.0 else 0.0,
                "representation_norm": round(float(np.linalg.norm(vector)), 8),
                "descriptor_scores": np.round(descriptor_scores, args.round_digits).tolist(),
                "notable_roles": notable_roles(character_labels, args.max_notable_roles),
            }
        )

    rows.sort(key=lambda row: (row["character_count"], row["role_count"], row["name"]), reverse=True)
    if args.representation == "sv1":
        representation_description = (
            "Every seiyuu is represented by the oriented first right singular vector of their uncentered character x "
            "orthogonalized descriptor matrix."
        )
    elif normalize_character_rows:
        representation_description = (
            "Every seiyuu is represented by the normalized midpoint of their normalized character rows after B @ G @ X."
        )
    else:
        representation_description = (
            "Every seiyuu is represented by the normalized weighted sum of their unnormalized character rows after B @ G @ X."
        )

    payload = {
        "source": "build_seiyuu_descriptor_map.py",
        "model": {
            "description": representation_description
            + " Descriptor query scores are dot products against the same orthogonalized descriptor basis.",
            "descriptor_count": len(descriptors),
            "orthogonal_rank": int(descriptor_atoms.shape[1]),
            "representation": args.representation,
            "shared_role_weight": shared_role_weight,
            "row_weight": row_weight,
            "normalize_character_rows": normalize_character_rows,
        },
        "descriptors": descriptors,
        "seiyuu": rows,
    }
    write_json(args.output, payload)
    print(f"wrote {args.output} with {len(rows)} seiyuu and {len(descriptors)} descriptors")


if __name__ == "__main__":
    main()
