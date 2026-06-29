#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a seiyuu-local B @ G @ X SVD descriptor experiment.")
    parser.add_argument("--seiyuu", required=True)
    parser.add_argument("--tags-input", type=Path, default=Path("data/external/merged/all_characters_llm_vndb_personality_tags.json"))
    parser.add_argument("--matrix-metadata", type=Path, default=Path("models/tag_descriptor_matrices/all_characters_llm_vndb_personality_tags_matrix_metadata.json"))
    parser.add_argument("--embeddings", type=Path, default=Path("models/tag_descriptor_matrices/descriptor_embeddings_baai_bge_small_en_v1_5.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/seiyuu_local_orthogonal_svd"))
    parser.add_argument("--categories", nargs="+", default=["personality", "traits"])
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--component-mass", type=float, default=0.99)
    parser.add_argument("--eigenvalue-tolerance-scale", type=float, default=100.0)
    parser.add_argument(
        "--center-embeddings",
        choices=["none", "descriptor", "population"],
        default="none",
        help=(
            "Center local descriptor embeddings before forming G. "
            "'descriptor' uses the unique local descriptor mean; "
            "'population' weights descriptors by their local character incidence."
        ),
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "value"


def norm_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def name_keys(value: str) -> set[str]:
    normalized = norm_name(value)
    keys = {normalized}
    parts = [part for part in normalized.split() if part]
    if len(parts) >= 2:
        keys.add(" ".join(reversed(parts)))
    return {key for key in keys if key}


METADATA_ONLY_EVIDENCE_RE = re.compile(
    r"^\s*(?:_+)?\s*"
    r"(?:age|birthday|blood\s*type|classification|gender|grade|height|occupation|race|rank|species|status|weight)"
    r"\s*:?\s*(?:_+)?",
    re.IGNORECASE,
)


def tag_supported_only_by_metadata_field(tag: dict) -> bool:
    evidence_values = tag.get("evidence") or tag.get("evidences") or []
    if isinstance(evidence_values, str):
        evidence_values = [evidence_values]
    evidence_values = [str(value).strip() for value in evidence_values if str(value).strip()]
    return bool(evidence_values) and all(METADATA_ONLY_EVIDENCE_RE.match(value) for value in evidence_values)


def finite_verb_like_descriptor_head(value: str) -> bool:
    tokens = re.findall(r"[a-z][a-z'-]*", value.lower())
    if len(tokens) < 2:
        return False
    head = tokens[0]
    if not (head.endswith("s") or head.endswith("ed")):
        return False
    try:
        from nltk.corpus import wordnet as wn

        return bool(wn.morphy(head, wn.VERB))
    except LookupError:
        return False


def has_possessive_descriptor_token(value: str) -> bool:
    return any(token.endswith("'s") or token.endswith("’s") for token in re.findall(r"[a-z][a-z'’-]*", value.lower()))


def character_descriptors(character: dict, categories: set[str]) -> list[str]:
    descriptors = []
    for category in categories:
        for tag in character.get("llm_tags", {}).get(category, []):
            if category in {"personality", "traits"} and tag_supported_only_by_metadata_field(tag):
                continue
            value = str(tag.get("tag") or "").strip().lower()
            if value and not finite_verb_like_descriptor_head(value) and not has_possessive_descriptor_token(value):
                descriptors.append(value)
    return sorted(set(descriptors))


def character_payload(character: dict, amplitude: float | None = None) -> dict:
    payload = {
        "character_id": int(character["anilist_character_id"]),
        "name": character.get("name") or "",
        "first_anime": character.get("first_anime") or "",
        "favourites": int(character.get("favourites") or 0),
        "site_url": character.get("site_url") or "",
        "image": character.get("image") or "",
        "descriptors": character["_descriptors"],
    }
    if amplitude is not None:
        payload["amplitude"] = round(float(amplitude), 10)
        payload["abs_amplitude"] = round(abs(float(amplitude)), 10)
    return payload


def component_list(values: np.ndarray, labels: list[str], cutoff: float) -> list[dict]:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(np.abs(values))[::-1]
    norm2 = float(np.sum(values * values))
    cumulative = 0.0
    output = []
    for index in order:
        value = float(values[int(index)])
        share = value * value / norm2 if norm2 > 0 else 0.0
        cumulative += share
        output.append(
            {
                "index": int(index),
                "label": labels[int(index)],
                "amplitude": round(value, 10),
                "abs_amplitude": round(abs(value), 10),
                "l2_mass_share": round(share, 10),
                "cumulative_l2_mass": round(cumulative, 10),
            }
        )
        if cumulative >= cutoff:
            break
    return output


def unit_vector(values: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if norm <= 0:
        return values.copy(), norm
    return values / norm, norm


def orient_pair(u: np.ndarray, v: np.ndarray, G: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    population_descriptor_loading = G @ X @ v
    pivot = int(np.argmax(np.abs(population_descriptor_loading)))
    if population_descriptor_loading[pivot] < 0:
        return -u, -v
    return u, v


def seiyuu_characters(payload: dict, seiyuu: str, categories: set[str]) -> list[dict]:
    requested = name_keys(seiyuu)
    output = []
    for character in payload.get("characters", []):
        if not any(requested.intersection(name_keys(row.get("name") or "")) for row in character.get("seiyuu", [])):
            continue
        row = dict(character)
        row["_descriptors"] = character_descriptors(row, categories)
        output.append(row)
    return output


def main() -> None:
    args = parse_args()
    categories = {category.strip() for category in args.categories}
    tag_payload = read_json(args.tags_input)
    all_characters = seiyuu_characters(tag_payload, args.seiyuu, categories)
    rows = [character for character in all_characters if character["_descriptors"]]
    if not rows:
        raise RuntimeError(f"No descriptor-bearing characters found for {args.seiyuu!r}.")

    descriptors = sorted({descriptor for character in rows for descriptor in character["_descriptors"]})
    descriptor_index = {descriptor: index for index, descriptor in enumerate(descriptors)}
    B = np.zeros((len(rows), len(descriptors)), dtype=np.float64)
    for row_index, character in enumerate(rows):
        for descriptor in character["_descriptors"]:
            B[row_index, descriptor_index[descriptor]] = 1.0

    matrix_metadata = read_json(args.matrix_metadata)
    global_descriptors = matrix_metadata["descriptors"]
    global_descriptor_index = {descriptor: index for index, descriptor in enumerate(global_descriptors)}
    embeddings = np.load(args.embeddings)["embeddings"].astype(np.float64)
    local_global_indices = np.asarray([global_descriptor_index[descriptor] for descriptor in descriptors], dtype=np.int64)
    E = embeddings[local_global_indices]
    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    if args.center_embeddings == "descriptor":
        E = E - E.mean(axis=0, keepdims=True)
    elif args.center_embeddings == "population":
        descriptor_weights = B.sum(axis=0)
        E = E - np.average(E, axis=0, weights=descriptor_weights).reshape(1, -1)
    G = E @ E.T
    G = (G + G.T) * 0.5

    eigenvalues, eigenvectors = np.linalg.eigh(G)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    tolerance = (
        max(G.shape) * np.finfo(np.float64).eps * max(float(eigenvalues[0]), 1.0) * args.eigenvalue_tolerance_scale
    )
    keep = eigenvalues > tolerance
    retained_eigenvalues = eigenvalues[keep]
    retained_eigenvectors = eigenvectors[:, keep]
    X = retained_eigenvectors @ np.diag(1.0 / np.sqrt(retained_eigenvalues))
    M = B @ G @ X
    left, singular_values, vt = np.linalg.svd(M, full_matrices=False)
    right = vt.T

    pairs = []
    for pair_index in range(min(args.pairs, len(singular_values))):
        u, v = orient_pair(left[:, pair_index].copy(), right[:, pair_index].copy(), G, X)
        population_descriptor_loading = G @ X @ v
        dual_descriptor_loading = X @ v
        population_descriptor_unit, population_descriptor_norm = unit_vector(population_descriptor_loading)
        dual_descriptor_unit, dual_descriptor_norm = unit_vector(dual_descriptor_loading)
        character_components = component_list(u, [character["name"] for character in rows], args.component_mass)
        pairs.append(
            {
                "rank": pair_index + 1,
                "singular_value": round(float(singular_values[pair_index]), 10),
                "vector_norms": {
                    "left_character_singular_vector": round(float(np.linalg.norm(u)), 10),
                    "right_orthogonal_singular_vector": round(float(np.linalg.norm(v)), 10),
                    "descriptor_population_projection_raw": round(population_descriptor_norm, 10),
                    "descriptor_dual_projection_raw": round(dual_descriptor_norm, 10),
                },
                "top_character_components": [
                    {
                        **character_payload(rows[component["index"]], component["amplitude"]),
                        "l2_mass_share": component["l2_mass_share"],
                        "cumulative_l2_mass": component["cumulative_l2_mass"],
                    }
                    for component in character_components
                ],
                "top_descriptor_population_unit_components": component_list(
                    population_descriptor_unit,
                    descriptors,
                    args.component_mass,
                ),
                "top_descriptor_dual_unit_components": component_list(
                    dual_descriptor_unit,
                    descriptors,
                    args.component_mass,
                ),
                "top_descriptor_population_raw_components": component_list(
                    population_descriptor_loading,
                    descriptors,
                    args.component_mass,
                ),
                "top_descriptor_dual_raw_components": component_list(
                    dual_descriptor_loading,
                    descriptors,
                    args.component_mass,
                ),
            }
        )

    output_suffix = "_local_b_g_x_svd"
    if args.center_embeddings != "none":
        output_suffix = f"_local_b_g_x_svd_center_{args.center_embeddings}"
    source_slug = slug(args.tags_input.stem)
    default_source_slug = "all_characters_llm_vndb_personality_tags"
    source_suffix = "" if source_slug == default_source_slug else f"_{source_slug}"
    output_base = args.output_dir / f"{slug(args.seiyuu)}{source_suffix}{output_suffix}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": utc_now(),
        "seiyuu": args.seiyuu,
        "source": "seiyuu_local_orthogonal_svd.py",
        "parameters": {
            "tags_input": str(args.tags_input),
            "matrix_metadata": str(args.matrix_metadata),
            "embeddings": str(args.embeddings),
            "categories": sorted(categories),
            "B": "seiyuu character x local descriptor binary incidence matrix",
            "G": "local descriptor x descriptor unthresholded cosine Gram/overlap matrix from normalized embeddings",
            "X": "Lowdin orthogonalizer over positive eigenspace of G: U @ Lambda^(-1/2)",
            "M": "B @ G @ X",
            "center_embeddings": args.center_embeddings,
            "component_mass": args.component_mass,
            "eigenvalue_tolerance": tolerance,
        },
        "counts": {
            "seiyuu_characters_total": len(all_characters),
            "characters_with_descriptors": len(rows),
            "local_descriptors": len(descriptors),
            "binary_nonzero_entries": int(B.sum()),
            "G_rank_retained": int(keep.sum()),
            "G_rank_dropped": int((~keep).sum()),
        },
        "G_eigenvalues_top20": [round(float(value), 10) for value in eigenvalues[:20]],
        "singular_values": [round(float(value), 10) for value in singular_values],
        "pairs": pairs,
        "characters": [character_payload(character) for character in rows],
        "descriptors": descriptors,
    }
    write_json(output_base.with_suffix(".json"), payload)
    np.savez_compressed(
        output_base.with_suffix(".npz"),
        B=B,
        G=G,
        X=X,
        M=M,
        G_eigenvalues=eigenvalues,
        G_eigenvectors=eigenvectors,
        G_eigen_keep=keep,
        singular_values=singular_values,
        left_singular_vectors=left,
        right_singular_vectors=right,
        local_descriptor_global_indices=local_global_indices,
    )
    print(f"wrote {output_base.with_suffix('.json')}")
    print(f"wrote {output_base.with_suffix('.npz')}")
    print(json.dumps({"counts": payload["counts"], "singular_values": payload["singular_values"][:8]}, indent=2))


if __name__ == "__main__":
    main()
