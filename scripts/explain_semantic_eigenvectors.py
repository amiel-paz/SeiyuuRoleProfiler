#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.linalg import eigh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain semantic character-lane eigenvectors with descriptor poles.")
    parser.add_argument("--seiyuu", default="Ayana Taketatsu")
    parser.add_argument("--role-cache", type=Path, default=Path("data/role_edges.json"))
    parser.add_argument("--semantic-model-dir", type=Path, default=Path("models/semantic_wordnet_descriptors"))
    parser.add_argument("--lane-dir", type=Path, default=Path("models/semantic_character_lanes"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--vectors", type=int, default=8)
    parser.add_argument("--top-characters", type=int, default=12)
    parser.add_argument("--top-descriptors", type=int, default=16)
    parser.add_argument("--top-contributors", type=int, default=8)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def norm_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def slug_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", norm_name(value)).strip("_")


def role_payload(role: dict, loading: float | None = None) -> dict:
    character = role["character"]
    anime = role.get("anime") or []
    first_anime = anime[0] if anime else {}
    payload = {
        "character_id": int(character["character_id"]),
        "name": character.get("name") or "",
        "image": character.get("image") or "",
        "site_url": character.get("site_url") or "",
        "anime_title": first_anime.get("title") or character.get("first_anime") or "",
        "role": role.get("character_role") or "",
    }
    if loading is not None:
        payload["loading"] = round(float(loading), 8)
    return payload


def roles_for_seiyuu(role_cache: dict, character_rows: list[dict], seiyuu: str) -> tuple[list[dict], np.ndarray]:
    char_to_row = {int(row["character_id"]): index for index, row in enumerate(character_rows)}
    roles = []
    row_indices = []
    seen = set()
    target = norm_name(seiyuu)
    for role in role_cache["roles"]:
        if norm_name(role["seiyuu"].get("name") or "") != target:
            continue
        character_id = int(role["character"]["character_id"])
        if character_id not in char_to_row or character_id in seen:
            continue
        seen.add(character_id)
        roles.append(role)
        row_indices.append(char_to_row[character_id])
    if not roles:
        raise RuntimeError(f"No in-scope roles found for {seiyuu!r}")
    return roles, np.asarray(row_indices, dtype=np.int64)


def laplacian_eigenvectors(affinity: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    degrees = affinity.sum(axis=1)
    inv_sqrt = np.divide(1.0, np.sqrt(degrees), out=np.zeros_like(degrees), where=degrees > 0)
    laplacian = np.eye(len(affinity)) - (inv_sqrt[:, None] * affinity * inv_sqrt[None, :])
    eigenvalues, eigenvectors = eigh(laplacian)
    return eigenvalues, eigenvectors, degrees


def orient_vector(vector: np.ndarray) -> np.ndarray:
    if len(vector) == 0:
        return vector
    pivot = int(np.argmax(np.abs(vector)))
    return -vector if float(vector[pivot]) < 0 else vector


def descriptor_pole(
    local_matrix,
    feature_names: list[str],
    roles: list[dict],
    loadings: np.ndarray,
    *,
    top_descriptors: int,
    top_contributors: int,
) -> list[dict]:
    weights = np.maximum(loadings, 0.0).astype(np.float64)
    total_loading = float(weights.sum())
    if total_loading <= 0:
        return []
    weighted_matrix = local_matrix.multiply(weights[:, None]).tocsr()
    pooled = np.asarray(weighted_matrix.sum(axis=0)).ravel()
    total = float(pooled.sum())
    if total <= 0:
        return []
    output = []
    for feature_index in np.argsort(pooled)[::-1]:
        value = float(pooled[int(feature_index)])
        if value <= 0:
            break
        column = np.asarray(weighted_matrix[:, int(feature_index)].todense()).ravel()
        contributor_indices = np.argsort(column)[::-1]
        contributors = [
            role_payload(roles[int(index)], float(column[int(index)]))
            for index in contributor_indices[:top_contributors]
            if float(column[int(index)]) > 0
        ]
        output.append(
            {
                "feature": feature_names[int(feature_index)],
                "weighted_sum": round(value, 8),
                "share": round(value / total, 8),
                "contributors": contributors,
            }
        )
        if len(output) >= top_descriptors:
            break
    return output


def cluster_loading_summary(labels: np.ndarray, loadings: np.ndarray) -> list[dict]:
    output = []
    for label in sorted(set(int(value) for value in labels)):
        values = loadings[labels == label]
        output.append(
            {
                "cluster": int(label),
                "size": int(len(values)),
                "mean_loading": round(float(values.mean()), 8),
                "min_loading": round(float(values.min()), 8),
                "max_loading": round(float(values.max()), 8),
            }
        )
    output.sort(key=lambda row: abs(row["mean_loading"]), reverse=True)
    return output


def main() -> None:
    args = parse_args()
    slug = slug_name(args.seiyuu)
    lane_json_path = args.lane_dir / f"{slug}_semantic_lanes.json"
    distance_path = args.lane_dir / f"{slug}_distance_matrix.npz"
    if not lane_json_path.exists() or not distance_path.exists():
        raise RuntimeError(f"Run semantic_character_lanes.py for {args.seiyuu!r} before explaining eigenvectors.")

    lane_payload = read_json(lane_json_path)
    distances_payload = np.load(distance_path)
    affinity = distances_payload["affinity"].astype(np.float64)
    labels = distances_payload["labels"].astype(np.int64)
    eigenvalues, eigenvectors, degrees = laplacian_eigenvectors(affinity)

    character_rows = read_json(args.semantic_model_dir / "character_rows.json")
    feature_names = [row["feature"] for row in read_json(args.semantic_model_dir / "semantic_vocabulary.json")]
    semantic_matrix = sparse.load_npz(args.semantic_model_dir / "character_semantic_tfidf.npz").tocsr()
    roles, row_indices = roles_for_seiyuu(read_json(args.role_cache), character_rows, args.seiyuu)
    local_matrix = semantic_matrix[row_indices].tocsr()

    vector_count = max(0, min(args.vectors, len(eigenvalues) - 1))
    vectors = []
    for offset in range(1, vector_count + 1):
        loadings = orient_vector(eigenvectors[:, offset].astype(np.float64))
        positive_order = np.argsort(loadings)[::-1]
        negative_order = np.argsort(loadings)
        vectors.append(
            {
                "eigenvector": int(offset),
                "eigenvalue": round(float(eigenvalues[offset]), 10),
                "spectral_gap_to_next": round(float(eigenvalues[offset + 1] - eigenvalues[offset]), 10)
                if offset + 1 < len(eigenvalues)
                else None,
                "interpretation": "positive and negative poles are arbitrary up to sign; this output orients each eigenvector so the largest absolute character loading is positive",
                "positive_characters": [
                    role_payload(roles[int(index)], float(loadings[int(index)]))
                    for index in positive_order[: args.top_characters]
                    if float(loadings[int(index)]) > 0
                ],
                "negative_characters": [
                    role_payload(roles[int(index)], float(loadings[int(index)]))
                    for index in negative_order[: args.top_characters]
                    if float(loadings[int(index)]) < 0
                ],
                "positive_descriptors": descriptor_pole(
                    local_matrix,
                    feature_names,
                    roles,
                    loadings,
                    top_descriptors=args.top_descriptors,
                    top_contributors=args.top_contributors,
                ),
                "negative_descriptors": descriptor_pole(
                    local_matrix,
                    feature_names,
                    roles,
                    -loadings,
                    top_descriptors=args.top_descriptors,
                    top_contributors=args.top_contributors,
                ),
                "cluster_loading_summary": cluster_loading_summary(labels, loadings),
            }
        )

    output = {
        "seiyuu": args.seiyuu,
        "source_lane_json": str(lane_json_path),
        "source_distance_matrix": str(distance_path),
        "parameters": {
            "eigenproblem": "normalized graph Laplacian from cached semantic affinity matrix",
            "dominant_vectors": "smallest nonzero Laplacian eigenvectors",
            "descriptor_mapping": "for each eigenvector pole, pool original semantic TF-IDF descriptor weights weighted by positive/negative character loadings",
            "top_characters": args.top_characters,
            "top_descriptors": args.top_descriptors,
            "top_contributors": args.top_contributors,
        },
        "counts": {
            "characters": len(roles),
            "features": len(feature_names),
            "clusters": int(len(set(int(value) for value in labels))),
        },
        "embedding_model": lane_payload.get("parameters", {}).get("embedding_model", ""),
        "eigenvalues": [round(float(value), 10) for value in eigenvalues[: min(len(eigenvalues), args.vectors + 2)]],
        "degree_summary": {
            "min": round(float(degrees.min()), 8),
            "median": round(float(np.median(degrees)), 8),
            "max": round(float(degrees.max()), 8),
        },
        "vectors": vectors,
        "clusters": lane_payload.get("clusters", []),
    }
    output_path = args.output or args.lane_dir / f"{slug}_spectral_explanation.json"
    write_json(output_path, output)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
