#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


TAG_CATEGORIES = ("personality", "traits")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build semantic character distances from LLM personality/trait tags.")
    parser.add_argument("--tags-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("models/llm_tag_semantic_lanes"))
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--include-empty", action="store_true")
    parser.add_argument("--vectors", type=int, default=8)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--top-descriptors", type=int, default=12)
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
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "tags"


def descriptor_rows(characters: list[dict], include_empty: bool) -> tuple[list[dict], list[str], np.ndarray]:
    descriptor_set = set()
    rows = []
    for character in characters:
        tags = []
        for category in TAG_CATEGORIES:
            for tag in character.get("llm_tags", {}).get(category, []):
                value = str(tag.get("tag", "")).strip().lower()
                if value:
                    tags.append({"descriptor": value, "category": category, "confidence": tag.get("confidence", "")})
                    descriptor_set.add(value)
        if tags or include_empty:
            rows.append(
                {
                    "character_id": int(character["anilist_character_id"]),
                    "name": character.get("name") or "",
                    "first_anime": character.get("first_anime") or "",
                    "favourites": int(character.get("favourites") or 0),
                    "site_url": character.get("site_url") or "",
                    "tags": tags,
                }
            )

    descriptors = sorted(descriptor_set)
    descriptor_index = {descriptor: index for index, descriptor in enumerate(descriptors)}
    membership = np.zeros((len(rows), len(descriptors)), dtype=np.float32)
    for row_index, row in enumerate(rows):
        for tag in row["tags"]:
            membership[row_index, descriptor_index[tag["descriptor"]]] = 1.0
    return rows, descriptors, membership


def load_or_create_embeddings(descriptors: list[str], output_dir: Path, model_name: str) -> np.ndarray:
    safe = slug(model_name)
    npz_path = output_dir / f"descriptor_embeddings_{safe}.npz"
    json_path = output_dir / f"descriptor_embeddings_{safe}.json"
    if npz_path.exists() and json_path.exists():
        meta = read_json(json_path)
        if meta.get("descriptors") == descriptors and meta.get("model") == model_name:
            return np.load(npz_path)["embeddings"].astype(np.float32)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        descriptors,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, embeddings=embeddings)
    write_json(json_path, {"model": model_name, "descriptors": descriptors})
    return embeddings


def descriptor_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    similarity = np.clip(embeddings @ embeddings.T, -1.0, 1.0)
    distances = 1.0 - similarity
    np.fill_diagonal(distances, 0.0)
    return distances.astype(np.float32)


def symmetric_best_match_distance(left: np.ndarray, right: np.ndarray, descriptor_distances: np.ndarray) -> float:
    left_indices = np.flatnonzero(left > 0)
    right_indices = np.flatnonzero(right > 0)
    if len(left_indices) == 0 and len(right_indices) == 0:
        return 0.0
    if len(left_indices) == 0 or len(right_indices) == 0:
        return 1.0
    pairwise = descriptor_distances[np.ix_(left_indices, right_indices)]
    return float(0.5 * (pairwise.min(axis=1).mean() + pairwise.min(axis=0).mean()))


def character_distance_matrix(membership: np.ndarray, descriptor_distances: np.ndarray) -> np.ndarray:
    count = membership.shape[0]
    distances = np.zeros((count, count), dtype=np.float32)
    for i in range(count):
        for j in range(i + 1, count):
            value = symmetric_best_match_distance(membership[i], membership[j], descriptor_distances)
            distances[i, j] = distances[j, i] = value
    return distances


def classical_mds(distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = distances.shape[0]
    if n == 0:
        return np.asarray([], dtype=np.float64), np.zeros((0, 0), dtype=np.float64)
    squared = distances.astype(np.float64) ** 2
    centered = np.eye(n) - np.ones((n, n), dtype=np.float64) / n
    gram = -0.5 * centered @ squared @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def orient(vector: np.ndarray) -> np.ndarray:
    if len(vector) == 0:
        return vector
    pivot = int(np.argmax(np.abs(vector)))
    return -vector if vector[pivot] < 0 else vector


def character_payload(row: dict, loading: float | None = None) -> dict:
    payload = {
        "character_id": row["character_id"],
        "name": row["name"],
        "first_anime": row["first_anime"],
        "favourites": row["favourites"],
        "site_url": row["site_url"],
        "descriptors": [tag["descriptor"] for tag in row["tags"]],
    }
    if loading is not None:
        payload["loading"] = round(float(loading), 8)
    return payload


def descriptor_pole(
    rows: list[dict],
    membership: np.ndarray,
    descriptors: list[str],
    loadings: np.ndarray,
    top_descriptors: int,
) -> list[dict]:
    weights = np.maximum(loadings, 0.0)
    if float(weights.sum()) <= 0:
        return []
    pooled = weights @ membership
    total = float(pooled.sum())
    if total <= 0:
        return []
    output = []
    for descriptor_index in np.argsort(pooled)[::-1]:
        value = float(pooled[int(descriptor_index)])
        if value <= 0:
            break
        contributors = []
        for row_index in np.argsort(weights * membership[:, int(descriptor_index)])[::-1]:
            score = float(weights[int(row_index)] * membership[int(row_index), int(descriptor_index)])
            if score <= 0:
                break
            contributors.append(character_payload(rows[int(row_index)], score))
            if len(contributors) >= 6:
                break
        output.append(
            {
                "descriptor": descriptors[int(descriptor_index)],
                "weighted_sum": round(value, 8),
                "share": round(value / total, 8),
                "contributors": contributors,
            }
        )
        if len(output) >= top_descriptors:
            break
    return output


def nearest_neighbors(distances: np.ndarray, rows: list[dict], limit: int) -> list[dict]:
    output = []
    for i, row in enumerate(rows):
        order = np.argsort(distances[i])
        neighbors = [
            {**character_payload(rows[int(j)]), "distance": round(float(distances[i, int(j)]), 8)}
            for j in order
            if int(j) != i
        ][:limit]
        output.append({**character_payload(row), "nearest_neighbors": neighbors})
    return output


def descriptor_union_summary(rows: list[dict]) -> dict:
    counts = {category: Counter() for category in TAG_CATEGORIES}
    for row in rows:
        for tag in row["tags"]:
            counts[tag["category"]][tag["descriptor"]] += 1
    return {category: counts[category].most_common() for category in TAG_CATEGORIES}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = read_json(args.tags_input)
    rows, descriptors, membership = descriptor_rows(source["characters"], args.include_empty)
    embeddings = load_or_create_embeddings(descriptors, args.output_dir, args.embedding_model)
    descriptor_distances = descriptor_distance_matrix(embeddings)
    character_distances = character_distance_matrix(membership, descriptor_distances)
    eigenvalues, eigenvectors = classical_mds(character_distances)

    base = args.output_dir / slug(args.tags_input.stem)
    np.savez_compressed(
        base.with_suffix(".npz"),
        membership=membership,
        descriptor_distances=descriptor_distances,
        character_distances=character_distances,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        embeddings=embeddings,
    )

    vector_count = max(0, min(args.vectors, len(rows)))
    vectors = []
    for index in range(vector_count):
        if float(eigenvalues[index]) <= 1e-10:
            continue
        loadings = orient(eigenvectors[:, index].astype(np.float64))
        positive_order = np.argsort(loadings)[::-1]
        negative_order = np.argsort(loadings)
        vectors.append(
            {
                "axis": int(index + 1),
                "eigenvalue": round(float(eigenvalues[index]), 10),
                "explained_positive_variance_share": round(
                    float(eigenvalues[index] / eigenvalues[eigenvalues > 0].sum()), 8
                ),
                "positive_characters": [
                    character_payload(rows[int(row_index)], float(loadings[int(row_index)]))
                    for row_index in positive_order[: args.neighbors]
                    if float(loadings[int(row_index)]) > 0
                ],
                "negative_characters": [
                    character_payload(rows[int(row_index)], float(loadings[int(row_index)]))
                    for row_index in negative_order[: args.neighbors]
                    if float(loadings[int(row_index)]) < 0
                ],
                "positive_descriptors": descriptor_pole(
                    rows, membership, descriptors, loadings, args.top_descriptors
                ),
                "negative_descriptors": descriptor_pole(
                    rows, membership, descriptors, -loadings, args.top_descriptors
                ),
            }
        )

    payload = {
        "generated_at": utc_now(),
        "source": "llm_tag_character_distances.py",
        "parameters": {
            "tags_input": str(args.tags_input),
            "embedding_model": args.embedding_model,
            "tag_categories": list(TAG_CATEGORIES),
            "character_distance": "symmetric best-match over descriptor cosine distances",
            "eigenvectors": "classical MDS eigenvectors of the character distance matrix",
            "include_empty": args.include_empty,
        },
        "counts": {
            "characters": len(rows),
            "descriptors": len(descriptors),
            "nonempty_characters": int((membership.sum(axis=1) > 0).sum()),
        },
        "descriptors": descriptors,
        "descriptor_counts": descriptor_union_summary(rows),
        "characters": [character_payload(row) for row in rows],
        "nearest_neighbors": nearest_neighbors(character_distances, rows, args.neighbors),
        "eigenvectors": vectors,
        "matrix_npz": str(base.with_suffix(".npz")),
    }
    write_json(base.with_suffix(".json"), payload)
    print(f"wrote {base.with_suffix('.json')}")
    print(f"wrote {base.with_suffix('.npz')}")
    print(json.dumps(payload["counts"], indent=2))


if __name__ == "__main__":
    main()
