#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse


DEFAULT_CATEGORIES = ("personality", "traits")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache character x descriptor membership and descriptor semantic matrices."
    )
    parser.add_argument("--tags-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("models/tag_descriptor_matrices"))
    parser.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES))
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--similarity-threshold", type=float, default=0.70)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--include-empty", action="store_true")
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


def character_payload(character: dict, descriptors: list[str]) -> dict:
    return {
        "character_id": int(character["anilist_character_id"]),
        "name": character.get("name") or "",
        "first_anime": character.get("first_anime") or "",
        "favourites": int(character.get("favourites") or 0),
        "site_url": character.get("site_url") or "",
        "image": character.get("image") or "",
        "descriptors": descriptors,
    }


def build_membership(
    characters: list[dict],
    categories: set[str],
    include_empty: bool,
) -> tuple[list[dict], list[str], sparse.csr_matrix, dict[str, Counter]]:
    descriptor_set: set[str] = set()
    rows: list[dict] = []
    descriptor_categories: dict[str, Counter] = {}
    descriptor_document_counts: Counter = Counter()

    for character in characters:
        descriptors: set[str] = set()
        for category in categories:
            for tag in character.get("llm_tags", {}).get(category, []):
                descriptor = str(tag.get("tag", "")).strip().lower()
                if not descriptor:
                    continue
                descriptors.add(descriptor)
                descriptor_set.add(descriptor)
                descriptor_categories.setdefault(descriptor, Counter())[category] += 1
        if descriptors or include_empty:
            sorted_descriptors = sorted(descriptors)
            rows.append(character_payload(character, sorted_descriptors))
            descriptor_document_counts.update(sorted_descriptors)

    descriptors = sorted(descriptor_set)
    descriptor_index = {descriptor: index for index, descriptor in enumerate(descriptors)}
    row_indices: list[int] = []
    col_indices: list[int] = []
    for row_index, row in enumerate(rows):
        for descriptor in row["descriptors"]:
            row_indices.append(row_index)
            col_indices.append(descriptor_index[descriptor])

    matrix = sparse.csr_matrix(
        (np.ones(len(row_indices), dtype=np.float32), (row_indices, col_indices)),
        shape=(len(rows), len(descriptors)),
        dtype=np.float32,
    )
    for descriptor in descriptors:
        descriptor_categories.setdefault(descriptor, Counter())
        descriptor_categories[descriptor]["character_count"] = descriptor_document_counts[descriptor]
    return rows, descriptors, matrix, descriptor_categories


def load_or_create_embeddings(
    descriptors: list[str],
    output_dir: Path,
    model_name: str,
    batch_size: int,
) -> np.ndarray:
    safe_model = slug(model_name)
    npz_path = output_dir / f"descriptor_embeddings_{safe_model}.npz"
    json_path = output_dir / f"descriptor_embeddings_{safe_model}.json"
    if npz_path.exists() and json_path.exists():
        metadata = read_json(json_path)
        if metadata.get("model") == model_name and metadata.get("descriptors") == descriptors:
            return np.load(npz_path)["embeddings"].astype(np.float32)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        descriptors,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, embeddings=embeddings)
    write_json(
        json_path,
        {
            "generated_at": utc_now(),
            "model": model_name,
            "descriptors": descriptors,
            "embedding_shape": list(embeddings.shape),
        },
    )
    return embeddings


def sparse_thresholded_similarity(cosine_similarity: np.ndarray, threshold: float) -> sparse.csr_matrix:
    similarity = cosine_similarity.copy()
    similarity[similarity < threshold] = 0.0
    np.fill_diagonal(similarity, 1.0)
    return sparse.csr_matrix(similarity.astype(np.float32))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = read_json(args.tags_input)
    categories = {category.strip() for category in args.categories}

    rows, descriptors, binary_matrix, descriptor_categories = build_membership(
        source.get("characters", []),
        categories,
        args.include_empty,
    )
    if not rows or not descriptors:
        raise RuntimeError("No character descriptor rows found.")

    embeddings = load_or_create_embeddings(descriptors, args.output_dir, args.embedding_model, args.batch_size)
    cosine_similarity = np.clip(embeddings @ embeddings.T, -1.0, 1.0).astype(np.float32)
    cosine_distance = (1.0 - cosine_similarity).astype(np.float32)
    np.fill_diagonal(cosine_distance, 0.0)
    thresholded_similarity = sparse_thresholded_similarity(cosine_similarity, args.similarity_threshold)
    smoothed_matrix = (binary_matrix @ thresholded_similarity).tocsr().astype(np.float32)

    base = args.output_dir / slug(args.tags_input.stem)
    binary_path = base.with_name(f"{base.name}_binary_membership.npz")
    similarity_path = base.with_name(
        f"{base.name}_descriptor_similarity_t{str(args.similarity_threshold).replace('.', 'p')}.npz"
    )
    smoothed_path = base.with_name(
        f"{base.name}_semantic_membership_t{str(args.similarity_threshold).replace('.', 'p')}.npz"
    )
    distance_path = base.with_name(f"{base.name}_descriptor_cosine_distance.npz")
    metadata_path = base.with_name(f"{base.name}_matrix_metadata.json")

    sparse.save_npz(binary_path, binary_matrix, compressed=True)
    sparse.save_npz(similarity_path, thresholded_similarity, compressed=True)
    sparse.save_npz(smoothed_path, smoothed_matrix, compressed=True)
    np.savez_compressed(distance_path, descriptor_cosine_distance=cosine_distance)

    offdiag_mask = ~np.eye(len(descriptors), dtype=bool)
    offdiag_similarity_nonzero = thresholded_similarity.nnz - len(descriptors)
    payload = {
        "generated_at": utc_now(),
        "source": "cache_tag_descriptor_matrices.py",
        "parameters": {
            "tags_input": str(args.tags_input),
            "categories": sorted(categories),
            "include_empty": args.include_empty,
            "embedding_model": args.embedding_model,
            "similarity_threshold": args.similarity_threshold,
            "semantic_membership": "binary_membership @ thresholded_descriptor_cosine_similarity",
            "distance_matrix": "descriptor_cosine_distance = 1 - normalized_embedding_dot_product",
        },
        "counts": {
            "characters": len(rows),
            "descriptors": len(descriptors),
            "binary_nonzero_entries": int(binary_matrix.nnz),
            "binary_density": round(float(binary_matrix.nnz / np.prod(binary_matrix.shape)), 10),
            "thresholded_similarity_nonzero_entries": int(thresholded_similarity.nnz),
            "thresholded_similarity_nonzero_offdiag_entries": int(offdiag_similarity_nonzero),
            "semantic_membership_nonzero_entries": int(smoothed_matrix.nnz),
            "semantic_membership_density": round(float(smoothed_matrix.nnz / np.prod(smoothed_matrix.shape)), 10),
            "mean_offdiag_cosine_similarity": round(float(cosine_similarity[offdiag_mask].mean()), 8),
            "mean_nonzero_offdiag_thresholded_similarity": round(
                float(thresholded_similarity.data[thresholded_similarity.data < 0.999999].mean()), 8
            )
            if offdiag_similarity_nonzero
            else 0.0,
        },
        "files": {
            "binary_membership_npz": str(binary_path),
            "descriptor_similarity_npz": str(similarity_path),
            "semantic_membership_npz": str(smoothed_path),
            "descriptor_cosine_distance_npz": str(distance_path),
            "descriptor_embeddings_npz": str(args.output_dir / f"descriptor_embeddings_{slug(args.embedding_model)}.npz"),
        },
        "descriptors": descriptors,
        "descriptor_metadata": {
            descriptor: dict(descriptor_categories[descriptor]) for descriptor in descriptors
        },
        "characters": rows,
    }
    write_json(metadata_path, payload)
    print(f"wrote {metadata_path}")
    print(f"wrote {binary_path}")
    print(f"wrote {similarity_path}")
    print(f"wrote {smoothed_path}")
    print(f"wrote {distance_path}")
    print(json.dumps(payload["counts"], indent=2))


if __name__ == "__main__":
    main()
