#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from pick_antipodal_adjective_basis import adjective_descriptors, iter_tags, slug, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit positive semantic NMF lanes over LLM adjective descriptors."
    )
    parser.add_argument("--tags-input", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", default=["personality"])
    parser.add_argument("--output-dir", type=Path, default=Path("models/llm_adjective_nmf"))
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ranks", nargs="+", type=int, required=True)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--force-embeddings", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_or_create_embeddings(
    descriptors: list[str],
    output_dir: Path,
    model_name: str,
    batch_size: int,
    force: bool,
) -> np.ndarray:
    model_slug = slug(model_name)
    json_path = output_dir / f"descriptor_embeddings_{model_slug}.json"
    npz_path = output_dir / f"descriptor_embeddings_{model_slug}.npz"
    if not force and json_path.exists() and npz_path.exists():
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
    embeddings = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1.0e-12)
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


def positive_centered_similarity(embeddings: np.ndarray) -> np.ndarray:
    E = embeddings.astype(np.float32)
    E = E - E.mean(axis=0, keepdims=True)
    E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1.0e-12)
    similarity = E @ E.T
    np.maximum(similarity, 0.0, out=similarity)
    np.fill_diagonal(similarity, 1.0)
    return similarity.astype(np.float32)


def top_terms(weights: np.ndarray, descriptors: list[str], counts: dict[str, int], top_n: int) -> list[dict[str, Any]]:
    total = float(weights.sum())
    order = np.argsort(-weights)[:top_n]
    return [
        {
            "descriptor": descriptors[int(index)],
            "weight": float(weights[int(index)]),
            "share": float(weights[int(index)] / total) if total > 0 else 0.0,
            "count": int(counts.get(descriptors[int(index)], 0)),
        }
        for index in order
        if weights[int(index)] > 0
    ]


def component_overlap(H: np.ndarray) -> dict[str, float]:
    Hn = H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1.0e-12)
    O = Hn @ Hn.T
    off = O[~np.eye(H.shape[0], dtype=bool)]
    return {
        "mean": float(off.mean()) if off.size else 0.0,
        "median": float(np.quantile(off, 0.5)) if off.size else 0.0,
        "p95": float(np.quantile(off, 0.95)) if off.size else 0.0,
        "p99": float(np.quantile(off, 0.99)) if off.size else 0.0,
        "max": float(off.max()) if off.size else 0.0,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    categories = {category.strip() for category in args.categories if category.strip()}
    category_slug = slug("_".join(sorted(categories)))
    source_slug = slug(args.tags_input.stem)

    tag_values = iter_tags(args.tags_input, categories)
    descriptor_rows, rejected_counts = adjective_descriptors(tag_values, limit=0, min_count=args.min_count)
    descriptors = [row["descriptor"] for row in descriptor_rows]
    counts = {row["descriptor"]: row["count"] for row in descriptor_rows}
    if len(descriptors) < max(args.ranks):
        raise RuntimeError(f"Need at least max(rank) descriptors; got {len(descriptors)}.")

    embeddings = load_or_create_embeddings(
        descriptors,
        args.output_dir / f"{source_slug}_{category_slug}_minc{args.min_count}",
        args.embedding_model,
        args.batch_size,
        args.force_embeddings,
    )
    S = positive_centered_similarity(embeddings)
    norm_s = float(np.linalg.norm(S, ord="fro"))

    from sklearn.decomposition import NMF

    summaries = []
    for rank in args.ranks:
        model = NMF(
            n_components=rank,
            init="nndsvda",
            solver="cd",
            beta_loss="frobenius",
            max_iter=args.max_iter,
            random_state=args.random_state,
            tol=1.0e-4,
        )
        W = model.fit_transform(S)
        H = model.components_
        overlap = component_overlap(H.astype(np.float64))
        relative_error = float(model.reconstruction_err_ / norm_s) if norm_s > 0 else 0.0
        explained = float(max(0.0, 1.0 - relative_error * relative_error))
        components = []
        for component_index in range(rank):
            weights = H[component_index]
            components.append(
                {
                    "component": component_index + 1,
                    "component_mass": float(weights.sum()),
                    "effective_descriptor_count": float((weights.sum() ** 2) / np.maximum(np.sum(weights * weights), 1.0e-12)),
                    "top_descriptors": top_terms(weights, descriptors, counts, args.top_n),
                }
            )
        components.sort(key=lambda row: row["component_mass"], reverse=True)

        stem = f"{source_slug}_{category_slug}_single_adjective_minc{args.min_count}_positive_centered_similarity_nmf_k{rank}"
        json_path = args.output_dir / f"{stem}.json"
        npz_path = args.output_dir / f"{stem}.npz"
        payload = {
            "generated_at": utc_now(),
            "source": "nmf_llm_adjective_descriptors.py",
            "parameters": {
                "tags_input": str(args.tags_input),
                "categories": sorted(categories),
                "min_count": args.min_count,
                "embedding_model": args.embedding_model,
                "rank": rank,
                "matrix": "S = max(cosine(mean-centered normalized descriptor embeddings), 0); diagonal set to 1",
                "model": "NMF(S) ~= W @ H, producing positive descriptor lanes",
                "random_state": args.random_state,
            },
            "counts": {
                "raw_tag_occurrences": len([value for value in tag_values if value]),
                "descriptors": len(descriptors),
                "rank": rank,
                "iterations": int(model.n_iter_),
                "rejected_occurrences": rejected_counts,
            },
            "diagnostics": {
                "frobenius_norm": norm_s,
                "reconstruction_error": float(model.reconstruction_err_),
                "relative_reconstruction_error": relative_error,
                "explained_frobenius_fraction": explained,
                "component_overlap": overlap,
            },
            "descriptors": descriptor_rows,
            "components": components,
        }
        write_json(json_path, payload)
        np.savez_compressed(npz_path, W=W.astype(np.float32), H=H.astype(np.float32))
        summaries.append(
            {
                "rank": rank,
                "iterations": int(model.n_iter_),
                "relative_error": relative_error,
                "explained": explained,
                "mean_overlap": overlap["mean"],
                "median_overlap": overlap["median"],
                "p95_overlap": overlap["p95"],
                "max_overlap": overlap["max"],
                "json": str(json_path),
                "top_component": [term["descriptor"] for term in components[0]["top_descriptors"][:12]] if components else [],
            }
        )

    print(json.dumps({"summaries": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
