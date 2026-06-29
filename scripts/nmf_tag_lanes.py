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
from sklearn.decomposition import NMF


DEFAULT_CATEGORIES = ("personality", "traits")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit NMF lanes on character x descriptor tags.")
    parser.add_argument("--tags-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("models/tag_nmf_lanes"))
    parser.add_argument("--components", type=int, default=6)
    parser.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES))
    parser.add_argument("--semantic-smoothing", action="store_true")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--similarity-threshold", type=float, default=0.65)
    parser.add_argument("--row-normalize", action="store_true")
    parser.add_argument("--top-descriptors", type=int, default=16)
    parser.add_argument("--top-characters", type=int, default=12)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--max-iter", type=int, default=2000)
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


def descriptor_similarity_matrix(
    descriptors: list[str],
    output_dir: Path,
    model_name: str,
    threshold: float,
) -> tuple[np.ndarray, dict]:
    embeddings = load_or_create_embeddings(descriptors, output_dir, model_name)
    similarity = np.clip(embeddings @ embeddings.T, 0.0, 1.0).astype(np.float64)
    similarity[similarity < threshold] = 0.0
    np.fill_diagonal(similarity, 1.0)
    nonzero_offdiag = int((similarity > 0).sum() - len(descriptors))
    return similarity, {
        "embedding_model": model_name,
        "similarity_threshold": threshold,
        "nonzero_offdiag_similarities": nonzero_offdiag,
        "mean_nonzero_offdiag_similarity": round(
            float(similarity[(similarity > 0) & ~np.eye(len(similarity), dtype=bool)].mean()), 8
        )
        if nonzero_offdiag
        else 0.0,
    }


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)


def build_matrix(characters: list[dict], categories: set[str]) -> tuple[list[dict], list[str], np.ndarray, dict[str, Counter]]:
    descriptor_set = set()
    rows = []
    descriptor_categories: dict[str, Counter] = {}
    for character in characters:
        descriptors = []
        for category in categories:
            for tag in character.get("llm_tags", {}).get(category, []):
                descriptor = str(tag.get("tag", "")).strip().lower()
                if not descriptor:
                    continue
                descriptors.append(descriptor)
                descriptor_set.add(descriptor)
                descriptor_categories.setdefault(descriptor, Counter())[category] += 1
        if descriptors:
            rows.append(
                {
                    "character_id": int(character["anilist_character_id"]),
                    "name": character.get("name") or "",
                    "first_anime": character.get("first_anime") or "",
                    "favourites": int(character.get("favourites") or 0),
                    "site_url": character.get("site_url") or "",
                    "descriptors": sorted(set(descriptors)),
                }
            )

    descriptors = sorted(descriptor_set)
    descriptor_index = {descriptor: index for index, descriptor in enumerate(descriptors)}
    matrix = np.zeros((len(rows), len(descriptors)), dtype=np.float64)
    for row_index, row in enumerate(rows):
        for descriptor in row["descriptors"]:
            matrix[row_index, descriptor_index[descriptor]] = 1.0
    return rows, descriptors, matrix, descriptor_categories


def character_payload(row: dict, loading: float | None = None) -> dict:
    payload = {
        "character_id": row["character_id"],
        "name": row["name"],
        "first_anime": row["first_anime"],
        "favourites": row["favourites"],
        "site_url": row["site_url"],
        "descriptors": row["descriptors"],
    }
    if loading is not None:
        payload["loading"] = round(float(loading), 8)
    return payload


def lane_payloads(
    W: np.ndarray,
    H: np.ndarray,
    rows: list[dict],
    descriptors: list[str],
    descriptor_categories: dict[str, Counter],
    top_descriptors: int,
    top_characters: int,
) -> list[dict]:
    lanes = []
    for lane_index in range(H.shape[0]):
        component = H[lane_index]
        descriptor_total = float(component.sum())
        descriptor_order = np.argsort(component)[::-1]
        weighted_descriptors = []
        for descriptor_index in descriptor_order:
            weight = float(component[int(descriptor_index)])
            if weight <= 0:
                break
            descriptor = descriptors[int(descriptor_index)]
            weighted_descriptors.append(
                {
                    "descriptor": descriptor,
                    "weight": round(weight, 8),
                    "share": round(weight / descriptor_total, 8) if descriptor_total else 0.0,
                    "categories": dict(descriptor_categories.get(descriptor, Counter())),
                }
            )
            if len(weighted_descriptors) >= top_descriptors:
                break

        loading = W[:, lane_index]
        loading_order = np.argsort(loading)[::-1]
        characters = [
            character_payload(rows[int(row_index)], float(loading[int(row_index)]))
            for row_index in loading_order[:top_characters]
            if float(loading[int(row_index)]) > 0
        ]
        lanes.append(
            {
                "lane": lane_index,
                "total_descriptor_weight": round(descriptor_total, 8),
                "top_descriptors": weighted_descriptors,
                "top_characters": characters,
            }
        )
    lanes.sort(key=lambda lane: sum(character["loading"] for character in lane["top_characters"]), reverse=True)
    return lanes


def explained_entries(matrix: np.ndarray, reconstruction: np.ndarray) -> dict:
    residual = matrix - reconstruction
    frobenius = float(np.linalg.norm(matrix))
    residual_norm = float(np.linalg.norm(residual))
    return {
        "frobenius_norm": round(frobenius, 8),
        "residual_norm": round(residual_norm, 8),
        "relative_reconstruction_error": round(residual_norm / frobenius, 8) if frobenius else 0.0,
        "matrix_density": round(float((matrix > 0).sum() / matrix.size), 8) if matrix.size else 0.0,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = read_json(args.tags_input)
    categories = {category.strip() for category in args.categories}
    rows, descriptors, matrix, descriptor_categories = build_matrix(source.get("characters", []), categories)
    if len(rows) == 0 or len(descriptors) == 0:
        raise RuntimeError("No nonempty character descriptor rows found.")

    smoothing_metadata: dict[str, Any] = {"enabled": False}
    similarity = np.eye(len(descriptors), dtype=np.float64)
    fit_matrix = matrix
    if args.semantic_smoothing:
        similarity, semantic_metadata = descriptor_similarity_matrix(
            descriptors,
            args.output_dir,
            args.embedding_model,
            args.similarity_threshold,
        )
        fit_matrix = matrix @ similarity
        smoothing_metadata = {"enabled": True, **semantic_metadata}
    if args.row_normalize:
        fit_matrix = row_normalize(fit_matrix)

    components = max(1, min(args.components, len(rows), len(descriptors)))
    nmf = NMF(
        n_components=components,
        init="nndsvda",
        random_state=args.random_state,
        max_iter=args.max_iter,
        solver="cd",
        beta_loss="frobenius",
    )
    W = nmf.fit_transform(fit_matrix)
    H = nmf.components_
    reconstruction = W @ H

    base = args.output_dir / f"{slug(args.tags_input.stem)}_k{components:02d}"
    if args.semantic_smoothing:
        base = args.output_dir / (
            f"{slug(args.tags_input.stem)}_semantic_t{str(args.similarity_threshold).replace('.', 'p')}_k{components:02d}"
        )
    if args.row_normalize:
        base = base.with_name(f"{base.name}_rownorm")
    np.savez_compressed(
        base.with_suffix(".npz"),
        binary_matrix=matrix,
        fit_matrix=fit_matrix,
        descriptor_similarity=similarity,
        W=W,
        H=H,
        reconstruction=reconstruction,
    )
    payload = {
        "generated_at": utc_now(),
        "source": "nmf_tag_lanes.py",
        "parameters": {
            "tags_input": str(args.tags_input),
            "components": components,
            "categories": sorted(categories),
            "semantic_smoothing": smoothing_metadata,
            "row_normalize": args.row_normalize,
            "random_state": args.random_state,
            "init": "nndsvda",
            "solver": "cd",
            "beta_loss": "frobenius",
        },
        "counts": {
            "characters": len(rows),
            "descriptors": len(descriptors),
            "nonzero_entries": int((matrix > 0).sum()),
        },
        "fit": {
            "n_iter": int(nmf.n_iter_),
            "reconstruction_err": round(float(nmf.reconstruction_err_), 8),
            **explained_entries(fit_matrix, reconstruction),
        },
        "descriptors": descriptors,
        "characters": [character_payload(row) for row in rows],
        "lanes": lane_payloads(
            W,
            H,
            rows,
            descriptors,
            descriptor_categories,
            args.top_descriptors,
            args.top_characters,
        ),
        "matrix_npz": str(base.with_suffix(".npz")),
    }
    write_json(base.with_suffix(".json"), payload)
    print(f"wrote {base.with_suffix('.json')}")
    print(f"wrote {base.with_suffix('.npz')}")
    print(json.dumps({"counts": payload["counts"], "fit": payload["fit"]}, indent=2))


if __name__ == "__main__":
    main()
