#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_GLOSSES_JSON = Path(
    "models/global_ollama_descriptor_glosses/"
    "all_characters_llm_only_personality_traits_qwen3_5_4b_personality_traits_filtered_all_ollama_glosses.json"
)
DEFAULT_GLOSSES_NPZ = Path(
    "models/global_ollama_descriptor_glosses/"
    "all_characters_llm_only_personality_traits_qwen3_5_4b_personality_traits_filtered_all_ollama_glosses.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pivoted-Cholesky ordering of global descriptor gloss basis functions."
    )
    parser.add_argument("--glosses-json", type=Path, default=DEFAULT_GLOSSES_JSON)
    parser.add_argument("--glosses-npz", type=Path, default=DEFAULT_GLOSSES_NPZ)
    parser.add_argument("--output-dir", type=Path, default=Path("models/global_descriptor_basis"))
    parser.add_argument("--relative-trace-tol", type=float, default=1.0e-6)
    parser.add_argument("--absolute-pivot-tol", type=float, default=1.0e-12)
    parser.add_argument("--max-rank", type=int, default=None)
    parser.add_argument(
        "--basis-centering",
        choices=["mean", "none"],
        default="mean",
        help="Use mean-centered or raw descriptor-gloss embeddings for the pivoted Cholesky Gram.",
    )
    parser.add_argument(
        "--pivot-priority",
        choices=["residual", "row_sum", "row_sum_first", "row_sum_residual"],
        default="residual",
        help=(
            "Choose pivots by residual diagonal only, raw off-diagonal row sum only, "
            "raw off-diagonal row sum for only the first pivot, "
            "or residual diagonal weighted by raw off-diagonal row sum."
        ),
    )
    parser.add_argument("--nearest-mode", choices=["centered_cosine", "raw_cosine"], default="centered_cosine")
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


def descriptor_matrix(variant_embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    E = variant_embeddings.mean(axis=1)
    E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1.0e-12)
    E_centered = E - E.mean(axis=0, keepdims=True)
    return E.astype(np.float64), E_centered.astype(np.float64)


def pivoted_cholesky_feature_space(
    E: np.ndarray,
    relative_trace_tol: float,
    absolute_pivot_tol: float,
    max_rank: int | None,
    pivot_priority: str,
    priority_values: np.ndarray,
) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    n, dim = E.shape
    limit = min(n, dim) if max_rank is None else min(n, dim, max_rank)
    diagonal = np.einsum("ij,ij->i", E, E).astype(np.float64)
    initial_trace = float(diagonal.sum())
    residual_diagonal = diagonal.copy()
    selected = np.zeros(n, dtype=bool)
    L = np.zeros((n, limit), dtype=np.float64)
    pivots: list[dict] = []

    for rank in range(limit):
        residual_trace_before = float(np.maximum(residual_diagonal, 0.0).sum())
        if pivot_priority == "residual" or (pivot_priority == "row_sum_first" and rank > 0):
            pivot_scores = residual_diagonal.copy()
        elif pivot_priority in {"row_sum", "row_sum_first"}:
            pivot_scores = priority_values.copy()
            pivot_scores[residual_diagonal <= absolute_pivot_tol] = -np.inf
        elif pivot_priority == "row_sum_residual":
            pivot_scores = residual_diagonal * priority_values
        else:
            raise RuntimeError(f"Unknown pivot priority: {pivot_priority}")
        pivot_scores[selected] = -np.inf
        pivot = int(np.argmax(pivot_scores))
        pivot_score = float(pivot_scores[pivot])
        pivot_value = float(residual_diagonal[pivot])
        residual_trace_before = float(np.maximum(residual_diagonal, 0.0).sum())
        if (
            not np.isfinite(pivot_score)
            or pivot_value <= absolute_pivot_tol
            or (initial_trace > 0 and residual_trace_before / initial_trace <= relative_trace_tol)
        ):
            break

        if rank == 0:
            residual_column = E @ E[pivot]
        else:
            residual_column = E @ E[pivot] - L[:, :rank] @ L[pivot, :rank]
        new_column = residual_column / np.sqrt(pivot_value)
        L[:, rank] = new_column
        residual_diagonal = np.maximum(residual_diagonal - new_column * new_column, 0.0)
        selected[pivot] = True
        residual_diagonal[pivot] = 0.0
        residual_trace_after = float(residual_diagonal.sum())
        pivots.append(
            {
                "rank": rank + 1,
                "descriptor_index": pivot,
                "pivot_residual": pivot_value,
                "pivot_priority_score": pivot_score,
                "raw_off_diagonal_row_sum": float(priority_values[pivot]),
                "pivot_residual_fraction_of_initial_trace": pivot_value / initial_trace if initial_trace > 0 else 0.0,
                "residual_trace_before": residual_trace_before,
                "residual_trace_after": residual_trace_after,
                "residual_trace_fraction_after": residual_trace_after / initial_trace if initial_trace > 0 else 0.0,
            }
        )

    return pivots, L[:, : len(pivots)], diagonal, residual_diagonal


def nearest_pivots(E: np.ndarray, pivot_indices: list[int], mode: str) -> tuple[np.ndarray, np.ndarray]:
    if not pivot_indices:
        return np.full(E.shape[0], -1, dtype=np.int64), np.zeros(E.shape[0], dtype=np.float64)
    matrix = E.copy()
    pivots = matrix[np.asarray(pivot_indices, dtype=np.int64)].copy()
    if mode in {"centered_cosine", "raw_cosine"}:
        matrix = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1.0e-12)
        pivots = pivots / np.maximum(np.linalg.norm(pivots, axis=1, keepdims=True), 1.0e-12)
    similarities = matrix @ pivots.T
    nearest = np.argmax(np.abs(similarities), axis=1)
    nearest_similarity = similarities[np.arange(E.shape[0]), nearest]
    return nearest.astype(np.int64), nearest_similarity.astype(np.float64)


def raw_off_diagonal_row_sums(E: np.ndarray) -> np.ndarray:
    diagonal = np.einsum("ij,ij->i", E, E).astype(np.float64)
    return (E @ E.sum(axis=0) - diagonal).astype(np.float64)


def main() -> None:
    args = parse_args()
    gloss_payload = read_json(args.glosses_json)
    descriptors = [row["descriptor"] for row in gloss_payload["rows"]]
    glosses_by_descriptor = {row["descriptor"]: row.get("glosses", []) for row in gloss_payload["rows"]}
    variant_embeddings = np.load(args.glosses_npz)["variant_embeddings"].astype(np.float64)
    E_raw, E_centered = descriptor_matrix(variant_embeddings)
    E_basis = E_centered if args.basis_centering == "mean" else E_raw
    priority_values = raw_off_diagonal_row_sums(E_raw)
    if args.pivot_priority in {"row_sum", "row_sum_first", "row_sum_residual"}:
        priority_values = priority_values - min(float(priority_values.min()), 0.0)
        priority_values = priority_values / max(float(priority_values.max()), 1.0e-12)

    pivots, L, initial_diagonal, residual_diagonal = pivoted_cholesky_feature_space(
        E_basis,
        relative_trace_tol=args.relative_trace_tol,
        absolute_pivot_tol=args.absolute_pivot_tol,
        max_rank=args.max_rank,
        pivot_priority=args.pivot_priority,
        priority_values=priority_values,
    )
    pivot_indices = [row["descriptor_index"] for row in pivots]
    pivot_rank_by_index = {index: rank + 1 for rank, index in enumerate(pivot_indices)}
    nearest_source = E_centered if args.nearest_mode == "centered_cosine" else E_raw
    nearest_index_in_pivots, nearest_similarity = nearest_pivots(nearest_source, pivot_indices, args.nearest_mode)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = (
        f"center{args.basis_centering}_priority{args.pivot_priority}_"
        f"r{len(pivots)}_trace{args.relative_trace_tol:g}_pivot{args.absolute_pivot_tol:g}"
    )
    json_path = args.output_dir / f"global_qwen_gloss_descriptor_pivoted_cholesky_{suffix}.json"
    npz_path = args.output_dir / f"global_qwen_gloss_descriptor_pivoted_cholesky_{suffix}.npz"

    pivot_rows = []
    for row in pivots:
        descriptor = descriptors[row["descriptor_index"]]
        pivot_rows.append(
            {
                **row,
                "descriptor": descriptor,
                "glosses": glosses_by_descriptor.get(descriptor, []),
            }
        )

    descriptor_rows = []
    for index, descriptor in enumerate(descriptors):
        nearest_pivot_rank = int(nearest_index_in_pivots[index]) + 1 if len(pivot_indices) else None
        nearest_descriptor_index = (
            int(pivot_indices[int(nearest_index_in_pivots[index])]) if len(pivot_indices) else None
        )
        descriptor_rows.append(
            {
                "descriptor_index": index,
                "descriptor": descriptor,
                "pivot_rank": pivot_rank_by_index.get(index),
                "initial_centered_norm2": round(float(initial_diagonal[index]), 10),
                "raw_off_diagonal_row_sum": round(float(raw_off_diagonal_row_sums(E_raw)[index]), 10),
                "final_residual_norm2": round(float(residual_diagonal[index]), 10),
                "final_residual_fraction_of_initial": round(
                    float(residual_diagonal[index] / initial_diagonal[index]) if initial_diagonal[index] > 0 else 0.0,
                    10,
                ),
                "nearest_pivot_rank": nearest_pivot_rank,
                "nearest_pivot_descriptor_index": nearest_descriptor_index,
                "nearest_pivot_descriptor": descriptors[nearest_descriptor_index]
                if nearest_descriptor_index is not None
                else None,
                "nearest_pivot_similarity": round(float(nearest_similarity[index]), 10),
                "glosses": glosses_by_descriptor.get(descriptor, []),
            }
        )

    write_json(
        json_path,
        {
            "generated_at": utc_now(),
            "source": "pivot_global_descriptor_basis.py",
            "parameters": {
                "glosses_json": str(args.glosses_json),
                "glosses_npz": str(args.glosses_npz),
                "descriptor_embedding": (
                    "E = normalized mean of 4 cached Qwen-gloss BGE-small embeddings per descriptor; "
                    "E_centered = E - mean(E); Gram uses E_centered @ E_centered.T when "
                    "basis_centering='mean' and E @ E.T when basis_centering='none'."
                ),
                "relative_trace_tol": args.relative_trace_tol,
                "absolute_pivot_tol": args.absolute_pivot_tol,
                "max_rank": args.max_rank,
                "basis_centering": args.basis_centering,
                "pivot_priority": args.pivot_priority,
                "pivot_priority_note": (
                    "row_sum means raw uncentered off-diagonal row sum of E @ E.T; "
                    "row_sum_first uses that row sum for the first pivot only; "
                    "row_sum_residual multiplies that normalized row sum by the current residual diagonal."
                ),
                "nearest_mode": args.nearest_mode,
            },
            "counts": {
                "descriptors": len(descriptors),
                "embedding_dim": int(E_basis.shape[1]),
                "pivot_count": len(pivots),
                "initial_trace": round(float(initial_diagonal.sum()), 10),
                "final_residual_trace": round(float(residual_diagonal.sum()), 10),
                "final_residual_trace_fraction": round(
                    float(residual_diagonal.sum() / initial_diagonal.sum()) if initial_diagonal.sum() > 0 else 0.0,
                    10,
                ),
            },
            "pivots": pivot_rows,
            "descriptors_by_pivot_order_then_residual": sorted(
                descriptor_rows,
                key=lambda row: (
                    row["pivot_rank"] is None,
                    row["pivot_rank"] if row["pivot_rank"] is not None else 10**9,
                    -row["final_residual_norm2"],
                ),
            ),
            "descriptors_by_final_residual_desc": sorted(
                descriptor_rows,
                key=lambda row: row["final_residual_norm2"],
                reverse=True,
            ),
        },
    )
    np.savez_compressed(
        npz_path,
        E_raw=E_raw,
        E_centered=E_centered,
        E_basis=E_basis,
        cholesky_L=L,
        initial_diagonal=initial_diagonal,
        residual_diagonal=residual_diagonal,
        raw_off_diagonal_row_sums=raw_off_diagonal_row_sums(E_raw),
        pivot_priority_values=priority_values,
        pivot_indices=np.asarray(pivot_indices, dtype=np.int64),
        nearest_pivot_indices=np.asarray(
            [pivot_indices[int(row)] if len(pivot_indices) else -1 for row in nearest_index_in_pivots],
            dtype=np.int64,
        ),
        nearest_pivot_similarity=nearest_similarity,
    )
    print(f"wrote {json_path}")
    print(f"wrote {npz_path}")
    print(
        json.dumps(
            {
                "descriptors": len(descriptors),
                "pivot_count": len(pivots),
                "final_residual_trace_fraction": float(residual_diagonal.sum() / initial_diagonal.sum())
                if initial_diagonal.sum() > 0
                else 0.0,
                "top_pivots": [row["descriptor"] for row in pivot_rows[:20]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
