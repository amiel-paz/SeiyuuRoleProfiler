#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ADJECTIVE_POS_TAGS = {"JJ", "JJR", "JJS"}
WORD_RE = re.compile(r"^[a-z]{3,}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache a simple common-adjective anchor basis selected by pivoted Cholesky."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models/simple_adjective_basis"))
    parser.add_argument("--language", default="en")
    parser.add_argument("--scan-top-n", type=int, default=120000)
    parser.add_argument("--candidate-count", type=int, default=5000)
    parser.add_argument("--max-rank", type=int, default=384)
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--relative-trace-tol", type=float, default=1.0e-6)
    parser.add_argument("--absolute-pivot-tol", type=float, default=1.0e-12)
    parser.add_argument(
        "--basis-centering",
        choices=["none", "mean"],
        default="none",
        help="Whether to subtract mean(E) before pivoted Cholesky.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_").lower()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def import_nltk_wordnet() -> tuple[Any, Any]:
    try:
        import nltk
        from nltk.corpus import wordnet as wn
    except ImportError as error:
        raise RuntimeError("Install nltk before building the adjective basis.") from error
    nltk.pos_tag(["a", "simple", "character"])
    wn.synsets("simple", pos=wn.ADJ)
    return nltk, wn


def wordnet_has_adjective(wn: Any, word: str) -> bool:
    return bool(wn.synsets(word, pos=wn.ADJ) or wn.synsets(word, pos=wn.ADJ_SAT))


def is_simple_adjective(word: str, nltk: Any, wn: Any) -> bool:
    if not WORD_RE.match(word):
        return False
    if not wordnet_has_adjective(wn, word):
        return False
    contexts = [
        ["a", word, "person"],
        ["very", word],
        ["is", word],
    ]
    tags = [nltk.pos_tag(context)[1 if len(context) == 3 else -1][1] for context in contexts]
    return sum(tag in ADJECTIVE_POS_TAGS for tag in tags) >= 2


def common_simple_adjectives(language: str, scan_top_n: int, candidate_count: int) -> list[dict]:
    from wordfreq import top_n_list, zipf_frequency

    nltk, wn = import_nltk_wordnet()
    rows = []
    seen = set()
    for word in top_n_list(language, scan_top_n):
        word = word.lower()
        if word in seen:
            continue
        seen.add(word)
        if not is_simple_adjective(word, nltk, wn):
            continue
        rows.append(
            {
                "descriptor": word,
                "frequency_rank": len(seen),
                "zipf_frequency": round(float(zipf_frequency(word, language)), 6),
            }
        )
        if len(rows) >= candidate_count:
            break
    if len(rows) < candidate_count:
        raise RuntimeError(f"Only found {len(rows)} adjectives after scanning {scan_top_n} words.")
    return rows


def encode(texts: list[str], model_name: str, batch_size: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embeddings.astype(np.float64)


def pivoted_cholesky_frequency_tiebreak(
    E: np.ndarray,
    max_rank: int,
    relative_trace_tol: float,
    absolute_pivot_tol: float,
) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    n, dim = E.shape
    limit = min(n, dim, max_rank)
    diagonal = np.einsum("ij,ij->i", E, E).astype(np.float64)
    initial_trace = float(diagonal.sum())
    residual_diagonal = diagonal.copy()
    selected = np.zeros(n, dtype=bool)
    L = np.zeros((n, limit), dtype=np.float64)
    pivots: list[dict] = []
    frequency_priority = np.linspace(1.0, 0.0, n, endpoint=False, dtype=np.float64)

    for rank in range(limit):
        residual_trace_before = float(np.maximum(residual_diagonal, 0.0).sum())
        if initial_trace > 0 and residual_trace_before / initial_trace <= relative_trace_tol:
            break
        eligible = np.where((~selected) & (residual_diagonal > absolute_pivot_tol))[0]
        if eligible.size == 0:
            break
        max_residual = float(np.max(residual_diagonal[eligible]))
        tolerance = max(1.0e-10, abs(max_residual) * 1.0e-10)
        tied = eligible[residual_diagonal[eligible] >= max_residual - tolerance]
        pivot = int(tied[np.argmax(frequency_priority[tied])])
        pivot_value = float(residual_diagonal[pivot])

        residual_column = E @ E[pivot] if rank == 0 else E @ E[pivot] - L[:, :rank] @ L[pivot, :rank]
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
                "pivot_priority_score": pivot_value,
                "frequency_tiebreak_score": float(frequency_priority[pivot]),
                "pivot_residual_fraction_of_initial_trace": pivot_value / initial_trace if initial_trace > 0 else 0.0,
                "residual_trace_before": residual_trace_before,
                "residual_trace_after": residual_trace_after,
                "residual_trace_fraction_after": residual_trace_after / initial_trace if initial_trace > 0 else 0.0,
            }
        )

    return pivots, L[:, : len(pivots)], diagonal, residual_diagonal


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_slug = slug(args.embedding_model)
    stem = (
        f"simple_common_adjectives_{args.language}_c{args.candidate_count}_"
        f"r{args.max_rank}_center{args.basis_centering}_{model_slug}"
    )
    json_path = args.output_dir / f"{stem}.json"
    npz_path = args.output_dir / f"{stem}.npz"

    rows = common_simple_adjectives(args.language, args.scan_top_n, args.candidate_count)
    descriptors = [row["descriptor"] for row in rows]
    E_raw = encode(descriptors, args.embedding_model, args.batch_size)
    E_raw = E_raw / np.maximum(np.linalg.norm(E_raw, axis=1, keepdims=True), 1.0e-12)
    E_basis = E_raw - E_raw.mean(axis=0, keepdims=True) if args.basis_centering == "mean" else E_raw
    pivots, L, initial_diagonal, residual_diagonal = pivoted_cholesky_frequency_tiebreak(
        E_basis,
        max_rank=args.max_rank,
        relative_trace_tol=args.relative_trace_tol,
        absolute_pivot_tol=args.absolute_pivot_tol,
    )
    pivot_indices = [row["descriptor_index"] for row in pivots]
    pivot_rows = []
    for pivot in pivots:
        source = rows[pivot["descriptor_index"]]
        pivot_rows.append(
            {
                **pivot,
                "descriptor": source["descriptor"],
                "frequency_rank": source["frequency_rank"],
                "zipf_frequency": source["zipf_frequency"],
            }
        )

    write_json(
        json_path,
        {
            "generated_at": utc_now(),
            "source": "cache_simple_adjective_basis.py",
            "parameters": {
                "language": args.language,
                "scan_top_n": args.scan_top_n,
                "candidate_count": args.candidate_count,
                "max_rank": args.max_rank,
                "embedding_model": args.embedding_model,
                "basis_centering": args.basis_centering,
                "selection": (
                    "Take the most common wordfreq English words, keep single-token WordNet/NLTK adjectives, "
                    "embed them with normalized BGE-small vectors, then choose nonredundant anchors by "
                    "pivoted Cholesky on E @ E.T."
                ),
                "tie_break": "Pivot ties follow candidate frequency order because np.argmax picks the first maximum.",
                "tie_break_implementation": "At each Cholesky step, pivots within 1e-10 of the max residual choose the most frequent remaining adjective.",
            },
            "counts": {
                "candidates": len(rows),
                "embedding_dim": int(E_basis.shape[1]),
                "pivot_count": len(pivots),
                "initial_trace": round(float(initial_diagonal.sum()), 10),
                "final_residual_trace": round(float(residual_diagonal.sum()), 10),
                "final_residual_trace_fraction": round(
                    float(residual_diagonal.sum() / initial_diagonal.sum()) if initial_diagonal.sum() > 0 else 0.0,
                    10,
                ),
            },
            "candidates": rows,
            "pivots": pivot_rows,
        },
    )
    np.savez_compressed(
        npz_path,
        E_raw=E_raw,
        E_basis=E_basis,
        cholesky_L=L,
        initial_diagonal=initial_diagonal,
        residual_diagonal=residual_diagonal,
        pivot_indices=np.asarray(pivot_indices, dtype=np.int64),
    )
    print(f"wrote {json_path}")
    print(f"wrote {npz_path}")
    print(
        json.dumps(
            {
                "candidates": len(rows),
                "pivot_count": len(pivots),
                "top_pivots": [row["descriptor"] for row in pivot_rows[:40]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
