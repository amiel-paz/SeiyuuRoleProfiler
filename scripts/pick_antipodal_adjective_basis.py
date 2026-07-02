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


ADJECTIVE_POS_TAGS = {"JJ", "JJR", "JJS"}
SINGLE_ADJECTIVE_RE = re.compile(r"^[a-z]+(?:-[a-z]+)*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pick antipodal adjective axes from LLM personality tags. The candidate space is restricted "
            "to single-token adjectives, allowing hyphenated compounds."
        )
    )
    parser.add_argument("--tags-input", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", default=["personality"])
    parser.add_argument("--output-dir", type=Path, default=Path("models/antipodal_descriptor_basis"))
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--center-embeddings",
        choices=["none", "mean"],
        default="none",
        help="Use raw normalized embeddings or subtract the descriptor mean and renormalize before forming G.",
    )
    parser.add_argument("--top-k-negative", type=int, default=32)
    parser.add_argument("--target-axes", type=int, default=384)
    parser.add_argument("--min-axis-gain", type=float, default=0.90)
    parser.add_argument("--min-antipodal-quality", type=float, default=0.0, help="Require -cosine >= this value.")
    parser.add_argument("--score-antipodal-power", type=float, default=1.0)
    parser.add_argument("--limit-descriptors", type=int, default=0, help="0 means keep all adjective descriptors.")
    parser.add_argument("--min-count", type=int, default=1, help="Minimum descriptor occurrence count to keep.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_").lower()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{Path(__file__).stem}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def iter_tags(path: Path, categories: set[str]) -> list[str]:
    values: list[str] = []
    if path.suffix == ".json":
        payload = read_json(path)
        for character in payload.get("characters", []):
            for category in categories:
                for tag in character.get("llm_tags", {}).get(category, []):
                    values.append(str(tag.get("tag", "")).strip().lower())
        return values

    for line in path.open(encoding="utf-8", errors="replace"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        for category in categories:
            for tag in (row.get("tags") or {}).get(category, []):
                values.append(str(tag.get("tag", "")).strip().lower())
    return values


def wordnet_has_adjective(wn: Any, word: str) -> bool:
    return bool(wn.synsets(word, pos=wn.ADJ) or wn.synsets(word, pos=wn.ADJ_SAT))


def is_single_adjective(phrase: str, nltk: Any, wn: Any) -> bool:
    if not SINGLE_ADJECTIVE_RE.fullmatch(phrase):
        return False
    pieces = phrase.split("-")
    if len(pieces) == 1:
        word = pieces[0]
        if not wordnet_has_adjective(wn, word):
            return False
        contexts = [
            (["a", word, "character"], 1),
            (["very", word], 1),
            (["is", word], 1),
        ]
        tags = [nltk.pos_tag(context)[index][1] for context, index in contexts]
        return sum(tag in ADJECTIVE_POS_TAGS for tag in tags) >= 2

    tagged = nltk.pos_tag(["a", *pieces, "character"])
    head_is_adjective = tagged[-2][1] in ADJECTIVE_POS_TAGS
    has_adjective_piece = any(wordnet_has_adjective(wn, piece) for piece in pieces)
    return head_is_adjective and has_adjective_piece


def adjective_descriptors(tag_values: list[str], limit: int, min_count: int) -> tuple[list[dict], dict[str, int]]:
    import nltk
    from nltk.corpus import wordnet as wn

    counts = Counter(value for value in tag_values if value)
    rows = []
    rejected = Counter()
    for descriptor, count in counts.items():
        if count < min_count:
            rejected["below_min_count"] += int(count)
            continue
        if is_single_adjective(descriptor, nltk, wn):
            rows.append({"descriptor": descriptor, "count": int(count)})
        else:
            rejected["not_single_word_adjective"] += int(count)
    rows.sort(key=lambda row: (-row["count"], row["descriptor"]))
    if limit:
        rows = rows[:limit]
    return rows, dict(rejected)


def load_or_create_embeddings(
    descriptors: list[str],
    output_dir: Path,
    model_name: str,
    batch_size: int,
    force: bool,
) -> np.ndarray:
    model_slug = slug(model_name)
    json_path = output_dir / f"single_adjective_embeddings_{model_slug}.json"
    npz_path = output_dir / f"single_adjective_embeddings_{model_slug}.npz"
    if not force and json_path.exists() and npz_path.exists():
        metadata = read_json(json_path)
        if metadata.get("model") == model_name and metadata.get("descriptors") == descriptors:
            return np.load(npz_path)["embeddings"].astype(np.float64)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        descriptors,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float64)
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


def candidate_pairs(G: np.ndarray, top_k: int) -> list[dict]:
    n = G.shape[0]
    pairs: dict[tuple[int, int], dict] = {}
    k = min(max(top_k, 1), max(n - 1, 1))
    for i in range(n):
        order = np.argsort(G[i])
        kept = [int(j) for j in order if int(j) != i][:k]
        for j in kept:
            a, b = sorted((i, j))
            cosine = float(G[a, b])
            if (a, b) not in pairs:
                pairs[(a, b)] = {
                    "left_index": a,
                    "right_index": b,
                    "cosine": cosine,
                    "antipodal_quality": -cosine,
                    "antipodal_defect": 2.0 * (1.0 + cosine),
                }
    return sorted(
        pairs.values(),
        key=lambda row: (-row["antipodal_quality"], row["left_index"], row["right_index"]),
    )


def pair_axes(E: np.ndarray, pairs: list[dict]) -> np.ndarray:
    axes = []
    for pair in pairs:
        vector = E[pair["left_index"]] - E[pair["right_index"]]
        axes.append(vector / max(float(np.linalg.norm(vector)), 1.0e-12))
    return np.asarray(axes, dtype=np.float64)


def greedy_antipodal_selection(
    axes: np.ndarray,
    pairs: list[dict],
    target_axes: int,
    min_axis_gain: float,
    min_antipodal_quality: float,
    score_antipodal_power: float,
) -> tuple[list[dict], np.ndarray]:
    if axes.size == 0:
        return [], np.zeros((axes.shape[1] if axes.ndim == 2 else 0, 0), dtype=np.float64)

    dim = axes.shape[1]
    target = min(target_axes, dim, len(pairs))
    used_descriptors: set[int] = set()
    selected_pair_indices: list[int] = []
    basis_columns: list[np.ndarray] = []
    Q = np.zeros((dim, 0), dtype=np.float64)

    for rank in range(target):
        eligible = [
            index
            for index, pair in enumerate(pairs)
            if index not in selected_pair_indices
            and pair["left_index"] not in used_descriptors
            and pair["right_index"] not in used_descriptors
            and pair["antipodal_quality"] >= min_antipodal_quality
        ]
        if not eligible:
            break

        candidate_axes = axes[eligible]
        if Q.shape[1]:
            residuals = candidate_axes - (candidate_axes @ Q) @ Q.T
        else:
            residuals = candidate_axes.copy()
        gains = np.einsum("ij,ij->i", residuals, residuals)
        qualities = np.asarray([max(pairs[index]["antipodal_quality"], 0.0) for index in eligible], dtype=np.float64)
        scores = gains * np.power(np.maximum(qualities, 1.0e-12), score_antipodal_power)
        local = int(np.argmax(scores))
        best_index = eligible[local]
        best_gain = float(gains[local])
        if best_gain < min_axis_gain:
            break

        residual = residuals[local]
        norm = float(np.linalg.norm(residual))
        if norm <= 1.0e-12:
            break
        q = residual / norm
        Q = np.column_stack([Q, q])
        used_descriptors.add(pairs[best_index]["left_index"])
        used_descriptors.add(pairs[best_index]["right_index"])
        selected_pair_indices.append(best_index)
        basis_columns.append(q)
        pairs[best_index]["selection_rank"] = rank + 1
        pairs[best_index]["axis_gain"] = best_gain
        pairs[best_index]["selection_score"] = float(scores[local])

    return [pairs[index] for index in selected_pair_indices], Q


def positive_span_feasibility(selected_vectors: np.ndarray) -> dict[str, Any]:
    if selected_vectors.size == 0:
        return {"tested": False, "reason": "no selected vectors"}
    try:
        from scipy.optimize import linprog
    except ImportError:
        return {"tested": False, "reason": "scipy unavailable"}
    n, dim = selected_vectors.shape
    equality = np.vstack([selected_vectors.T, np.ones((1, n), dtype=np.float64)])
    target = np.zeros(dim + 1, dtype=np.float64)
    target[-1] = 1.0
    result = linprog(
        c=np.zeros(n, dtype=np.float64),
        A_eq=equality,
        b_eq=target,
        bounds=[(1.0e-9, None)] * n,
        method="highs",
    )
    payload = {
        "tested": True,
        "feasible": bool(result.success),
        "message": result.message,
    }
    if result.success:
        weights = result.x
        payload.update(
            {
                "min_weight": float(weights.min()),
                "max_weight": float(weights.max()),
                "residual_norm": float(np.linalg.norm(selected_vectors.T @ weights)),
            }
        )
    return payload


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_slug = slug(args.tags_input.stem)
    category_slug = slug("_".join(args.categories))
    model_slug = slug(args.embedding_model)
    stem = (
        f"{source_slug}_llm_{category_slug}_single_adjective_antipodal_"
        f"k{args.top_k_negative}_a{args.target_axes}_minc{args.min_count}_center{args.center_embeddings}_"
        f"gain{str(args.min_axis_gain).replace('.', 'p')}_{model_slug}"
    )
    json_path = args.output_dir / f"{stem}.json"
    npz_path = args.output_dir / f"{stem}.npz"

    categories = {category.strip() for category in args.categories if category.strip()}
    tag_values = iter_tags(args.tags_input, categories)
    descriptor_rows, rejected_counts = adjective_descriptors(tag_values, args.limit_descriptors, args.min_count)
    descriptors = [row["descriptor"] for row in descriptor_rows]
    if len(descriptors) < 2:
        raise RuntimeError("Need at least two adjective descriptors.")

    E = load_or_create_embeddings(descriptors, args.output_dir, args.embedding_model, args.batch_size, args.force)
    E_raw = E.copy()
    if args.center_embeddings == "mean":
        E = E - E.mean(axis=0, keepdims=True)
        E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1.0e-12)
    G = np.clip(E @ E.T, -1.0, 1.0)
    pairs = candidate_pairs(G, args.top_k_negative)
    axes = pair_axes(E, pairs)
    selected, Q = greedy_antipodal_selection(
        axes,
        pairs,
        args.target_axes,
        args.min_axis_gain,
        args.min_antipodal_quality,
        args.score_antipodal_power,
    )
    selected_indices = [(row["left_index"], row["right_index"]) for row in selected]
    selected_vectors = (
        np.asarray([E[left] for left, _ in selected_indices] + [E[right] for _, right in selected_indices])
        if selected_indices
        else np.zeros((0, E.shape[1]), dtype=np.float64)
    )
    positive_span = positive_span_feasibility(selected_vectors)

    selected_rows = []
    for row in selected:
        left = descriptor_rows[row["left_index"]]
        right = descriptor_rows[row["right_index"]]
        selected_rows.append(
            {
                **row,
                "left_descriptor": left["descriptor"],
                "right_descriptor": right["descriptor"],
                "left_count": left["count"],
                "right_count": right["count"],
            }
        )

    offdiag = G[~np.eye(G.shape[0], dtype=bool)]
    payload = {
        "generated_at": utc_now(),
        "source": "pick_antipodal_adjective_basis.py",
        "parameters": {
            "tags_input": str(args.tags_input),
            "categories": sorted(categories),
            "candidate_filter": (
                "Selected LLM tag categories only; lowercase ASCII single-token adjectives with optional hyphens; "
                "validated by WordNet adjective synsets and NLTK POS contexts."
            ),
            "embedding_model": args.embedding_model,
            "center_embeddings": args.center_embeddings,
            "top_k_negative": args.top_k_negative,
            "target_axes": args.target_axes,
            "min_count": args.min_count,
            "min_axis_gain": args.min_axis_gain,
            "min_antipodal_quality": args.min_antipodal_quality,
            "score": "axis_gain * max(-cosine, 0)^score_antipodal_power",
        },
        "counts": {
            "raw_personality_tag_occurrences": len([value for value in tag_values if value]),
            "candidate_descriptors": len(descriptors),
            "candidate_pairs": len(pairs),
            "selected_axes": len(selected_rows),
            "selected_original_vectors": len(selected_rows) * 2,
            "embedding_dim": int(E.shape[1]),
            "rejected_occurrences": rejected_counts,
        },
        "diagnostics": {
            "offdiag_cosine_min": float(offdiag.min()) if offdiag.size else 0.0,
            "offdiag_cosine_p01": float(np.quantile(offdiag, 0.01)) if offdiag.size else 0.0,
            "offdiag_cosine_p05": float(np.quantile(offdiag, 0.05)) if offdiag.size else 0.0,
            "offdiag_cosine_mean": float(offdiag.mean()) if offdiag.size else 0.0,
            "selected_cosine_min": float(min((row["cosine"] for row in selected_rows), default=0.0)),
            "selected_cosine_max": float(max((row["cosine"] for row in selected_rows), default=0.0)),
            "selected_axis_gain_min": float(min((row["axis_gain"] for row in selected_rows), default=0.0)),
            "selected_axis_gain_max": float(max((row["axis_gain"] for row in selected_rows), default=0.0)),
            "positive_span": positive_span,
        },
        "descriptors": descriptor_rows,
        "selected_pairs": selected_rows,
        "top_negative_pairs": [
            {
                **row,
                "left_descriptor": descriptor_rows[row["left_index"]]["descriptor"],
                "right_descriptor": descriptor_rows[row["right_index"]]["descriptor"],
            }
            for row in pairs[:50]
        ],
    }
    write_json(json_path, payload)
    np.savez_compressed(
        npz_path,
        embeddings=E.astype(np.float32),
        raw_embeddings=E_raw.astype(np.float32),
        gram=G.astype(np.float32),
        axes=axes.astype(np.float32),
        orthonormal_axes=Q.astype(np.float32),
        selected_pair_indices=np.asarray(
            [pairs.index(row) for row in selected],
            dtype=np.int64,
        ),
    )
    print(f"wrote {json_path}")
    print(f"wrote {npz_path}")
    print(
        json.dumps(
            {
                "candidate_descriptors": len(descriptors),
                "candidate_pairs": len(pairs),
                "selected_axes": len(selected_rows),
                "top_selected_pairs": [
                    {
                        "pair": f"{row['left_descriptor']} / {row['right_descriptor']}",
                        "cosine": round(row["cosine"], 4),
                        "axis_gain": round(row["axis_gain"], 4),
                    }
                    for row in selected_rows[:20]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
