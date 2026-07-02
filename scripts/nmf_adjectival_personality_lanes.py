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
    parser = argparse.ArgumentParser(description="Scan NMF lane counts for adjectival personality descriptors.")
    parser.add_argument(
        "--descriptors-input",
        type=Path,
        default=Path("run/adjectival_personality_union/adjectival_personality_union.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models/adjectival_personality_nmf"))
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--ranks",
        nargs="+",
        type=int,
        default=[16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 320, 384],
    )
    parser.add_argument("--top-n", type=int, default=24)
    parser.add_argument("--max-iter", type=int, default=600)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--force-embeddings", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_").lower()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def descriptor_rows(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    rows = payload["descriptors"]
    return sorted(rows, key=lambda row: row["tag"])


def load_or_create_embeddings(
    descriptors: list[str],
    output_dir: Path,
    model_name: str,
    batch_size: int,
    force: bool,
) -> np.ndarray:
    model_slug = slug(model_name)
    json_path = output_dir / f"adjectival_personality_embeddings_{model_slug}.json"
    npz_path = output_dir / f"adjectival_personality_embeddings_{model_slug}.npz"
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
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1.0e-12)
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


def uncentered_positive_overlap(embeddings: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    cosine = embeddings @ embeddings.T
    cosine = cosine.astype(np.float32)
    offdiag = cosine[~np.eye(cosine.shape[0], dtype=bool)]
    S = cosine.copy()
    negative_fraction = float(np.mean(offdiag < 0))
    np.maximum(S, 0.0, out=S)
    np.fill_diagonal(S, 1.0)
    clipped_offdiag = S[~np.eye(S.shape[0], dtype=bool)]
    diagnostics = {
        "matrix": "S = max(normalized_embedding @ normalized_embedding.T, 0), diagonal set to 1",
        "cosine_min": float(offdiag.min()),
        "cosine_max_offdiag": float(offdiag.max()),
        "cosine_mean_offdiag": float(offdiag.mean()),
        "cosine_negative_fraction_offdiag": negative_fraction,
        "positive_overlap_mean_offdiag": float(clipped_offdiag.mean()),
        "positive_overlap_density_offdiag": float(np.mean(clipped_offdiag > 0)),
    }
    return S, diagnostics


def effective_count(weights: np.ndarray) -> float:
    total = float(weights.sum())
    if total <= 0:
        return 0.0
    return float((total * total) / max(float(np.dot(weights, weights)), 1.0e-12))


def component_overlap(H: np.ndarray) -> dict[str, Any]:
    norms = np.linalg.norm(H, axis=1, keepdims=True)
    normalized = H / np.maximum(norms, 1.0e-12)
    overlap = normalized @ normalized.T
    mask = ~np.eye(overlap.shape[0], dtype=bool)
    values = overlap[mask]
    max_partner = []
    for i in range(overlap.shape[0]):
        row = np.delete(overlap[i], i)
        max_partner.append(float(row.max()) if row.size else 0.0)
    return {
        "mean_component_overlap": float(values.mean()) if values.size else 0.0,
        "median_component_overlap": float(np.median(values)) if values.size else 0.0,
        "p90_component_overlap": float(np.quantile(values, 0.90)) if values.size else 0.0,
        "max_component_overlap": float(values.max()) if values.size else 0.0,
        "mean_nearest_component_overlap": float(np.mean(max_partner)) if max_partner else 0.0,
        "max_nearest_component_overlap": float(np.max(max_partner)) if max_partner else 0.0,
    }


def top_terms(weights: np.ndarray, descriptor_rows: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    total = float(weights.sum())
    order = np.argsort(-weights)[:top_n]
    rows = []
    for index in order:
        weight = float(weights[int(index)])
        if weight <= 0:
            continue
        descriptor = descriptor_rows[int(index)]
        rows.append(
            {
                "descriptor": descriptor["tag"],
                "weight": weight,
                "share": weight / total if total > 0 else 0.0,
                "character_count": descriptor.get("character_count", 0),
                "assignment_count": descriptor.get("assignment_count", 0),
                "sources": descriptor.get("sources", {}),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = descriptor_rows(args.descriptors_input)
    descriptors = [row["tag"] for row in rows]
    embeddings = load_or_create_embeddings(
        descriptors,
        args.output_dir,
        args.embedding_model,
        args.batch_size,
        args.force_embeddings,
    )
    S, matrix_diagnostics = uncentered_positive_overlap(embeddings)
    norm_s = float(np.linalg.norm(S, ord="fro"))
    np.savez_compressed(args.output_dir / "adjectival_personality_uncentered_positive_overlap.npz", S=S)

    from sklearn.decomposition import NMF
    from sklearn.exceptions import ConvergenceWarning
    import warnings

    summaries = []
    for rank in args.ranks:
        if rank >= len(descriptors):
            continue
        print(f"fitting K={rank}", flush=True)
        model = NMF(
            n_components=rank,
            init="nndsvda",
            solver="cd",
            beta_loss="frobenius",
            max_iter=args.max_iter,
            random_state=args.random_state,
            tol=1.0e-4,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            W = model.fit_transform(S)
        H = model.components_.astype(np.float32)
        overlap_diagnostics = component_overlap(H)
        relative_error = float(model.reconstruction_err_ / norm_s) if norm_s > 0 else 0.0
        explained_frobenius = float(max(0.0, 1.0 - relative_error * relative_error))
        component_masses = H.sum(axis=1)
        effective_counts = np.asarray([effective_count(row) for row in H], dtype=np.float32)
        component_rows = []
        for component_index in range(rank):
            component_rows.append(
                {
                    "component": component_index + 1,
                    "component_mass": float(component_masses[component_index]),
                    "effective_descriptor_count": float(effective_counts[component_index]),
                    "top_descriptors": top_terms(H[component_index], rows, args.top_n),
                }
            )
        component_rows.sort(key=lambda row: row["component_mass"], reverse=True)

        stem = f"adjectival_personality_uncentered_positive_overlap_nmf_k{rank}"
        json_path = args.output_dir / f"{stem}.json"
        npz_path = args.output_dir / f"{stem}.npz"
        diagnostics = {
            "frobenius_norm": norm_s,
            "reconstruction_error": float(model.reconstruction_err_),
            "relative_reconstruction_error": relative_error,
            "explained_frobenius_fraction": explained_frobenius,
            "iterations": int(model.n_iter_),
            "converged": not any(isinstance(item.message, ConvergenceWarning) for item in caught),
            "component_effective_count_mean": float(effective_counts.mean()),
            "component_effective_count_median": float(np.median(effective_counts)),
            **overlap_diagnostics,
        }
        write_json(
            json_path,
            {
                "generated_at": utc_now(),
                "source": "nmf_adjectival_personality_lanes.py",
                "parameters": {
                    "descriptors_input": str(args.descriptors_input),
                    "embedding_model": args.embedding_model,
                    "rank": rank,
                    "max_iter": args.max_iter,
                    "random_state": args.random_state,
                    "top_n": args.top_n,
                    "model": "NMF(S) ~= W @ H",
                    "lane_overlap": "cosine overlap between normalized NMF component rows H",
                },
                "counts": {
                    "descriptors": len(descriptors),
                    "rank": rank,
                },
                "matrix_diagnostics": matrix_diagnostics,
                "diagnostics": diagnostics,
                "components": component_rows,
            },
        )
        np.savez_compressed(npz_path, W=W.astype(np.float32), H=H.astype(np.float32))
        summaries.append(
            {
                "rank": rank,
                "json": str(json_path),
                "npz": str(npz_path),
                **diagnostics,
            }
        )

    best_mean = min(summaries, key=lambda row: row["mean_component_overlap"]) if summaries else None
    best_nearest = min(summaries, key=lambda row: row["mean_nearest_component_overlap"]) if summaries else None
    summary_payload = {
        "generated_at": utc_now(),
        "source": "nmf_adjectival_personality_lanes.py",
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "counts": {
            "descriptors": len(descriptors),
            "embedding_dim": int(embeddings.shape[1]),
        },
        "matrix_diagnostics": matrix_diagnostics,
        "best_by_mean_component_overlap": best_mean,
        "best_by_mean_nearest_component_overlap": best_nearest,
        "summaries": summaries,
    }
    write_json(args.output_dir / "adjectival_personality_nmf_rank_scan_summary.json", summary_payload)
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
