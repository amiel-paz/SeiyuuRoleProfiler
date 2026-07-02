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
    parser = argparse.ArgumentParser(
        description="Fit positive semantic lanes over common adjectives using NMF on positive centered-cosine overlap."
    )
    parser.add_argument(
        "--basis-json",
        type=Path,
        default=Path("models/simple_adjective_basis/simple_common_adjectives_en_c5000_r384_centermean_baai_bge-small-en-v1.5.json"),
    )
    parser.add_argument(
        "--basis-npz",
        type=Path,
        default=Path("models/simple_adjective_basis/simple_common_adjectives_en_c5000_r384_centermean_baai_bge-small-en-v1.5.npz"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models/common_adjective_nmf"))
    parser.add_argument("--ranks", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--random-state", type=int, default=13)
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


def positive_centered_similarity(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path)
    E = data["E_basis"].astype(np.float32)
    E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1.0e-12)
    similarity = E @ E.T
    np.maximum(similarity, 0.0, out=similarity)
    np.fill_diagonal(similarity, 1.0)
    return similarity.astype(np.float32)


def top_terms(weights: np.ndarray, descriptors: list[str], top_n: int) -> list[dict[str, Any]]:
    total = float(weights.sum())
    order = np.argsort(-weights)[:top_n]
    return [
        {
            "descriptor": descriptors[int(index)],
            "weight": float(weights[int(index)]),
            "share": float(weights[int(index)] / total) if total > 0 else 0.0,
        }
        for index in order
        if weights[int(index)] > 0
    ]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = read_json(args.basis_json)
    descriptors = [row["descriptor"] for row in metadata["candidates"]]
    S = positive_centered_similarity(args.basis_npz)
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
        relative_error = float(model.reconstruction_err_ / norm_s) if norm_s > 0 else 0.0
        explained_frobenius = float(max(0.0, 1.0 - relative_error * relative_error))
        component_rows = []
        for component_index in range(rank):
            weights = H[component_index]
            component_rows.append(
                {
                    "component": component_index + 1,
                    "component_mass": float(weights.sum()),
                    "effective_descriptor_count": float((weights.sum() ** 2) / np.maximum(np.sum(weights * weights), 1.0e-12)),
                    "top_descriptors": top_terms(weights, descriptors, args.top_n),
                }
            )
        component_rows.sort(key=lambda row: row["component_mass"], reverse=True)

        stem = f"common_adjectives_positive_centered_similarity_nmf_k{rank}"
        json_path = args.output_dir / f"{stem}.json"
        npz_path = args.output_dir / f"{stem}.npz"
        payload = {
            "generated_at": utc_now(),
            "source": "nmf_common_adjective_similarity.py",
            "parameters": {
                "basis_json": str(args.basis_json),
                "basis_npz": str(args.basis_npz),
                "rank": rank,
                "top_n": args.top_n,
                "max_iter": args.max_iter,
                "random_state": args.random_state,
                "matrix": "S = max(cosine(mean-centered normalized adjective embeddings), 0); diagonal set to 1",
                "model": "NMF(S) ~= W @ H, producing positive adjective lanes",
            },
            "counts": {
                "adjectives": len(descriptors),
                "rank": rank,
                "iterations": int(model.n_iter_),
            },
            "diagnostics": {
                "frobenius_norm": norm_s,
                "reconstruction_error": float(model.reconstruction_err_),
                "relative_reconstruction_error": relative_error,
                "explained_frobenius_fraction": explained_frobenius,
            },
            "components": component_rows,
        }
        write_json(json_path, payload)
        np.savez_compressed(npz_path, W=W.astype(np.float32), H=H.astype(np.float32))
        summaries.append(
            {
                "rank": rank,
                "iterations": int(model.n_iter_),
                "relative_error": relative_error,
                "explained": explained_frobenius,
                "top_component": [
                    item["descriptor"] for item in component_rows[0]["top_descriptors"][:12]
                ]
                if component_rows
                else [],
                "json": str(json_path),
            }
        )

    print(json.dumps({"summaries": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
