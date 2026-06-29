#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DIAGNOSTIC_PAIRS = [
    ("tsundere", "sharp-tongued", "close"),
    ("tsundere", "cold", "close"),
    ("tsundere", "hostile", "close"),
    ("tsundere", "blunt", "close"),
    ("tsundere", "harsh", "close"),
    ("tsundere", "cheeky", "close"),
    ("tsundere", "rebellious", "close"),
    ("tsundere", "brother complex", "related"),
    ("brother complex", "little sister fetish", "close"),
    ("perfectionist", "overachiever", "close"),
    ("clumsy", "careless", "related"),
    ("tsundere", "japanese language teacher", "far"),
    ("tsundere", "athletic", "far"),
    ("tsundere", "vr gamer enthusiast", "far"),
    ("clumsy", "diligent worker", "far"),
    ("hostile", "kind spirit", "far"),
]


MODEL_PRESETS = {
    "bge-large": {
        "backend": "sentence-transformers",
        "model": "BAAI/bge-large-en-v1.5",
        "prefix": "",
        "trust_remote_code": False,
    },
    "e5-large": {
        "backend": "sentence-transformers",
        "model": "intfloat/e5-large-v2",
        "prefix": "query: ",
        "trust_remote_code": False,
    },
    "gte-large": {
        "backend": "sentence-transformers",
        "model": "Alibaba-NLP/gte-large-en-v1.5",
        "prefix": "",
        "trust_remote_code": True,
    },
    "bge-small": {
        "backend": "sentence-transformers",
        "model": "BAAI/bge-small-en-v1.5",
        "prefix": "",
        "trust_remote_code": False,
    },
    "mxbai-embed-large": {
        "backend": "ollama",
        "model": "mxbai-embed-large",
        "prefix": "",
        "trust_remote_code": False,
    },
    "nomic-embed-text": {
        "backend": "ollama",
        "model": "nomic-embed-text",
        "prefix": "",
        "trust_remote_code": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache descriptor embeddings and diagnostic similarities.")
    parser.add_argument(
        "--matrix-metadata",
        type=Path,
        default=Path("models/tag_descriptor_matrices_llm_only/all_characters_llm_only_personality_traits_matrix_metadata.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models/embedding_bakeoff"))
    parser.add_argument("--preset", choices=sorted(MODEL_PRESETS), required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--force", action="store_true")
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
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "model"


def encode_sentence_transformers(
    descriptors: list[str],
    model_name: str,
    prefix: str,
    batch_size: int,
    trust_remote_code: bool,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, trust_remote_code=trust_remote_code)
    texts = [f"{prefix}{descriptor}" for descriptor in descriptors]
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)


def ollama_embed_one(ollama_url: str, model_name: str, text: str) -> list[float]:
    request = urllib.request.Request(
        f"{ollama_url.rstrip('/')}/api/embeddings",
        data=json.dumps({"model": model_name, "prompt": text}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["embedding"]


def encode_ollama(descriptors: list[str], model_name: str, prefix: str, ollama_url: str) -> np.ndarray:
    vectors = []
    for index, descriptor in enumerate(descriptors, 1):
        vectors.append(ollama_embed_one(ollama_url, model_name, f"{prefix}{descriptor}"))
        if index % 250 == 0:
            print(f"embedded {index}/{len(descriptors)}", flush=True)
    embeddings = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return np.divide(embeddings, norms, out=np.zeros_like(embeddings), where=norms > 0)


def diagnostic_report(descriptors: list[str], embeddings: np.ndarray) -> dict:
    descriptor_index = {descriptor: index for index, descriptor in enumerate(descriptors)}
    pair_rows = []
    for left, right, expected in DIAGNOSTIC_PAIRS:
        if left not in descriptor_index or right not in descriptor_index:
            pair_rows.append({"left": left, "right": right, "expected": expected, "missing": True})
            continue
        similarity = float(embeddings[descriptor_index[left]] @ embeddings[descriptor_index[right]])
        pair_rows.append(
            {
                "left": left,
                "right": right,
                "expected": expected,
                "cosine_similarity": round(similarity, 8),
            }
        )

    neighbor_terms = ["tsundere", "tsundere-like", "brother complex", "clumsy", "hostile"]
    neighbors = {}
    for term in neighbor_terms:
        if term not in descriptor_index:
            continue
        index = descriptor_index[term]
        similarities = embeddings @ embeddings[index]
        order = np.argsort(similarities)[::-1]
        neighbors[term] = [
            {"descriptor": descriptors[int(row)], "cosine_similarity": round(float(similarities[int(row)]), 8)}
            for row in order
            if int(row) != index
        ][:25]
    return {"diagnostic_pairs": pair_rows, "nearest_neighbors": neighbors}


def main() -> None:
    args = parse_args()
    preset = MODEL_PRESETS[args.preset]
    metadata = read_json(args.matrix_metadata)
    descriptors = metadata["descriptors"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    safe = slug(args.preset)
    npz_path = args.output_dir / f"{safe}_descriptor_embeddings.npz"
    json_path = args.output_dir / f"{safe}_descriptor_embeddings.json"
    report_path = args.output_dir / f"{safe}_diagnostics.json"
    if npz_path.exists() and json_path.exists() and not args.force:
        existing = read_json(json_path)
        if existing.get("descriptors") == descriptors and existing.get("preset") == args.preset:
            embeddings = np.load(npz_path)["embeddings"].astype(np.float32)
        else:
            raise RuntimeError(f"Existing cache metadata does not match {args.preset}; use --force.")
    else:
        if preset["backend"] == "sentence-transformers":
            embeddings = encode_sentence_transformers(
                descriptors,
                preset["model"],
                preset["prefix"],
                args.batch_size,
                bool(preset["trust_remote_code"]),
            )
        elif preset["backend"] == "ollama":
            embeddings = encode_ollama(descriptors, preset["model"], preset["prefix"], args.ollama_url)
        else:
            raise RuntimeError(f"Unknown backend: {preset['backend']}")
        np.savez_compressed(npz_path, embeddings=embeddings)
        write_json(
            json_path,
            {
                "generated_at": utc_now(),
                "preset": args.preset,
                "backend": preset["backend"],
                "model": preset["model"],
                "prefix": preset["prefix"],
                "trust_remote_code": preset["trust_remote_code"],
                "matrix_metadata": str(args.matrix_metadata),
                "descriptors": descriptors,
                "embedding_shape": list(embeddings.shape),
            },
        )

    report = {
        "generated_at": utc_now(),
        "preset": args.preset,
        "backend": preset["backend"],
        "model": preset["model"],
        "prefix": preset["prefix"],
        "embedding_cache": str(npz_path),
        "embedding_shape": list(embeddings.shape),
        **diagnostic_report(descriptors, embeddings),
    }
    write_json(report_path, report)
    print(f"wrote {npz_path}")
    print(f"wrote {json_path}")
    print(f"wrote {report_path}")
    print(json.dumps({"preset": args.preset, "embedding_shape": list(embeddings.shape)}, indent=2))
    print("diagnostic pairs")
    for row in report["diagnostic_pairs"]:
        if row.get("missing"):
            print(f"{row['left']} ~ {row['right']}: missing")
        else:
            print(f"{row['left']} ~ {row['right']}: {row['cosine_similarity']:.4f} ({row['expected']})")


if __name__ == "__main__":
    main()
