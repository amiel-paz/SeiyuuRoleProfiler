#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from fit_topics import read_json, utc_now, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache the character x descriptor n-gram TF-IDF matrix.")
    parser.add_argument("--legal-ngrams-input", type=Path, default=None)
    parser.add_argument("--character-rows-input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("models/k96_pos_adj1to6_descriptors"))
    parser.add_argument("--min-df", type=int, default=3)
    parser.add_argument("--max-df", type=float, default=0.45)
    parser.add_argument("--max-features", type=int, default=30000)
    return parser.parse_args()


def identity_analyzer(value: list[str]) -> list[str]:
    return value


def vectorize_documents(documents: list[dict[str, Any]], args: argparse.Namespace):
    vectorizer = TfidfVectorizer(
        analyzer=identity_analyzer,
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        sublinear_tf=True,
        norm="l2",
        lowercase=False,
    )
    tokenized = [list(document["ngrams"]) for document in documents]
    matrix = vectorizer.fit_transform(tokenized).astype(np.float32)
    vocab = np.asarray(vectorizer.get_feature_names_out())
    return matrix, vocab, vectorizer.idf_


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    legal_path = args.legal_ngrams_input or args.output_dir / "legal_ngrams.json"
    rows_path = args.character_rows_input or args.output_dir / "character_rows.json"
    legal_payload = read_json(legal_path)
    rows = read_json(rows_path)
    documents = legal_payload["documents"]
    if len(rows) != len(documents):
        raise RuntimeError(f"Row count {len(rows)} does not match legal-ngram document count {len(documents)}")
    for row, document in zip(rows, documents, strict=True):
        if int(row["character_id"]) != int(document["character_id"]):
            raise RuntimeError("character_rows.json and legal_ngrams.json are not in the same character order")

    matrix, vocab, idf = vectorize_documents(documents, args)
    sparse.save_npz(args.output_dir / "character_ngram_tfidf.npz", matrix)
    write_json(
        args.output_dir / "tfidf_vocabulary.json",
        [{"index": int(index), "ngram": str(term), "idf": round(float(idf[index]), 6)} for index, term in enumerate(vocab)],
    )
    write_json(
        args.output_dir / "character_ngram_matrix_metadata.json",
        {
            "generated_at": utc_now(),
            "source": "cache_character_ngram_matrix.py",
            "inputs": {
                "legal_ngrams": str(legal_path),
                "character_rows": str(rows_path),
            },
            "parameters": {
                "min_df": args.min_df,
                "max_df": args.max_df,
                "max_features": args.max_features,
                "sublinear_tf": True,
                "norm": "l2",
            },
            "matrix": {
                "file": "character_ngram_tfidf.npz",
                "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
                "nonzero_entries": int(matrix.nnz),
            },
            "counts": {
                "characters": len(rows),
                "features": int(len(vocab)),
            },
        },
    )
    print(f"wrote {args.output_dir / 'character_ngram_tfidf.npz'}")
    print(f"wrote {args.output_dir / 'character_ngram_matrix_metadata.json'}")


if __name__ == "__main__":
    main()
