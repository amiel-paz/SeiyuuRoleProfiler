#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from fit_topics import (
    adjective_modifier_suppression_set,
    build_corpus,
    descriptor_analyzer_factory,
    import_nltk,
    read_json,
    utc_now,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache legal descriptor n-grams for character descriptions.")
    parser.add_argument("--characters-input", type=Path, required=True)
    parser.add_argument("--descriptions-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("models/k96_pos_adj1to6_descriptors"))
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=6)
    parser.add_argument("--min-token-chars", type=int, default=3)
    parser.add_argument("--adjective-modifier-suppression-threshold", type=float, default=0.80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, corpus, corpus_stats = build_corpus(read_json(args.characters_input), read_json(args.descriptions_input))
    nltk = import_nltk()
    suppressed_adjectives, adjective_filter_stats = adjective_modifier_suppression_set(
        nltk,
        corpus,
        args.min_token_chars,
        args.ngram_max,
        args.adjective_modifier_suppression_threshold,
    )
    analyzer = descriptor_analyzer_factory(
        nltk,
        args.ngram_min,
        args.ngram_max,
        args.min_token_chars,
        suppressed_adjectives,
    )

    documents = []
    document_frequency: Counter[str] = Counter()
    token_frequency: Counter[str] = Counter()
    for row, description in zip(rows, corpus, strict=True):
        ngrams = analyzer(description)
        counts = Counter(ngrams)
        document_frequency.update(counts.keys())
        token_frequency.update(counts)
        documents.append(
            {
                "character_id": row["character_id"],
                "ngrams": ngrams,
                "unique_ngrams": len(counts),
                "ngram_count": len(ngrams),
            }
        )

    legal_payload = {
        "generated_at": utc_now(),
        "source": "cache_legal_ngrams.py",
        "parameters": {
            "ngram_range": [args.ngram_min, args.ngram_max],
            "min_token_chars": args.min_token_chars,
            "adjective_modifier_suppression_threshold": args.adjective_modifier_suppression_threshold,
            "feature_mode": "standalone adjective unigrams plus noun/gerund phrase n-grams; adjective-modified multi-word chunks suppress but do not emit",
        },
        "corpus": corpus_stats,
        "counts": {
            "documents": len(documents),
            "raw_unique_ngrams": len(document_frequency),
            "raw_ngram_occurrences": int(sum(token_frequency.values())),
        },
        "documents": documents,
    }
    raw_vocab = [
        {
            "ngram": ngram,
            "document_frequency": int(document_frequency[ngram]),
            "occurrences": int(token_frequency[ngram]),
        }
        for ngram in sorted(document_frequency)
    ]

    write_json(args.output_dir / "character_rows.json", rows)
    write_json(args.output_dir / "legal_ngrams.json", legal_payload)
    write_json(args.output_dir / "legal_ngram_vocabulary_raw.json", raw_vocab)
    write_json(args.output_dir / "suppressed_adjectives.json", adjective_filter_stats)
    print(f"wrote {args.output_dir / 'legal_ngrams.json'}")
    print(f"wrote {args.output_dir / 'legal_ngram_vocabulary_raw.json'}")


if __name__ == "__main__":
    main()
