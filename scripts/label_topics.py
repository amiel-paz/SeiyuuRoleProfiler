#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate cached English labels for NMF topics using Ollama.")
    parser.add_argument("--model-dir", type=Path, default=Path("models/k96_pos_descriptors"))
    parser.add_argument("--k", type=int, default=96)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-model", default="qwen3.5:4b")
    parser.add_argument("--top-terms", type=int, default=30)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--topic-index", type=int, action="append", default=None)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dry-run", action="store_true")
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


def weighted_terms(model_dir: Path, k: int, top_terms: int) -> dict[int, list[dict[str, float | str]]]:
    vocab = [row["ngram"] for row in read_json(model_dir / "tfidf_vocabulary.json")]
    matrices = np.load(model_dir / f"nmf_k{k:03d}_matrices.npz")
    components = matrices["topic_ngram_components"]
    output: dict[int, list[dict[str, float | str]]] = {}
    for topic_index, component in enumerate(components):
        order = np.argsort(component)[::-1]
        max_weight = float(component[int(order[0])]) if len(order) else 0.0
        terms = []
        for term_index in order[:top_terms]:
            weight = float(component[int(term_index)])
            if weight <= 0:
                break
            terms.append(
                {
                    "ngram": vocab[int(term_index)],
                    "weight": round(weight, 6),
                    "relative_weight": round(weight / max_weight, 6) if max_weight > 0 else 0.0,
                }
            )
        output[topic_index] = terms
    return output


def prompt_for_topic(topic_index: int, terms: list[dict[str, float | str]]) -> str:
    term_lines = "\n".join(
        f"- {row['ngram']}: weight={row['weight']}, relative={row['relative_weight']}"
        for row in terms
    )
    return f"""You label NMF topics from character-description TF-IDF n-grams.

Only use the weighted n-grams below. Do not infer from anime titles, character names, voice actors, or outside knowledge.
Write a compact semantic description of the shared descriptor pattern.

Topic index: {topic_index}
Weighted n-grams:
{term_lines}

Return only JSON with this shape:
{{
  "label": "2 to 6 words, noun phrase, no colon",
  "description": "one plain-English sentence under 24 words",
  "confidence": "high|medium|low",
  "evidence_terms": ["3 to 8 exact n-grams from the input"]
}}
"""


def parse_json_object(value: str) -> dict:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        start = value.find("{")
        end = value.rfind("}")
        if start >= 0 and end > start:
            return json.loads(value[start : end + 1])
        raise


def call_ollama(args: argparse.Namespace, prompt: str) -> dict:
    payload = {
        "model": args.ollama_model,
        "messages": [
            {
                "role": "system",
                "content": "You are a precise data-labeling assistant. Return valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": args.temperature,
            "seed": args.seed,
        },
    }
    request = urllib.request.Request(
        f"{args.ollama_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Could not reach Ollama at {args.ollama_url}. Start Ollama and run `ollama pull {args.ollama_model}`."
        ) from error
    content = result.get("message", {}).get("content", "")
    if not content:
        raise RuntimeError(f"Ollama returned no message content: {result}")
    return parse_json_object(content)


def validate_label(raw: dict, terms: list[dict[str, float | str]]) -> dict:
    allowed_terms = {str(row["ngram"]) for row in terms}
    evidence_terms = [str(term) for term in raw.get("evidence_terms", []) if str(term) in allowed_terms]
    confidence = str(raw.get("confidence", "medium")).lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"
    return {
        "label": str(raw.get("label", "")).strip()[:80],
        "description": str(raw.get("description", "")).strip()[:240],
        "confidence": confidence,
        "evidence_terms": evidence_terms[:8],
    }


def selected_topics(all_terms: dict[int, list[dict[str, float | str]]], args: argparse.Namespace) -> list[int]:
    if args.topic_index:
        topics = [topic for topic in args.topic_index if topic in all_terms]
    else:
        topics = sorted(all_terms)
    return topics[: args.limit] if args.limit else topics


def main() -> None:
    args = parse_args()
    output = args.output or args.model_dir / "topic_labels.json"
    all_terms = weighted_terms(args.model_dir, args.k, args.top_terms)
    topics = selected_topics(all_terms, args)
    if args.dry_run:
        topic_index = topics[0]
        print(prompt_for_topic(topic_index, all_terms[topic_index]))
        return

    labels = []
    for topic_index in topics:
        terms = all_terms[topic_index]
        prompt = prompt_for_topic(topic_index, terms)
        label = validate_label(call_ollama(args, prompt), terms)
        labels.append({"topic_index": topic_index, "weighted_terms": terms, **label})
        print(f"T{topic_index:02d}: {label['label']}", flush=True)

    write_json(
        output,
        {
            "generated_at": utc_now(),
            "source": "label_topics.py",
            "parameters": {
                "k": args.k,
                "ollama_model": args.ollama_model,
                "top_terms": args.top_terms,
                "temperature": args.temperature,
                "seed": args.seed,
                "characters_in_prompt": False,
            },
            "labels": labels,
        },
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
