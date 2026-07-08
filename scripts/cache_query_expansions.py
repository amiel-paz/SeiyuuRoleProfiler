#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROMPT_VERSION = "query_expansion_v2"

PROMPT_TEMPLATE = """Expand one anime character-personality query into reusable descriptor words.

Goal:
- Map a query such as "kuudere" onto a small set of stable fictional-character personality descriptors.
- Use only descriptors from the allowed vocabulary.
- Prefer specific descriptors over generic hubs.
- Include near-synonyms and trope-equivalent personality descriptors.
- Do not include occupations, roles, appearance terms, nationalities, species, temporary moods, or ability words.
- Keep emotional reserve, aloofness, shyness, and stoicism separate from moral coldness, cruelty, hostility, or malice unless the query explicitly asks for that darker meaning.
- If the exact query appears in the allowed vocabulary and is personality-like, include it with weight 1.0.
- Return 3 to 8 descriptors when possible.

Return only JSON in this shape:
{{
  "expanded": [
    {{"descriptor": "allowed_descriptor", "weight": 1.0, "reason": "short reason"}}
  ]
}}

Query: {query}

Allowed vocabulary:
{vocabulary}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache deterministic LLM query expansions into descriptor vocabulary.")
    parser.add_argument("queries", nargs="+", help="Query terms to expand, e.g. kuudere")
    parser.add_argument(
        "--descriptor-source",
        type=Path,
        default=Path("site/seiyuu_descriptor_map.json"),
        help="JSON payload with a descriptors list, or TSV/CSV with a descriptor column.",
    )
    parser.add_argument("--output", type=Path, default=Path("run/query_expansions/query_expansions.jsonl"))
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--ollama-chat-url", default="http://127.0.0.1:11434/api/chat")
    parser.add_argument("--ollama-tags-url", default="http://127.0.0.1:11434/api/tags")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--request-timeout", type=int, default=240)
    parser.add_argument("--retry-count", type=int, default=3)
    parser.add_argument("--force", action="store_true", help="Recompute even when a matching cache row exists.")
    return parser.parse_args()


def canonical_query(query: str) -> str:
    return " ".join(query.strip().lower().replace("_", " ").split())


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_descriptors(path: Path) -> list[str]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        descriptors = payload.get("descriptors")
        if not isinstance(descriptors, list):
            raise ValueError(f"{path} does not contain a top-level descriptors list")
        return sorted({str(descriptor).strip().lower() for descriptor in descriptors if str(descriptor).strip()})

    with path.open(newline="", encoding="utf-8") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,")
        reader = csv.DictReader(handle, dialect=dialect)
        if not reader.fieldnames or "descriptor" not in reader.fieldnames:
            raise ValueError(f"{path} must have a descriptor column")
        return sorted({str(row["descriptor"]).strip().lower() for row in reader if str(row.get("descriptor", "")).strip()})


def cache_key(
    query: str,
    descriptor_hash: str,
    model: str,
    model_digest: str,
    prompt_hash: str,
    temperature: float,
    seed: int,
) -> str:
    return sha256_text(
        json.dumps(
            {
                "query": query,
                "descriptor_hash": descriptor_hash,
                "model": model,
                "model_digest": model_digest,
                "prompt_version": PROMPT_VERSION,
                "prompt_hash": prompt_hash,
                "temperature": temperature,
                "seed": seed,
                "think": False,
            },
            sort_keys=True,
        )
    )


def read_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row.get("cache_key") or "")] = row
    return rows


def append_cache(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def model_digest(args: argparse.Namespace) -> str:
    request = urllib.request.Request(args.ollama_tags_url)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return ""
    for model in payload.get("models") or []:
        if model.get("name") == args.model or model.get("model") == args.model:
            return str(model.get("digest") or "")
    return ""


def call_ollama(args: argparse.Namespace, prompt: str) -> tuple[dict[str, Any], str]:
    body = json.dumps(
        {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",
            "think": False,
            "options": {
                "temperature": args.temperature,
                "top_p": 1,
                "seed": args.seed,
                "num_predict": 900,
            },
        }
    ).encode("utf-8")
    request = urllib.request.Request(args.ollama_chat_url, data=body, headers={"Content-Type": "application/json"})
    last_error: Exception | None = None
    for attempt in range(1, args.retry_count + 1):
        try:
            with urllib.request.urlopen(request, timeout=args.request_timeout) as response:
                payload = json.loads(response.read())
            content = (payload.get("message") or {}).get("content") or payload.get("response") or ""
            return json.loads(content), content
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            time.sleep(2 * attempt)
    raise RuntimeError(f"Ollama expansion failed after {args.retry_count} attempts: {last_error}")


def clean_expansion(parsed: dict[str, Any], allowed: set[str]) -> list[dict[str, Any]]:
    rows = parsed.get("expanded")
    if not isinstance(rows, list):
        return []
    cleaned = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        descriptor = canonical_query(str(row.get("descriptor") or ""))
        if descriptor not in allowed:
            dehyphenated = descriptor.replace("-", "")
            if dehyphenated in allowed:
                descriptor = dehyphenated
        if descriptor not in allowed or descriptor in seen:
            continue
        seen.add(descriptor)
        try:
            weight = float(row.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        cleaned.append(
            {
                "descriptor": descriptor,
                "weight": round(max(0.0, min(weight, 1.0)), 4),
                "reason": str(row.get("reason") or "")[:240],
            }
        )
    cleaned.sort(key=lambda row: (-row["weight"], row["descriptor"]))
    return cleaned


def ensure_exact_query(expanded: list[dict[str, Any]], query: str, allowed: set[str]) -> list[dict[str, Any]]:
    if query not in allowed:
        return expanded
    if any(row.get("descriptor") == query for row in expanded):
        return expanded
    return [
        {
            "descriptor": query,
            "weight": 1.0,
            "reason": "Exact query exists in the descriptor vocabulary.",
        },
        *expanded,
    ]


def main() -> None:
    args = parse_args()
    descriptors = read_descriptors(args.descriptor_source)
    allowed = set(descriptors)
    vocabulary = json.dumps(descriptors, ensure_ascii=False)
    descriptor_hash = sha256_text("\n".join(descriptors))
    digest = model_digest(args)
    cache = read_cache(args.output)

    for raw_query in args.queries:
        query = canonical_query(raw_query)
        prompt = PROMPT_TEMPLATE.format(query=json.dumps(query, ensure_ascii=False), vocabulary=vocabulary)
        prompt_hash = sha256_text(prompt)
        key = cache_key(query, descriptor_hash, args.model, digest, prompt_hash, args.temperature, args.seed)
        if key in cache and not args.force:
            row = cache[key]
            terms = ", ".join(item["descriptor"] for item in row.get("expanded", []))
            print(f"{query}: cached -> {terms}")
            continue
        parsed, raw_response = call_ollama(args, prompt)
        expanded = ensure_exact_query(clean_expansion(parsed, allowed), query, allowed)
        row = {
            "cache_key": key,
            "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "query": query,
            "expanded": expanded,
            "model": args.model,
            "model_digest": digest,
            "temperature": args.temperature,
            "seed": args.seed,
            "think": False,
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": prompt_hash,
            "descriptor_source": str(args.descriptor_source),
            "descriptor_count": len(descriptors),
            "descriptor_sha256": descriptor_hash,
            "raw_response": raw_response,
        }
        append_cache(args.output, row)
        terms = ", ".join(item["descriptor"] for item in expanded)
        print(f"{query}: wrote -> {terms}")


if __name__ == "__main__":
    main()
