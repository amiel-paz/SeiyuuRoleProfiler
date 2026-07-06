#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROMPT_HEADER = """Judge whether each descriptor is suitable as a stable fictional-character PERSONALITY descriptor for an anime/seiyuu profiler.

KEEP descriptors that name stable temperament, demeanor, interpersonal style, moral disposition, recurring social attitude, or anime personality archetype. Anime archetypes such as tsundere, yandere, kuudere, dandere, chuunibyou, flirtatious, seductive, blunt, frank, kind, arrogant are valid if they describe how the character tends to behave. Stable temperaments such as bad-tempered, irritable, grouchy, guarded, reserved, shy, cheerful, and hot-headed are valid.

REJECT physical appearance/body descriptors, occupations, species, story roles, relationship labels, abilities/skills, nationalities/demonyms, ancestry/lineage/origin words (e.g. ancestral, hereditary), legal/status words, settings, age groups, generic meta words, audience-evaluation words (e.g. enjoyable, interesting, popular, worthy), moral/story-status words (e.g. redeemed, fallen, chosen) unless they directly name a stable way the character behaves, pure competence words (e.g. proficient, fluent), relational/structural words (e.g. constituent, voluntary), domain-only adjectives (e.g. forte, oral), and temporary emotions/reactions/states (e.g. angered, infuriated, ashamed, horrified, pleased, warmed).

The descriptor does NOT need to already appear on a character. Judge whether it is a reusable personality-basis word in the abstract, not whether it has current character support.

Return only JSON with one boolean for every input descriptor:
{"keep":{"descriptor":true}}
Descriptors: """


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache LLM decisions for production personality basis candidates.")
    parser.add_argument(
        "--basis",
        type=Path,
        default=Path("run/production_personality_basis/production_personality_basis.tsv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("run/production_personality_basis"))
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434/api/chat")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--request-timeout", type=int, default=240)
    parser.add_argument("--retry-count", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cache-name", default="llm_personality_decisions_v4.jsonl")
    return parser.parse_args()


def read_basis(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, dialect="excel-tab"))


def read_decisions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    decisions = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            decisions[str(row["descriptor"])] = row
    return decisions


def append_decisions(path: Path, decisions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in decisions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def call_ollama(args: argparse.Namespace, descriptors: list[str]) -> list[dict[str, Any]]:
    prompt = PROMPT_HEADER + json.dumps(descriptors, ensure_ascii=False)
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
                "num_predict": max(500, 18 * len(descriptors)),
            },
        }
    ).encode("utf-8")
    request = urllib.request.Request(args.ollama_url, data=body, headers={"Content-Type": "application/json"})
    last_error: Exception | None = None
    for attempt in range(1, args.retry_count + 1):
        try:
            with urllib.request.urlopen(request, timeout=args.request_timeout) as response:
                payload = json.loads(response.read())
            content = (payload.get("message") or {}).get("content") or payload.get("response") or ""
            parsed = json.loads(content)
            if isinstance(parsed.get("keep"), dict):
                by_descriptor = {
                    str(descriptor): {"keep": keep, "reason": ""}
                    for descriptor, keep in parsed["keep"].items()
                }
            else:
                rows = parsed.get("decisions") or []
                by_descriptor = {str(row.get("descriptor") or ""): row for row in rows}
            missing = [descriptor for descriptor in descriptors if descriptor not in by_descriptor]
            if missing and len(descriptors) > 1:
                raise RuntimeError(f"LLM response omitted {len(missing)} descriptors")
            output = []
            for descriptor in descriptors:
                row = by_descriptor.get(descriptor)
                if row is None:
                    output.append(
                        {
                            "descriptor": descriptor,
                            "llm_keep": False,
                            "llm_reason": "missing_from_llm_response",
                            "llm_model": args.model,
                        }
                    )
                else:
                    output.append(
                        {
                            "descriptor": descriptor,
                            "llm_keep": bool(row.get("keep")),
                            "llm_reason": str(row.get("reason") or ""),
                            "llm_model": args.model,
                        }
                    )
            return output
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as error:
            last_error = error
            time.sleep(2 * attempt)
    raise RuntimeError(f"Ollama decision batch failed after {args.retry_count} attempts: {last_error}")


def call_ollama_resilient(args: argparse.Namespace, descriptors: list[str]) -> list[dict[str, Any]]:
    try:
        return call_ollama(args, descriptors)
    except RuntimeError:
        if len(descriptors) <= 1:
            descriptor = descriptors[0]
            return [
                {
                    "descriptor": descriptor,
                    "llm_keep": False,
                    "llm_reason": "llm_decision_failed",
                    "llm_model": args.model,
                }
            ]
        midpoint = len(descriptors) // 2
        return call_ollama_resilient(args, descriptors[:midpoint]) + call_ollama_resilient(
            args,
            descriptors[midpoint:],
        )


def write_final_outputs(args: argparse.Namespace, basis_rows: list[dict[str, Any]], decisions: dict[str, dict[str, Any]]) -> None:
    merged = []
    for row in basis_rows:
        decision = decisions.get(row["descriptor"], {})
        output_row = dict(row)
        output_row["anchor_keep"] = output_row.pop("keep")
        merged.append(
            {
                **output_row,
                "llm_keep": bool(decision.get("llm_keep")),
                "llm_reason": decision.get("llm_reason") or "",
                "llm_model": decision.get("llm_model") or args.model,
                "final_keep": bool(decision.get("llm_keep")),
            }
        )
    merged.sort(key=lambda row: row["descriptor"])
    kept = [row for row in merged if row["final_keep"]]
    kept_public = [
        {
            key: value
            for key, value in row.items()
            if key not in {"anchor_keep", "llm_keep", "final_keep"}
        }
        for row in kept
    ]
    outputs = [
        (args.output_dir / "production_personality_basis_llm.tsv", "excel-tab", merged),
        (args.output_dir / "production_personality_basis_llm.csv", "excel", merged),
        (args.output_dir / "production_personality_basis_llm_kept.tsv", "excel-tab", kept_public),
        (args.output_dir / "production_personality_basis_llm_kept.csv", "excel", kept_public),
    ]
    for path, dialect, rows in outputs:
        fieldnames = list(rows[0].keys()) if rows else []
        with path.with_name(path.name + ".tmp").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, dialect=dialect)
            writer.writeheader()
            writer.writerows(rows)
        path.with_name(path.name + ".tmp").replace(path)
    summary = {
        "source": "cache_llm_personality_basis_decisions.py",
        "model": args.model,
        "basis": str(args.basis),
        "decision_count": len(decisions),
        "candidate_count": len(basis_rows),
        "final_kept": len(kept),
        "outputs": {
            "all_tsv": str(outputs[0][0]),
            "all_csv": str(outputs[1][0]),
            "kept_tsv": str(outputs[2][0]),
            "kept_csv": str(outputs[3][0]),
        },
    }
    (args.output_dir / "production_personality_basis_llm_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    basis_rows = read_basis(args.basis)
    descriptors = [row["descriptor"] for row in basis_rows]
    if args.limit:
        descriptors = descriptors[: args.limit]
    cache_path = args.output_dir / args.cache_name
    decisions = read_decisions(cache_path)
    pending = [descriptor for descriptor in descriptors if descriptor not in decisions]
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        rows = call_ollama_resilient(args, batch)
        append_decisions(cache_path, rows)
        decisions.update({row["descriptor"]: row for row in rows})
        print(f"cached {min(start + len(batch), len(pending))}/{len(pending)} pending decisions", flush=True)
    write_final_outputs(args, basis_rows, decisions)
    print(json.dumps({"decisions": len(decisions), "candidates": len(basis_rows)}, indent=2))


if __name__ == "__main__":
    main()
