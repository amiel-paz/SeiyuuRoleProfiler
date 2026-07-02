#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_COMMUNITIES = (
    "anime",
    "hero",
    "villains",
    "jojo",
    "naruto",
    "onepiece",
    "bleach",
    "dragonball",
    "myheroacademia",
    "kimetsu-no-yaiba",
    "rezero",
    "typemoon",
    "konosuba",
    "danganronpa",
    "date-a-live",
    "fairytail",
    "gintama",
    "chainsaw-man",
    "jujutsu-kaisen",
    "spy-x-family",
)

FANDOM_TERMS_URLS = {
    "terms_of_use": "https://www.fandom.com/terms-of-use",
    "licensing": "https://www.fandom.com/licensing",
    "community_creation_policy": "https://www.fandom.com/community-creation-policy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Fandom API/domain reuse terms for public-facing seiyuu profiler enrichment."
    )
    parser.add_argument("--roles-input", type=Path, default=Path("data/role_edges.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/external/fandom_rights_audit"))
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-model", default="qwen3.5:4b")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--num-predict", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.4)
    parser.add_argument("--min-favourites", type=int, default=100)
    parser.add_argument("--max-title-communities", type=int, default=80)
    parser.add_argument("--limit-domains", type=int, default=20)
    parser.add_argument("--community", action="append", default=[], help="Extra Fandom community subdomain.")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--skip-llm", action="store_true")
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


def normalize_slug(value: str) -> str:
    value = html.unescape(value or "").lower()
    value = re.sub(r"['’]", "", value)
    value = re.sub(r"&", " and ", value)
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value


def short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def fetch_url(url: str, *, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "SeiyuuRoleProfiler/0.1 rights-audit contact: local-research",
            "Accept": "application/json,text/html;q=0.9,text/plain;q=0.8,*/*;q=0.5",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
        content_type = response.headers.get("content-type", "")
        text = raw.decode("utf-8", errors="replace")
        return {
            "url": url,
            "status": response.status,
            "content_type": content_type,
            "text": text,
            "retrieved_at": utc_now(),
        }


def strip_html(text: str) -> str:
    text = re.sub(r"<script\b[^>]*>.*?</script>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def snippets(text: str, patterns: list[str], window: int = 420) -> list[str]:
    clean = strip_html(text)
    found = []
    lowered = clean.lower()
    for pattern in patterns:
        idx = lowered.find(pattern.lower())
        if idx < 0:
            continue
        start = max(0, idx - window // 2)
        end = min(len(clean), idx + len(pattern) + window // 2)
        snippet = clean[start:end].strip()
        if snippet and snippet not in found:
            found.append(snippet)
    return found


def load_roles(path: Path) -> list[dict]:
    payload = read_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("roles"), list):
        return payload["roles"]
    if isinstance(payload, list):
        return payload
    raise TypeError(f"Unsupported roles payload at {path}")


def title_candidates(roles: list[dict], min_favourites: int, limit: int) -> list[dict]:
    counts: Counter[str] = Counter()
    examples: dict[str, set[str]] = {}
    for role in roles:
        character = role.get("character", {})
        if int(character.get("favourites") or 0) < min_favourites:
            continue
        titles = [character.get("first_anime") or ""]
        titles.extend(anime.get("title") or "" for anime in role.get("anime") or [])
        for title in titles:
            slug = normalize_slug(title)
            if not slug or len(slug) < 3:
                continue
            counts[slug] += 1
            examples.setdefault(slug, set()).add(title)
    rows = []
    for slug, count in counts.most_common(limit):
        rows.append({"community": slug, "source": "local_title_slug", "count": count, "examples": sorted(examples[slug])[:5]})
    return rows


def community_url(community: str) -> str:
    return f"https://{community}.fandom.com"


def api_url(community: str, params: dict[str, str]) -> str:
    return f"{community_url(community)}/api.php?{urllib.parse.urlencode(params)}"


def raw_cache_path(output_dir: Path, key: str, url: str) -> Path:
    return output_dir / "raw" / key / f"{short_hash(url)}.json"


def cached_fetch(output_dir: Path, key: str, url: str, args: argparse.Namespace) -> dict:
    path = raw_cache_path(output_dir, key, url)
    if path.exists() and not args.refresh:
        return read_json(path)
    if args.offline:
        raise FileNotFoundError(f"missing raw cache for {url}")
    try:
        response = fetch_url(url, timeout=args.timeout)
        cached = {"ok": True, **response}
    except urllib.error.HTTPError as exc:
        cached = {"ok": False, "url": url, "status": exc.code, "error": str(exc), "retrieved_at": utc_now()}
    except Exception as exc:
        cached = {"ok": False, "url": url, "status": None, "error": repr(exc), "retrieved_at": utc_now()}
    write_json(path, cached)
    time.sleep(args.sleep_seconds)
    return cached


def fetch_global_terms(args: argparse.Namespace) -> dict[str, dict]:
    output = {}
    for key, url in FANDOM_TERMS_URLS.items():
        output[key] = cached_fetch(args.output_dir, "global_terms", url, args)
    return output


def fetch_siteinfo(community: str, args: argparse.Namespace) -> dict:
    params = {
        "action": "query",
        "meta": "siteinfo",
        "siprop": "general|rightsinfo",
        "format": "json",
        "formatversion": "2",
    }
    return cached_fetch(args.output_dir, "siteinfo", api_url(community, params), args)


def fetch_robots(community: str, args: argparse.Namespace) -> dict:
    return cached_fetch(args.output_dir, "robots", f"{community_url(community)}/robots.txt", args)


def relevant_evidence(global_terms: dict[str, dict], siteinfo: dict, robots: dict) -> dict:
    terms_patterns = [
        "automated",
        "scrape",
        "scraping",
        "Artificial Intelligence",
        "machine learning",
        "prior written consent",
        "Creative Commons",
        "CC BY-SA",
        "attribution",
        "commercial",
    ]
    evidence = {"global_terms": {}, "siteinfo": {}, "robots": {}}
    for key, row in global_terms.items():
        evidence["global_terms"][key] = {
            "url": row.get("url"),
            "status": row.get("status"),
            "snippets": snippets(row.get("text") or "", terms_patterns),
        }
    if siteinfo.get("ok"):
        try:
            body = json.loads(siteinfo.get("text") or "{}")
        except json.JSONDecodeError:
            body = {}
        query = body.get("query") or {}
        evidence["siteinfo"] = {
            "url": siteinfo.get("url"),
            "status": siteinfo.get("status"),
            "general": query.get("general") or {},
            "rightsinfo": query.get("rightsinfo") or {},
        }
    else:
        evidence["siteinfo"] = {
            "url": siteinfo.get("url"),
            "status": siteinfo.get("status"),
            "error": siteinfo.get("error"),
        }
    evidence["robots"] = {
        "url": robots.get("url"),
        "status": robots.get("status"),
        "snippets": snippets(robots.get("text") or "", ["api.php", "Disallow", "Allow", "User-agent"], window=360),
    }
    return evidence


def prompt_for_domain(community: str, candidate: dict, evidence: dict) -> str:
    return f"""You are auditing whether a Fandom wiki API source can be used in a public-facing, non-monetized hosted seiyuu profiler.

The profiler would query/cache API text or metadata, use an LLM to derive compact character descriptors, and display only derived descriptors plus source attribution links. It would not display raw wiki prose.

Return a cautious, evidence-based JSON classification. Do not provide legal advice. If platform terms conflict with the content license, mark the source as not recommended or requires permission.
Because the intended use includes LLM descriptor derivation, if llm_descriptor_derivation_ok is "no", the decision must be "requires_permission", "not_recommended", or "blocked"; it must not be "usable_with_attribution".

Fandom community:
{json.dumps(candidate, ensure_ascii=False, indent=2)}

Evidence:
{json.dumps(evidence, ensure_ascii=False, indent=2)[:18000]}

Return only JSON with this shape:
{{
  "community": "{community}",
  "decision": "usable_with_attribution" | "requires_permission" | "not_recommended" | "blocked" | "unknown",
  "public_hosted_project": "yes" | "no" | "conditional" | "unknown",
  "api_access_ok": "yes" | "no" | "conditional" | "unknown",
  "llm_descriptor_derivation_ok": "yes" | "no" | "conditional" | "unknown",
  "raw_text_redistribution_ok": "yes" | "no" | "conditional" | "unknown",
  "required_conditions": ["condition"],
  "blocking_or_risky_terms": ["term or clause summary"],
  "attribution_requirements": ["requirement"],
  "evidence": [
    {{
      "source": "terms_of_use|licensing|community_creation_policy|siteinfo|robots",
      "url": "source URL",
      "quote": "short exact quote or tightly copied snippet from the supplied evidence",
      "supports": "what this quote supports"
    }}
  ],
  "confidence": "high" | "medium" | "low",
  "rationale": "one short paragraph"
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
                "content": "You are a strict policy-classification assistant. Return valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "format": "json",
        "think": False,
        "options": {
            "temperature": args.temperature,
            "seed": args.seed,
            "num_predict": args.num_predict,
            "num_ctx": args.num_ctx,
        },
    }
    request = urllib.request.Request(
        f"{args.ollama_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        raw = json.loads(response.read().decode("utf-8"))
    parsed = parse_json_object(raw.get("message", {}).get("content", ""))
    return {"raw": raw, "parsed": parsed}


def normalize_policy_decision(parsed: dict) -> dict:
    """Apply conservative consistency checks for this project's intended use."""
    parsed = dict(parsed)
    llm_ok = str(parsed.get("llm_descriptor_derivation_ok") or "").lower()
    decision = str(parsed.get("decision") or "").lower()
    if llm_ok == "no" and decision == "usable_with_attribution":
        parsed["decision"] = "requires_permission"
        parsed["public_hosted_project"] = "conditional"
        parsed.setdefault("required_conditions", [])
        permission_note = "Prior written permission for LLM descriptor derivation from Fandom content."
        if permission_note not in parsed["required_conditions"]:
            parsed["required_conditions"].append(permission_note)
        rationale = parsed.get("rationale") or ""
        parsed["rationale"] = (
            "Conservative override: the model marked LLM descriptor derivation as not allowed, "
            "so this source is not treated as usable with attribution alone. " + rationale
        ).strip()
    return parsed


def llm_cache_path(output_dir: Path, community: str) -> Path:
    return output_dir / "llm_domain_audits" / f"{community}.json"


def audit_domain(candidate: dict, global_terms: dict[str, dict], args: argparse.Namespace) -> dict:
    community = candidate["community"]
    siteinfo = fetch_siteinfo(community, args)
    robots = fetch_robots(community, args)
    evidence = relevant_evidence(global_terms, siteinfo, robots)
    result = {
        "generated_at": utc_now(),
        "community": community,
        "community_url": community_url(community),
        "candidate": candidate,
        "evidence": evidence,
        "llm_audit": None,
    }
    if not args.skip_llm:
        path = llm_cache_path(args.output_dir, community)
        if path.exists() and not args.refresh:
            result["llm_audit"] = read_json(path)
        elif args.offline:
            result["llm_audit"] = {"error": f"missing cached LLM audit for {community}"}
        else:
            llm = call_ollama(args, prompt_for_domain(community, candidate, evidence))
            parsed = normalize_policy_decision(llm["parsed"])
            audit = {
                "generated_at": utc_now(),
                "ollama_model": args.ollama_model,
                "temperature": args.temperature,
                "seed": args.seed,
                "parsed": parsed,
                "raw_message": llm["raw"].get("message", {}),
            }
            write_json(path, audit)
            result["llm_audit"] = audit
    return result


def build_candidates(args: argparse.Namespace) -> list[dict]:
    roles = load_roles(args.roles_input)
    candidates = [
        {"community": community, "source": "default_allowlist", "count": None, "examples": []}
        for community in DEFAULT_COMMUNITIES
    ]
    candidates.extend(title_candidates(roles, args.min_favourites, args.max_title_communities))
    candidates.extend({"community": normalize_slug(value), "source": "cli", "count": None, "examples": []} for value in args.community)

    by_community = {}
    for candidate in candidates:
        community = candidate["community"]
        if not community:
            continue
        current = by_community.get(community)
        if current is None:
            by_community[community] = candidate
            continue
        current["source"] = "+".join(sorted(set(current["source"].split("+") + candidate["source"].split("+"))))
        current["count"] = max(current.get("count") or 0, candidate.get("count") or 0) or None
        current["examples"] = sorted(set(current.get("examples") or []).union(candidate.get("examples") or []))[:8]

    return sorted(
        by_community.values(),
        key=lambda row: (
            0 if "default_allowlist" in row["source"] else 1,
            -(row.get("count") or 0),
            row["community"],
        ),
    )[: args.limit_domains]


def build_summary(audits: list[dict]) -> dict:
    counts = Counter()
    rows = []
    for audit in audits:
        parsed = ((audit.get("llm_audit") or {}).get("parsed") or {})
        decision = parsed.get("decision") or "not_run"
        counts[decision] += 1
        rows.append(
            {
                "community": audit["community"],
                "community_url": audit["community_url"],
                "decision": decision,
                "public_hosted_project": parsed.get("public_hosted_project"),
                "api_access_ok": parsed.get("api_access_ok"),
                "llm_descriptor_derivation_ok": parsed.get("llm_descriptor_derivation_ok"),
                "confidence": parsed.get("confidence"),
                "rationale": parsed.get("rationale"),
                "evidence": parsed.get("evidence") or [],
            }
        )
    return {"generated_at": utc_now(), "counts": dict(counts), "domains": rows}


def main() -> None:
    args = parse_args()
    candidates = build_candidates(args)
    print(json.dumps({"candidate_count": len(candidates), "candidates": candidates[:10]}, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    global_terms = fetch_global_terms(args)
    audits = []
    for index, candidate in enumerate(candidates, start=1):
        audit = audit_domain(candidate, global_terms, args)
        audits.append(audit)
        write_json(args.output_dir / "domain_audits" / f"{candidate['community']}.json", audit)
        parsed = ((audit.get("llm_audit") or {}).get("parsed") or {})
        print(
            json.dumps(
                {
                    "index": index,
                    "community": candidate["community"],
                    "decision": parsed.get("decision") or "not_run",
                    "public_hosted_project": parsed.get("public_hosted_project"),
                    "confidence": parsed.get("confidence"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    write_json(args.output_dir / "summary.json", build_summary(audits))


if __name__ == "__main__":
    main()
