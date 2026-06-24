#!/usr/bin/env python3

from __future__ import annotations

import argparse
import html
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the SeiyuuRoleProfiler web payload.")
    parser.add_argument("--role-cache", type=Path, default=Path("data/role_edges.json"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/k96_pos_descriptors"))
    parser.add_argument("--k", type=int, default=96)
    parser.add_argument("--output-dir", type=Path, default=Path("site"))
    parser.add_argument("--top-lanes", type=int, default=16)
    parser.add_argument("--top-characters", type=int, default=16)
    parser.add_argument("--min-topic-proportion", type=float, default=0.03)
    parser.add_argument("--smoothing", type=float, default=0.0025)
    parser.add_argument("--label-mass", type=float, default=0.90)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def norm_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def search_aliases(name: str, native_name: str = "") -> list[str]:
    aliases = {norm_name(name), norm_name(native_name)}
    parts = [part for part in norm_name(name).split() if part]
    if len(parts) >= 2:
        aliases.add(" ".join(reversed(parts)))
    return sorted(alias for alias in aliases if alias)


def topic_terms_for_mass(component: np.ndarray, vocab: list[str], label_mass: float) -> tuple[list[str], float]:
    total = float(component.sum())
    if total <= 0:
        return [], 0.0
    output: list[str] = []
    cumulative = 0.0
    for index in np.argsort(component)[::-1]:
        value = float(component[index])
        if value <= 0:
            break
        output.append(vocab[int(index)])
        cumulative += value / total
        if cumulative >= label_mass:
            break
    return output, cumulative


def compact_topic_catalog(model_dir: Path, k: int, label_mass: float) -> dict[str, dict]:
    vocab = [row["ngram"] for row in read_json(model_dir / "tfidf_vocabulary.json")]
    summary = read_json(model_dir / f"nmf_k{k:03d}_summary.json")
    matrices = np.load(model_dir / f"nmf_k{k:03d}_matrices.npz")
    components = matrices["topic_ngram_components"]
    topics: dict[str, dict] = {}
    for topic in summary["topics"]:
        topic_index = int(topic["topic_index"])
        mass_terms, cumulative = topic_terms_for_mass(components[topic_index], vocab, label_mass)
        topics[str(topic_index)] = {
            "topic_index": topic_index,
            "top_terms": [term["ngram"] for term in topic["top_terms"]],
            "mass_terms": mass_terms,
            "mass_term_count": len(mass_terms),
            "mass_covered": round(cumulative, 4),
        }
    return topics


def character_payload(role: dict, score: float) -> dict:
    character = role["character"]
    anime = role.get("anime") or []
    first_anime = anime[0] if anime else {}
    return {
        "character_id": character["character_id"],
        "name": character.get("name") or "",
        "native_name": character.get("native_name") or "",
        "image": character.get("image") or "",
        "site_url": character.get("site_url") or "",
        "favourites": character.get("favourites") or 0,
        "role": role.get("character_role") or "",
        "score": round(float(score), 5),
        "anime_title": first_anime.get("title") or character.get("first_anime") or "",
        "anime_url": first_anime.get("mal_url") or first_anime.get("site_url") or "",
        "all_anime": anime,
    }


def build_profiles(args: argparse.Namespace) -> dict:
    role_cache = read_json(args.role_cache)
    rows = read_json(args.model_dir / "character_rows.json")
    matrices = np.load(args.model_dir / f"nmf_k{args.k:03d}_matrices.npz")
    proportions = matrices["character_topic_proportions"]
    character_ids = matrices["character_ids"]
    character_index = {int(character_id): index for index, character_id in enumerate(character_ids)}
    global_mean = proportions.mean(axis=0)
    topics = compact_topic_catalog(args.model_dir, args.k, args.label_mass)

    roles_by_seiyuu: dict[int, list[dict]] = defaultdict(list)
    for role in role_cache["roles"]:
        if int(role["character"]["character_id"]) in character_index:
            roles_by_seiyuu[int(role["seiyuu"]["seiyuu_id"])].append(role)

    profiles = []
    for seiyuu_id, roles in sorted(roles_by_seiyuu.items(), key=lambda item: item[1][0]["seiyuu"]["name"]):
        indices = np.asarray([character_index[int(role["character"]["character_id"])] for role in roles], dtype=np.int64)
        local = proportions[indices]
        local_mean = local.mean(axis=0)
        seiyuu = roles[0]["seiyuu"]
        lanes = []
        for topic_index in range(local.shape[1]):
            support_mask = local[:, topic_index] >= args.min_topic_proportion
            support = int(support_mask.sum())
            if support == 0:
                continue
            enrichment = (float(local_mean[topic_index]) + args.smoothing) / (float(global_mean[topic_index]) + args.smoothing)
            lane_score = enrichment * math.log1p(support)
            evidence_order = np.argsort(local[:, topic_index])[::-1]
            evidence = [
                character_payload(roles[int(position)], float(local[int(position), topic_index]))
                for position in evidence_order[: args.top_characters]
                if float(local[int(position), topic_index]) > 0
            ]
            lanes.append(
                {
                    "topic_index": topic_index,
                    "score": round(float(lane_score), 5),
                    "enrichment": round(float(enrichment), 5),
                    "mean": round(float(local_mean[topic_index]), 5),
                    "global_mean": round(float(global_mean[topic_index]), 5),
                    "support": support,
                    "characters": evidence,
                }
            )
        lanes.sort(key=lambda lane: lane["score"], reverse=True)
        profiles.append(
            {
                "seiyuu_id": seiyuu_id,
                "name": seiyuu.get("name") or "",
                "native_name": seiyuu.get("native_name") or "",
                "image": seiyuu.get("image") or "",
                "site_url": seiyuu.get("site_url") or "",
                "role_count": seiyuu.get("role_count") or 0,
                "character_count": len(roles),
                "first_year": seiyuu.get("first_year"),
                "aliases": search_aliases(seiyuu.get("name") or "", seiyuu.get("native_name") or ""),
                "top_score": lanes[0]["score"] if lanes else 0,
                "lanes": lanes[: args.top_lanes],
            }
        )

    samples = sorted(
        [
            {
                "name": profile["name"],
                "native_name": profile["native_name"],
                "image": profile["image"],
                "top_score": profile["top_score"],
                "role_count": profile["role_count"],
            }
            for profile in profiles
            if profile["lanes"]
        ],
        key=lambda row: (row["top_score"], row["role_count"]),
        reverse=True,
    )[:40]

    return {
        "generated_at": utc_now(),
        "source": "build_profiles.py",
        "parameters": {
            "k": args.k,
            "topic_ranking": "(seiyuu mean topic proportion + smoothing) / (global mean topic proportion + smoothing) * log1p(support)",
            "label_mass": args.label_mass,
            "min_topic_proportion": args.min_topic_proportion,
        },
        "counts": {
            "profiles": len(profiles),
            "topics": len(topics),
            "roles": sum(len(roles) for roles in roles_by_seiyuu.values()),
        },
        "topics": topics,
        "samples": samples,
        "profiles": profiles,
    }


def html_page() -> str:
    return r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SeiyuuRoleProfiler</title>
  <style>
    :root { color-scheme: light; --ink:#121b17; --muted:#5f6f69; --line:#dce5e1; --soft:#eef5f2; --accent:#a83d57; --bg:#fbfcfb; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:var(--bg); color:var(--ink); }
    a { color: inherit; text-decoration: none; }
    .shell { max-width: 1120px; margin: 0 auto; padding: 32px 22px 64px; }
    .searchHome { min-height: 78vh; display:grid; place-items:center; }
    .searchBox { width:min(720px, 100%); position:relative; }
    h1 { font-size: clamp(40px, 7vw, 84px); letter-spacing:0; margin: 0 0 22px; line-height:.95; }
    input { width:100%; border:1px solid var(--line); border-radius: 8px; padding: 20px 22px; font-size: 28px; font-weight: 750; color:var(--ink); background:white; outline:none; }
    input:focus { border-color:#b7c7c0; box-shadow:0 0 0 4px rgba(64,96,84,.1); }
    .suggestions { position:absolute; left:0; right:0; top: calc(100% + 8px); background:white; border:1px solid var(--line); border-radius:8px; overflow:hidden; z-index:5; box-shadow:0 18px 50px rgba(24,34,30,.12); }
    .suggestion { display:flex; gap:12px; align-items:center; padding:10px 12px; cursor:pointer; border-bottom:1px solid #eef2f0; }
    .suggestion:last-child { border-bottom:0; }
    .suggestion:hover { background:var(--soft); }
    .suggestion img { width:42px; height:42px; border-radius:6px; object-fit:cover; background:var(--soft); }
    .samples { display:flex; flex-wrap:wrap; gap:8px; margin-top:18px; }
    .sample { border:1px solid var(--line); background:white; border-radius:999px; padding:7px 12px; font-size:14px; color:var(--muted); cursor:pointer; }
    .profileHead { display:grid; grid-template-columns: 140px 1fr; gap:28px; align-items:center; margin: 12px 0 34px; }
    .profileHead img { width:140px; height:140px; border-radius:8px; object-fit:cover; border:1px solid var(--line); background:var(--soft); }
    .profileHead h2 { font-size: clamp(42px, 7vw, 76px); margin:0 0 14px; line-height:.95; letter-spacing:0; }
    .native { color:var(--muted); }
    .chips { display:flex; flex-wrap:wrap; gap:10px; }
    .chip { background:var(--soft); border-radius:999px; padding:9px 14px; color:var(--muted); font-size:20px; }
    .topbar { margin-bottom: 26px; }
    .topbar button { border:1px solid var(--line); background:white; border-radius:8px; padding:10px 14px; cursor:pointer; color:var(--muted); }
    .lane { border-top:1px solid var(--line); padding:28px 0; }
    .laneTitle { display:grid; grid-template-columns: 1fr auto; gap:16px; align-items:end; }
    .terms { font-size: clamp(26px, 4vw, 44px); font-weight:850; line-height:1.05; }
    .score { color:var(--accent); font-size:34px; font-weight:850; white-space:nowrap; }
    .meta { color:var(--muted); margin:10px 0 18px; font-size:18px; }
    details { margin: 8px 0 16px; color:var(--muted); }
    summary { cursor:pointer; }
    .termList { margin-top:8px; font-size:14px; line-height:1.5; }
    .chars { display:flex; flex-wrap:wrap; gap:14px; }
    .char { position:relative; width:92px; }
    .char img { width:92px; height:92px; object-fit:cover; border-radius:8px; border:1px solid var(--line); background:var(--soft); display:block; }
    .char:after { content:attr(data-tooltip); position:absolute; left:50%; bottom:calc(100% + 8px); transform:translateX(-50%); background:#15201b; color:white; padding:8px 10px; border-radius:6px; width:max-content; max-width:260px; white-space:pre-line; opacity:0; pointer-events:none; transition:opacity .12s ease; font-size:13px; z-index:10; }
    .char:hover:after { opacity:1; }
    .empty { color:var(--muted); font-size:20px; margin-top:22px; }
    @media (max-width: 640px) {
      .profileHead { grid-template-columns: 92px 1fr; gap:16px; }
      .profileHead img { width:92px; height:92px; }
      input { font-size:22px; }
      .chip { font-size:16px; }
      .char, .char img { width:74px; height:74px; }
      .score { font-size:26px; }
    }
  </style>
</head>
<body>
  <main class="shell" id="app"></main>
  <script>
    const app = document.getElementById('app');
    let data;
    const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    const norm = value => String(value || '').toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
    const findProfile = q => {
      const nq = norm(q);
      if (!nq) return null;
      return data.profiles.find(p => p.aliases.includes(nq)) || data.profiles.find(p => p.aliases.some(a => a.includes(nq) || nq.includes(a)));
    };
    function setQuery(name) {
      history.pushState(null, '', `?q=${encodeURIComponent(name)}`);
      renderProfile(findProfile(name));
    }
    function searchMarkup() {
      return `<section class="searchHome"><div class="searchBox"><h1>SeiyuuRoleProfiler</h1><input id="search" autocomplete="off" placeholder="Seiyuu name" autofocus><div id="suggestions" class="suggestions" hidden></div><div class="samples">${data.samples.slice(0,16).map(s => `<button class="sample" data-name="${esc(s.name)}">${esc(s.name)}</button>`).join('')}</div></div></section>`;
    }
    function wireSearch() {
      const input = document.getElementById('search');
      const suggestions = document.getElementById('suggestions');
      const update = () => {
        const q = norm(input.value);
        const rows = q ? data.profiles.filter(p => p.aliases.some(a => a.includes(q))).slice(0, 8) : [];
        suggestions.hidden = rows.length === 0;
        suggestions.innerHTML = rows.map(p => `<div class="suggestion" data-name="${esc(p.name)}"><img src="${esc(p.image)}" alt=""><strong>${esc(p.name)}</strong><span class="native">${esc(p.native_name)}</span></div>`).join('');
      };
      input.addEventListener('input', update);
      input.addEventListener('keydown', event => {
        if (event.key === 'Enter') {
          const p = findProfile(input.value);
          if (p) setQuery(p.name);
        }
      });
      app.addEventListener('click', event => {
        const target = event.target.closest('[data-name]');
        if (target) setQuery(target.dataset.name);
      });
    }
    function renderHome() {
      app.innerHTML = searchMarkup();
      wireSearch();
    }
    function laneTerms(topic) {
      return topic.top_terms.slice(0, 8).join(' / ');
    }
    function renderProfile(profile) {
      if (!profile) {
        renderHome();
        return;
      }
      app.innerHTML = `<div class="topbar"><button id="back">Search</button></div>
        <section class="profileHead">
          <a href="${esc(profile.site_url)}" target="_blank" rel="noreferrer"><img src="${esc(profile.image)}" alt="${esc(profile.name)}"></a>
          <div><h2>${esc(profile.name)} <span class="native">${esc(profile.native_name)}</span></h2><div class="chips"><span class="chip">roles ${profile.role_count}</span><span class="chip">characters ${profile.character_count}</span><span class="chip">first year ${profile.first_year || ''}</span></div></div>
        </section>
        ${profile.lanes.map(lane => {
          const topic = data.topics[String(lane.topic_index)];
          return `<section class="lane">
            <div class="laneTitle"><div class="terms">${esc(laneTerms(topic))}</div><div class="score">${lane.enrichment.toFixed(2)}x</div></div>
            <div class="meta">support ${lane.support} · mean ${lane.mean.toFixed(3)} · global ${lane.global_mean.toFixed(3)} · topic ${lane.topic_index}</div>
            <details><summary>${topic.mass_term_count} n-grams cover ${(topic.mass_covered * 100).toFixed(1)}% of lane mass</summary><div class="termList">${esc(topic.mass_terms.join(', '))}</div></details>
            <div class="chars">${lane.characters.map(c => `<a class="char" href="${esc(c.site_url)}" target="_blank" rel="noreferrer" data-tooltip="${esc(c.name + '\\n' + c.anime_title)}"><img src="${esc(c.image)}" alt="${esc(c.name)}"></a>`).join('')}</div>
          </section>`;
        }).join('') || '<p class="empty">No lanes found.</p>'}`;
      document.getElementById('back').addEventListener('click', () => { history.pushState(null, '', location.pathname); renderHome(); });
    }
    fetch('profiles.json').then(r => r.json()).then(payload => {
      data = payload;
      const params = new URLSearchParams(location.search);
      const q = params.get('q');
      q ? renderProfile(findProfile(q)) : renderHome();
    });
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    payload = build_profiles(args)
    write_json(args.output_dir / "profiles.json", payload)
    write_text(args.output_dir / "index.html", html_page())
    print(f"wrote {args.output_dir / 'profiles.json'}")
    print(f"wrote {args.output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
