#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import NMF


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit seiyuu-local descriptor NMF lanes, name each lane with a compact adjective, "
            "then decode local B@G@X SVD axes only through those lane adjectives."
        )
    )
    parser.add_argument("seiyuu", help="Romaji seiyuu query, e.g. 'Ayana Taketatsu'.")
    parser.add_argument("--site-profile-input", type=Path, default=Path("site/profiles.json"))
    parser.add_argument(
        "--tags-input",
        type=Path,
        default=Path("data/external/merged/all_characters_llm_vndb_personality_tags.json"),
    )
    parser.add_argument(
        "--safe-llm-tags",
        nargs="*",
        type=Path,
        default=[
            Path("data/external/safe_enrichment_llm/character_tags.jsonl"),
            Path("run/gpu_llm_tagging/returned_latest/character_tags.jsonl"),
        ],
    )
    parser.add_argument(
        "--safe-enrichment",
        type=Path,
        default=Path("data/external/safe_enrichment/character_safe_enrichment.jsonl"),
    )
    parser.add_argument(
        "--bangumi-raw-dir",
        type=Path,
        default=Path("data/external/safe_enrichment/raw/bangumi"),
    )
    parser.add_argument("--character-display-input", type=Path, default=Path("site/character_display.json"))
    parser.add_argument("--role-edges", type=Path, default=Path("data/role_edges.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/seiyuu_local_nmf_lane_svd"))
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=12)
    parser.add_argument("--top-lane-descriptors", type=int, default=12)
    parser.add_argument("--top-characters", type=int, default=12)
    parser.add_argument("--top-svd", type=int, default=4)
    parser.add_argument(
        "--row-weight",
        choices=["none", "sqrt_log_combined_favourites", "log_combined_favourites", "sqrt_combined_favourites"],
        default="sqrt_log_combined_favourites",
    )
    parser.add_argument("--normalize-b-rows", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalize-z-columns", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shared-role-weight", choices=["none", "inverse_sqrt", "inverse"], default="inverse_sqrt")
    parser.add_argument("--max-role-edge-count", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--max-iter", type=int, default=2000)
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
    return re.sub(r"[^a-z0-9]+", "_", (value or "").lower()).strip("_") or "value"


def norm_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def name_keys(value: str) -> set[str]:
    normalized = norm_name(value)
    parts = [part for part in normalized.split() if part]
    keys = {normalized}
    if len(parts) >= 2:
        keys.add(" ".join(reversed(parts)))
    return {key for key in keys if key}


def alias_keys(profile: dict) -> set[str]:
    keys = name_keys(profile.get("name") or "")
    for alias in profile.get("aliases") or []:
        keys.update(name_keys(alias))
    return keys


def is_pop_team_character(character: dict) -> bool:
    anime = norm_name(character.get("first_anime") or "")
    name = norm_name(character.get("name") or "")
    return "pop team epic" in anime or "poputepipikku" in anime or name in {"pipimi", "popuko"}


def shared_role_weight(character: dict, mode: str) -> float:
    count = max(int(character.get("role_edge_count") or 1), 1)
    if mode == "none":
        return 1.0
    if mode == "inverse_sqrt":
        return 1.0 / math.sqrt(count)
    if mode == "inverse":
        return 1.0 / count
    raise ValueError(f"unknown shared role weight mode: {mode}")


def character_row_weight(character: dict, mode: str) -> float:
    combined = max(float(character.get("combined_favourites") or character.get("favourites") or 0.0), 0.0)
    if mode == "none":
        return 1.0
    if mode == "sqrt_log_combined_favourites":
        return math.sqrt(math.log1p(combined) + 1.0)
    if mode == "log_combined_favourites":
        return math.log1p(combined) + 1.0
    if mode == "sqrt_combined_favourites":
        return math.sqrt(combined + 1.0)
    raise ValueError(f"unknown row weight mode: {mode}")


def single_adjective_shape(value: str) -> bool:
    return re.fullmatch(r"[a-z]+(?:-[a-z]+)*", (value or "").strip()) is not None


def descriptor_tokens(value: str) -> list[str]:
    return re.findall(r"[a-z]+(?:-[a-z]+)*", (value or "").lower())


def normalized_tag(value: str) -> str:
    value = re.sub(r"\s+", " ", (value or "").strip().lower())
    return value


def load_bangumi_collects(safe_enrichment_path: Path, raw_dir: Path) -> dict[int, dict]:
    selected_bangumi_by_anilist: dict[int, int] = {}
    if safe_enrichment_path.exists():
        with safe_enrichment_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                matches = (row.get("bangumi") or {}).get("matches") or []
                if matches:
                    selected_bangumi_by_anilist[int(row["anilist_character_id"])] = int(matches[0]["bangumi_character_id"])

    raw_by_bangumi: dict[int, dict] = {}
    if raw_dir.exists():
        for path in raw_dir.glob("*.json"):
            try:
                payload = read_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            for item in ((payload.get("response") or {}).get("data") or []):
                if item.get("id") is not None:
                    raw_by_bangumi[int(item["id"])] = item

    output: dict[int, dict] = {}
    for anilist_id, bangumi_id in selected_bangumi_by_anilist.items():
        stat = (raw_by_bangumi.get(bangumi_id) or {}).get("stat") or {}
        output[anilist_id] = {
            "bangumi_character_id": bangumi_id,
            "bangumi_url": f"https://bgm.tv/character/{bangumi_id}",
            "bangumi_collects": int(stat.get("collects") or 0),
            "bangumi_comments": int(stat.get("comments") or 0),
        }
    return output


def load_safe_llm_personality(paths: list[Path]) -> dict[int, list[dict]]:
    by_id: dict[int, list[dict]] = defaultdict(list)
    seen: set[tuple[int, str, str]] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                character_id = int(row.get("anilist_character_id") or row.get("character_id") or 0)
                if not character_id:
                    continue
                tags = ((row.get("tags") or {}).get("personality") or [])
                for tag in tags:
                    descriptor = normalized_tag(tag.get("tag") or "")
                    if not descriptor:
                        continue
                    source_key = str(tag.get("source_key") or tag.get("source_url") or path)
                    key = (character_id, descriptor, source_key)
                    if key in seen:
                        continue
                    seen.add(key)
                    by_id[character_id].append(
                        {
                            "tag": descriptor,
                            "category": "personality",
                            "source": "safe_llm",
                            "source_key": tag.get("source_key"),
                            "source_url": tag.get("source_url"),
                            "confidence": tag.get("confidence"),
                            "evidence": tag.get("evidence"),
                        }
                    )
    return by_id


def descriptor_rows_from_character(source: dict, safe_tags: dict[int, list[dict]]) -> list[dict]:
    character_id = int(source["anilist_character_id"])
    rows: list[dict] = []
    for tag in ((source.get("llm_tags") or {}).get("personality") or []):
        descriptor = normalized_tag(tag.get("tag") or "")
        if descriptor:
            rows.append(
                {
                    "tag": descriptor,
                    "category": "personality",
                    "source": ",".join(tag.get("sources") or ["merged"]),
                    "confidence": tag.get("confidence"),
                    "evidence": "; ".join(tag.get("evidence") or []),
                }
            )
    rows.extend(safe_tags.get(character_id, []))
    deduped = {}
    for row in rows:
        deduped.setdefault(row["tag"], row)
    return list(deduped.values())


def find_profile(profiles: list[dict], query: str) -> dict:
    query_keys = name_keys(query)
    for profile in profiles:
        if query_keys & alias_keys(profile):
            return profile
    raise RuntimeError(f"No profile matched query: {query}")


def load_role_character_display(role_edges_path: Path) -> dict[int, dict]:
    payload = read_json(role_edges_path)
    by_id: dict[int, dict] = {}
    for role in payload.get("roles", []):
        character = role.get("character") or {}
        character_id = character.get("character_id")
        if character_id is not None and int(character_id) not in by_id:
            by_id[int(character_id)] = character
    return by_id


def enrich_character(source: dict, display_by_id: dict[int, dict], role_display_by_id: dict[int, dict], bangumi: dict) -> dict:
    character_id = int(source["anilist_character_id"])
    display = display_by_id.get(character_id, {})
    role_display = role_display_by_id.get(character_id, {})
    anilist_favourites = int(display.get("favourites") or role_display.get("favourites") or source.get("favourites") or 0)
    bangumi_collects = int(bangumi.get("bangumi_collects") or 0)
    return {
        "character_id": character_id,
        "name": display.get("name") or role_display.get("name") or source.get("name") or "",
        "native_name": source.get("native_name") or "",
        "anime": display.get("anime_title") or role_display.get("first_anime") or source.get("first_anime") or "",
        "image": display.get("image") or role_display.get("image") or source.get("image") or "",
        "site_url": display.get("site_url") or role_display.get("site_url") or source.get("site_url") or "",
        "favourites": anilist_favourites,
        "anilist_favourites": anilist_favourites,
        "bangumi_collects": bangumi_collects,
        "combined_favourites": anilist_favourites + bangumi_collects,
        "bangumi_character_id": bangumi.get("bangumi_character_id"),
        "bangumi_url": bangumi.get("bangumi_url") or "",
        "role_edge_count": max(len(source.get("seiyuu") or []), 1),
    }


def load_or_create_embeddings(descriptors: list[str], output_dir: Path, model_name: str) -> np.ndarray:
    digest = hashlib.sha256("\n".join(descriptors).encode("utf-8")).hexdigest()[:16]
    safe_model = slug(model_name)
    npz_path = output_dir / f"descriptor_embeddings_{safe_model}_{digest}.npz"
    json_path = output_dir / f"descriptor_embeddings_{safe_model}_{digest}.json"
    if npz_path.exists() and json_path.exists():
        metadata = read_json(json_path)
        if metadata.get("model") == model_name and metadata.get("descriptors") == descriptors:
            return np.load(npz_path)["embeddings"].astype(np.float64)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(descriptors, batch_size=64, normalize_embeddings=True, show_progress_bar=False).astype(np.float64)
    embeddings = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1.0e-12)
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


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1.0e-12)


def normalize_columns(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=0, keepdims=True)
    return matrix / np.maximum(norms, 1.0e-12)


def mean_lane_overlap(H: np.ndarray) -> float:
    if H.shape[0] < 2:
        return 0.0
    normalized = normalize_rows(H)
    overlap = normalized @ normalized.T
    mask = ~np.eye(overlap.shape[0], dtype=bool)
    return float(overlap[mask].mean())


def fit_nmf_scan(G: np.ndarray, min_k: int, max_k: int, random_state: int, max_iter: int) -> tuple[dict, list[dict]]:
    scans = []
    best: dict | None = None
    upper = max(min(max_k, G.shape[0] - 1), min_k)
    for k in range(min_k, upper + 1):
        model = NMF(
            n_components=k,
            init="nndsvda",
            solver="cd",
            beta_loss="frobenius",
            random_state=random_state,
            max_iter=max_iter,
        )
        W = model.fit_transform(G)
        H = model.components_
        reconstruction = W @ H
        overlap = mean_lane_overlap(H)
        rel_error = float(np.linalg.norm(G - reconstruction) / max(np.linalg.norm(G), 1.0e-12))
        scan = {
            "k": k,
            "mean_lane_overlap": round(overlap, 8),
            "relative_reconstruction_error": round(rel_error, 8),
            "n_iter": int(model.n_iter_),
            "_W": W,
            "_H": H,
        }
        scans.append(scan)
        if best is None or (overlap, rel_error, k) < (
            best["mean_lane_overlap"],
            best["relative_reconstruction_error"],
            best["k"],
        ):
            best = scan
    assert best is not None
    public_scans = [{key: value for key, value in scan.items() if not key.startswith("_")} for scan in scans]
    return best, public_scans


def lane_label_candidates(descriptors: list[str], embeddings: np.ndarray, lane_weights: np.ndarray) -> list[dict]:
    lane_embedding = lane_weights @ embeddings
    lane_embedding = lane_embedding / max(float(np.linalg.norm(lane_embedding)), 1.0e-12)
    candidates = []
    for index, descriptor in enumerate(descriptors):
        if not single_adjective_shape(descriptor):
            continue
        score = float(embeddings[index] @ lane_embedding)
        candidates.append(
            {
                "descriptor_index": index,
                "label": descriptor,
                "score": score,
                "weight": float(lane_weights[index]),
            }
        )
    candidates.sort(key=lambda row: (row["weight"], row["score"]), reverse=True)
    return candidates


def choose_lane_labels(H: np.ndarray, descriptors: list[str], embeddings: np.ndarray, top_terms: int) -> tuple[list[dict], list[int]]:
    used: set[str] = set()
    label_indices: list[int] = []
    lanes = []
    strengths = H.sum(axis=1)
    total_strength = max(float(strengths.sum()), 1.0e-12)
    for lane_index in np.argsort(strengths)[::-1]:
        weights = H[int(lane_index)]
        descriptor_total = max(float(weights.sum()), 1.0e-12)
        candidates = lane_label_candidates(descriptors, embeddings, weights)
        label = next((candidate for candidate in candidates if candidate["label"] not in used), candidates[0] if candidates else None)
        if label:
            used.add(label["label"])
            label_indices.append(int(label["descriptor_index"]))
        order = np.argsort(weights)[::-1]
        lanes.append(
            {
                "lane": int(lane_index),
                "label": label["label"] if label else "",
                "label_score": round(float(label["score"]), 8) if label else 0.0,
                "strength_share": round(float(strengths[int(lane_index)] / total_strength), 8),
                "top_descriptors": [
                    {
                        "descriptor": descriptors[int(index)],
                        "weight": round(float(weights[int(index)]), 8),
                        "share": round(float(weights[int(index)] / descriptor_total), 8),
                    }
                    for index in order[:top_terms]
                    if float(weights[int(index)]) > 0.0
                ],
                "label_candidates": [
                    {
                        "label": row["label"],
                        "weight": round(float(row["weight"]), 8),
                        "score": round(float(row["score"]), 8),
                    }
                    for row in candidates[:10]
                ],
            }
        )
    return lanes, label_indices


def orthogonal_descriptor_atoms(raw_gram: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh((raw_gram + raw_gram.T) * 0.5)
    keep = eigenvalues > raw_gram.shape[0] * np.finfo(np.float64).eps * max(float(eigenvalues.max()), 1.0) * 100.0
    atoms = eigenvectors[:, keep] * np.sqrt(eigenvalues[keep])
    return atoms, eigenvalues[keep]


def nnls_pg(atoms: np.ndarray, target: np.ndarray, max_iter: int = 1000) -> np.ndarray:
    if atoms.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    gram = atoms @ atoms.T
    rhs = atoms @ target
    vector = np.ones(atoms.shape[0], dtype=np.float64)
    vector /= max(float(np.linalg.norm(vector)), 1.0e-12)
    for _ in range(40):
        vector = gram @ vector
        vector /= max(float(np.linalg.norm(vector)), 1.0e-12)
    lipschitz = max(float(vector @ (gram @ vector)), 1.0e-12)
    coef = np.zeros(atoms.shape[0], dtype=np.float64)
    y = coef.copy()
    t = 1.0
    for _ in range(max_iter):
        next_coef = np.maximum(0.0, y - (gram @ y - rhs) / lipschitz)
        next_t = (1.0 + math.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = next_coef + ((t - 1.0) / next_t) * (next_coef - coef)
        if np.linalg.norm(next_coef - coef) < 1.0e-10 * max(1.0, np.linalg.norm(coef)):
            coef = next_coef
            break
        coef = next_coef
        t = next_t
    return coef


def fit_axis_to_lane_labels(target: np.ndarray, label_indices: list[int], descriptor_atoms: np.ndarray, descriptors: list[str]) -> dict:
    if not label_indices:
        return {"fit_percent": 0.0, "descriptors": []}
    target = target / max(float(np.linalg.norm(target)), 1.0e-12)
    atoms = descriptor_atoms[label_indices]
    atoms = normalize_rows(atoms)
    coefficients = nnls_pg(atoms, target)
    approximation = coefficients @ atoms
    fit = float(approximation @ target / max(float(np.linalg.norm(approximation)), 1.0e-12))
    total = max(float(coefficients.sum()), 1.0e-12)
    order = np.argsort(coefficients)[::-1]
    return {
        "fit_percent": round(fit * 100.0, 6),
        "descriptors": [
            {
                "descriptor": descriptors[label_indices[int(index)]],
                "weight": round(float(coefficients[int(index)]), 10),
                "share": round(float(coefficients[int(index)] / total), 8),
            }
            for index in order
            if float(coefficients[int(index)]) > 1.0e-8
        ],
    }


def character_rows(vector: np.ndarray, characters: list[dict], sign: int | None = None, limit: int = 12) -> list[dict]:
    rows = []
    for index in np.argsort(vector * vector)[::-1]:
        value = float(vector[int(index)])
        if sign == 1 and value <= 0.0:
            continue
        if sign == -1 and value >= 0.0:
            continue
        character = characters[int(index)]
        rows.append(
            {
                "character_id": character["character_id"],
                "name": character["name"],
                "anime": character["anime"],
                "image": character.get("image") or "",
                "site_url": character.get("site_url") or "",
                "amplitude": round(value, 10),
                "combined_favourites": int(character.get("combined_favourites") or 0),
                "descriptors": character.get("descriptors") or [],
            }
        )
        if len(rows) >= limit:
            break
    return rows


def orient_axis(u: np.ndarray, v: np.ndarray, characters: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    scores = np.abs(u) * np.asarray([math.log1p(max(float(row.get("combined_favourites") or 0), 0.0)) for row in characters])
    if float(scores.max(initial=0.0)) <= 0.0:
        scores = np.abs(u)
    pivot = int(np.argmax(scores))
    if u[pivot] < 0.0:
        return -u, -v
    return u, v


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    site_profiles = read_json(args.site_profile_input)["profiles"]
    tag_payload = read_json(args.tags_input)
    profile = find_profile(site_profiles, args.seiyuu)
    safe_tags = load_safe_llm_personality(args.safe_llm_tags)
    bangumi_collects = load_bangumi_collects(args.safe_enrichment, args.bangumi_raw_dir)
    display_payload = read_json(args.character_display_input)
    display_by_id = {int(key): value for key, value in display_payload.get("characters", {}).items()}
    role_display_by_id = load_role_character_display(args.role_edges)

    seiyuu_to_characters: dict[str, list[dict]] = defaultdict(list)
    descriptor_to_sources: dict[str, list[dict]] = defaultdict(list)
    for source in tag_payload.get("characters", []):
        if is_pop_team_character(source) or len(source.get("seiyuu") or []) > args.max_role_edge_count:
            continue
        tag_rows = descriptor_rows_from_character(source, safe_tags)
        if not tag_rows:
            continue
        character = enrich_character(
            source,
            display_by_id,
            role_display_by_id,
            bangumi_collects.get(int(source["anilist_character_id"]), {}),
        )
        descriptor_weights: dict[str, float] = {}
        for tag_row in tag_rows:
            descriptor = tag_row["tag"]
            descriptor_weights[descriptor] = max(descriptor_weights.get(descriptor, 0.0), 1.0)
            descriptor_to_sources[descriptor].append(
                {
                    "character_id": character["character_id"],
                    "character": character["name"],
                    "source": tag_row.get("source"),
                    "evidence": tag_row.get("evidence"),
                }
            )
        character["descriptor_weights"] = descriptor_weights
        character["descriptors"] = sorted(descriptor_weights)
        for seiyuu in source.get("seiyuu", []):
            for key in name_keys(seiyuu.get("name") or ""):
                seiyuu_to_characters[key].append(character)

    seen = set()
    characters = []
    for key in alias_keys(profile):
        for character in seiyuu_to_characters.get(key, []):
            if character["character_id"] in seen:
                continue
            seen.add(character["character_id"])
            characters.append(character)
    if len(characters) < 2:
        raise RuntimeError(f"Need at least two described characters for {profile['name']}; found {len(characters)}.")

    descriptors = sorted({descriptor for character in characters for descriptor in character["descriptors"]})
    descriptor_index = {descriptor: index for index, descriptor in enumerate(descriptors)}
    embeddings = load_or_create_embeddings(descriptors, args.output_dir, args.embedding_model)
    raw_gram = np.clip(embeddings @ embeddings.T, -1.0, 1.0)
    nmf_gram = np.clip(raw_gram, 0.0, 1.0)
    np.fill_diagonal(nmf_gram, 1.0)

    B = np.zeros((len(characters), len(descriptors)), dtype=np.float64)
    for row_index, character in enumerate(characters):
        for descriptor, weight in character["descriptor_weights"].items():
            B[row_index, descriptor_index[descriptor]] = float(weight)
    if args.normalize_b_rows:
        B = normalize_rows(B)

    best_scan, scans = fit_nmf_scan(nmf_gram, args.min_k, args.max_k, args.random_state, args.max_iter)
    H = best_scan["_H"]
    lanes, label_indices = choose_lane_labels(H, descriptors, embeddings, args.top_lane_descriptors)

    descriptor_atoms, kept_eigenvalues = orthogonal_descriptor_atoms(raw_gram)
    Z = B @ descriptor_atoms
    if args.normalize_z_columns:
        Z = normalize_columns(Z)
    row_weights = np.asarray(
        [
            character_row_weight(character, args.row_weight) * shared_role_weight(character, args.shared_role_weight)
            for character in characters
        ],
        dtype=np.float64,
    )
    Z_weighted = Z * row_weights.reshape(-1, 1)
    left, singular_values, vt = np.linalg.svd(Z_weighted, full_matrices=False)
    singular_mass = singular_values * singular_values
    total_mass = max(float(singular_mass.sum()), 1.0e-12)

    svd_axes = []
    for axis_index in range(min(args.top_svd, len(singular_values))):
        u, v = orient_axis(left[:, axis_index].copy(), vt[axis_index].copy(), characters)
        positive_fit = fit_axis_to_lane_labels(v, label_indices, descriptor_atoms, descriptors)
        negative_fit = fit_axis_to_lane_labels(-v, label_indices, descriptor_atoms, descriptors)
        svd_axes.append(
            {
                "rank": axis_index + 1,
                "singular_value": round(float(singular_values[axis_index]), 10),
                "mass_percent": round(float(singular_mass[axis_index] / total_mass * 100.0), 6),
                "positive_fit": positive_fit,
                "negative_fit": negative_fit,
                "positive_characters": character_rows(u, characters, sign=1, limit=args.top_characters),
                "negative_characters": character_rows(u, characters, sign=-1, limit=args.top_characters),
            }
        )

    # Attach representative characters to NMF lanes after lane ordering.
    for lane in lanes:
        lane_index = lane["lane"]
        loading = B @ H[lane_index]
        lane["top_characters"] = [
            {
                "name": characters[int(index)]["name"],
                "anime": characters[int(index)]["anime"],
                "loading": round(float(loading[int(index)]), 8),
                "combined_favourites": int(characters[int(index)].get("combined_favourites") or 0),
            }
            for index in np.argsort(loading)[::-1][: args.top_characters]
            if float(loading[int(index)]) > 0.0
        ]

    output = {
        "generated_at": utc_now(),
        "source": "seiyuu_local_nmf_lane_svd.py",
        "parameters": {
            "seiyuu_query": args.seiyuu,
            "profile_name": profile["name"],
            "descriptor_categories": ["personality"],
            "safe_llm_tags": [str(path) for path in args.safe_llm_tags],
            "nmf_matrix": "G_seiyuu = max(E @ E.T, 0), uncentered descriptor cosine overlap",
            "svd_matrix": "Z = weighted_rows(column_normalize(B @ X)), where X are symmetric orthogonalized descriptor atoms from raw E @ E.T",
            "positive_descriptor_fit_space": "V = one unique single-adjective label per local NMF lane",
            "row_weight": args.row_weight,
            "shared_role_weight": args.shared_role_weight,
            "normalize_b_rows": args.normalize_b_rows,
            "normalize_z_columns": args.normalize_z_columns,
            "random_state": args.random_state,
        },
        "counts": {
            "characters": len(characters),
            "descriptors": len(descriptors),
            "single_adjective_lane_labels": len(label_indices),
            "orthogonal_rank": int(len(kept_eigenvalues)),
        },
        "nmf_scan": scans,
        "chosen_nmf": {key: value for key, value in best_scan.items() if not key.startswith("_")},
        "lane_label_space": [descriptors[index] for index in label_indices],
        "lanes": lanes,
        "singular_spectrum": [
            {
                "rank": index + 1,
                "singular_value": round(float(value), 10),
                "mass_percent": round(float(singular_mass[index] / total_mass * 100.0), 6),
            }
            for index, value in enumerate(singular_values[: min(12, len(singular_values))])
        ],
        "svd_axes": svd_axes,
        "characters": [
            {
                "character_id": character["character_id"],
                "name": character["name"],
                "anime": character["anime"],
                "combined_favourites": int(character.get("combined_favourites") or 0),
                "row_weight": round(float(row_weights[index]), 8),
                "descriptors": character["descriptors"],
            }
            for index, character in enumerate(characters)
        ],
    }
    json_path = args.output_dir / f"{slug(profile['name'])}_local_nmf_lane_svd.json"
    write_json(json_path, output)
    print(f"wrote {json_path}")
    print(json.dumps({"counts": output["counts"], "chosen_nmf": output["chosen_nmf"]}, indent=2))
    print("lane labels:", ", ".join(output["lane_label_space"]))
    for axis in output["svd_axes"][:2]:
        print(f"SV{axis['rank']} +", [row["descriptor"] for row in axis["positive_fit"]["descriptors"][:6]])
        print(f"SV{axis['rank']} -", [row["descriptor"] for row in axis["negative_fit"]["descriptors"][:6]])
        print("  +chars", [row["name"] for row in axis["positive_characters"][:6]])
        print("  -chars", [row["name"] for row in axis["negative_characters"][:6]])


if __name__ == "__main__":
    main()
