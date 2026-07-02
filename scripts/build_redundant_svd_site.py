#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the redundant-basis SVD seiyuu profiler site payload.")
    parser.add_argument("--site-profile-input", type=Path, default=Path("site/profiles.json"))
    parser.add_argument("--character-display-input", type=Path, default=Path("site/character_display.json"))
    parser.add_argument("--role-edges", type=Path, default=Path("data/role_edges.json"))
    parser.add_argument(
        "--tags-input",
        type=Path,
        default=Path("data/external/merged/all_characters_llm_vndb_personality_tags.json"),
    )
    parser.add_argument(
        "--descriptor-union",
        type=Path,
        default=Path("run/adjectival_personality_union/adjectival_personality_union.json"),
    )
    parser.add_argument(
        "--descriptor-assignments",
        type=Path,
        default=Path("run/adjectival_personality_union/adjectival_personality_assignments.jsonl"),
    )
    parser.add_argument(
        "--embedding-npz",
        type=Path,
        default=Path("models/adjectival_personality_nmf/adjectival_personality_embeddings_baai_bge-small-en-v1.5.npz"),
    )
    parser.add_argument("--output", type=Path, default=Path("site/profiles.json"))
    parser.add_argument("--profile-dir", type=Path, default=Path("site/profile_payloads"))
    parser.add_argument("--max-profiles", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=18)
    parser.add_argument("--sv-relative-cutoff", type=float, default=0.85)
    parser.add_argument("--sv1-fit-target", type=float, default=0.95)
    parser.add_argument("--sv1-min-terms", type=int, default=10)
    parser.add_argument("--sv1-max-terms", type=int, default=10)
    parser.add_argument("--variation-gain-cutoff", type=float, default=0.01)
    parser.add_argument("--max-variation-terms", type=int, default=50)
    parser.add_argument("--max-role-edge-count", type=int, default=20)
    parser.add_argument("--shared-role-weight", choices=["none", "inverse_sqrt", "inverse"], default="inverse_sqrt")
    parser.add_argument(
        "--safe-enrichment",
        type=Path,
        default=Path("data/external/safe_enrichment/character_safe_enrichment.jsonl"),
        help="Safe enrichment cache used to join selected Bangumi character ids.",
    )
    parser.add_argument(
        "--bangumi-raw-dir",
        type=Path,
        default=Path("data/external/safe_enrichment/raw/bangumi"),
        help="Raw Bangumi search cache containing stat.collects for selected matches.",
    )
    parser.add_argument(
        "--row-weight",
        choices=["none", "sqrt_log_combined_favourites", "log_combined_favourites", "sqrt_combined_favourites"],
        default="sqrt_log_combined_favourites",
        help="Optional character popularity weighting applied to rows before per-seiyuu SVD.",
    )
    parser.add_argument(
        "--normalize-character-rows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize each character descriptor vector before applying row weights.",
    )
    parser.add_argument(
        "--descriptor-shape",
        choices=["all", "single_word_or_hyphenated"],
        default="single_word_or_hyphenated",
        help="Restrict the redundant descriptor basis to compact adjective atoms.",
    )
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


def norm_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (value or "").lower()).strip("_") or "value"


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


def descriptor_shape_ok(descriptor: str, mode: str) -> bool:
    if mode == "all":
        return True
    if mode == "single_word_or_hyphenated":
        return re.fullmatch(r"[a-z]+(?:-[a-z]+)*", descriptor.strip()) is not None
    raise ValueError(f"unknown descriptor shape mode: {mode}")


def descriptor_assignment_targets(descriptor: str, descriptor_index: dict[str, int]) -> list[str]:
    descriptor = (descriptor or "").strip()
    if descriptor in descriptor_index:
        return [descriptor]
    atoms = []
    for token in re.findall(r"[a-z]+(?:-[a-z]+)*", descriptor):
        if token in descriptor_index and token not in atoms:
            atoms.append(token)
    return atoms


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


def load_bangumi_collects(safe_enrichment_path: Path, raw_dir: Path) -> dict[int, dict]:
    selected_bangumi_by_anilist: dict[int, int] = {}
    if safe_enrichment_path.exists():
        with safe_enrichment_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                matches = (row.get("bangumi") or {}).get("matches") or []
                if not matches:
                    continue
                selected_bangumi_by_anilist[int(row["anilist_character_id"])] = int(matches[0]["bangumi_character_id"])

    raw_by_bangumi: dict[int, dict] = {}
    if raw_dir.exists():
        for path in raw_dir.glob("*.json"):
            try:
                payload = read_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            for item in ((payload.get("response") or {}).get("data") or []):
                bangumi_id = item.get("id")
                if bangumi_id is not None:
                    raw_by_bangumi[int(bangumi_id)] = item

    rows: dict[int, dict] = {}
    for anilist_id, bangumi_id in selected_bangumi_by_anilist.items():
        raw = raw_by_bangumi.get(bangumi_id) or {}
        stat = raw.get("stat") or {}
        rows[anilist_id] = {
            "bangumi_character_id": bangumi_id,
            "bangumi_url": f"https://bgm.tv/character/{bangumi_id}",
            "bangumi_collects": int(stat.get("collects") or 0),
            "bangumi_comments": int(stat.get("comments") or 0),
        }
    return rows


def enrich_character(character: dict, display_by_id: dict[int, dict], role_character_by_id: dict[int, dict]) -> dict:
    display = display_by_id.get(int(character["character_id"]), {})
    role_character = role_character_by_id.get(int(character["character_id"]), {})
    anilist_favourites = int(display.get("favourites") or role_character.get("favourites") or character.get("favourites") or 0)
    bangumi_collects = int(character.get("bangumi_collects") or 0)
    return {
        "character_id": int(character["character_id"]),
        "name": display.get("name") or role_character.get("name") or character.get("name") or "",
        "anime": display.get("anime_title") or role_character.get("first_anime") or character.get("first_anime") or "",
        "image": display.get("image") or role_character.get("image") or character.get("image") or "",
        "site_url": display.get("site_url") or role_character.get("site_url") or character.get("site_url") or "",
        "favourites": anilist_favourites,
        "anilist_favourites": anilist_favourites,
        "bangumi_collects": bangumi_collects,
        "bangumi_comments": int(character.get("bangumi_comments") or 0),
        "bangumi_character_id": character.get("bangumi_character_id"),
        "bangumi_url": character.get("bangumi_url") or "",
        "combined_favourites": anilist_favourites + bangumi_collects,
        "role_edge_count": int(character.get("role_edge_count") or 1),
    }


def character_rows(values: np.ndarray, labels: list[dict], sign: int | None = None) -> list[dict]:
    norm2 = float(np.sum(values * values))
    rows = []
    for index in np.argsort(values * values)[::-1]:
        amplitude = float(values[int(index)])
        if sign == 1 and amplitude <= 0.0:
            continue
        if sign == -1 and amplitude >= 0.0:
            continue
        percent = (amplitude * amplitude / norm2 * 100.0) if norm2 > 0 else 0.0
        rows.append(
            {
                **labels[int(index)],
                "amplitude": round(amplitude, 10),
                "abs_amplitude": round(abs(amplitude), 10),
                "percent": round(percent, 6),
            }
        )
    return rows


def participation(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    norm2 = float(np.sum(values * values))
    if norm2 <= 0:
        return {"effective_count": 0.0, "top5_mass_percent": 0.0, "top10_mass_percent": 0.0}
    p = (values * values) / norm2
    return {
        "effective_count": round(float(1.0 / max(np.sum(p * p), 1.0e-12)), 6),
        "top5_mass_percent": round(float(np.sum(np.sort(p)[::-1][:5]) * 100.0), 6),
        "top10_mass_percent": round(float(np.sum(np.sort(p)[::-1][:10]) * 100.0), 6),
    }


def nnls_pg(atoms: np.ndarray, target: np.ndarray, max_iter: int = 1000) -> np.ndarray:
    count = atoms.shape[0]
    if count == 0:
        return np.zeros(0, dtype=np.float64)
    gram = atoms @ atoms.T
    rhs = atoms @ target
    if count == 1:
        lipschitz = max(float(gram[0, 0]), 1.0e-12)
    else:
        vector = np.ones(count, dtype=np.float64)
        vector /= np.linalg.norm(vector)
        for _ in range(40):
            vector = gram @ vector
            vector /= max(float(np.linalg.norm(vector)), 1.0e-12)
        lipschitz = max(float(vector @ (gram @ vector)), 1.0e-12)

    coefficients = np.zeros(count, dtype=np.float64)
    y = coefficients.copy()
    t = 1.0
    for _ in range(max_iter):
        next_coefficients = np.maximum(0.0, y - (gram @ y - rhs) / lipschitz)
        next_t = (1.0 + math.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = next_coefficients + ((t - 1.0) / next_t) * (next_coefficients - coefficients)
        if np.linalg.norm(next_coefficients - coefficients) < 1.0e-10 * max(1.0, np.linalg.norm(coefficients)):
            coefficients = next_coefficients
            break
        coefficients = next_coefficients
        t = next_t
    return coefficients


def decode_axis(
    target: np.ndarray,
    descriptor_atoms: np.ndarray,
    descriptors: list[str],
    *,
    stop_fit: float | None,
    min_terms: int = 1,
    stop_gain: float,
    max_terms: int,
) -> dict:
    target = target / max(float(np.linalg.norm(target)), 1.0e-12)
    active: list[int] = []
    selected: set[int] = set()
    residual = target.copy()
    coefficients = np.zeros(0, dtype=np.float64)
    previous_sse = float(residual @ residual)
    steps = []

    for step in range(1, max_terms + 1):
        correlations = descriptor_atoms @ residual
        if selected:
            correlations[list(selected)] = -np.inf
        descriptor_index = int(np.argmax(correlations))
        if not np.isfinite(correlations[descriptor_index]) or correlations[descriptor_index] <= 0.0:
            steps.append({"step": step, "stop": "no_positive_residual_descriptor"})
            break

        active.append(descriptor_index)
        selected.add(descriptor_index)
        coefficients = nnls_pg(descriptor_atoms[active], target)
        approximation = coefficients @ descriptor_atoms[active]
        residual = target - approximation
        sse = float(residual @ residual)
        fit = float(approximation @ target / max(float(np.linalg.norm(approximation)), 1.0e-12))
        gain = (previous_sse - sse) / previous_sse if previous_sse > 0.0 else 0.0
        steps.append(
            {
                "step": step,
                "descriptor": descriptors[descriptor_index],
                "fit_percent": round(fit * 100.0, 6),
                "residual_gain_percent": round(gain * 100.0, 6),
            }
        )
        previous_sse = sse

        if stop_fit is not None and fit >= stop_fit and step >= min_terms:
            break
        if stop_fit is None and step > 1 and gain < stop_gain and step >= min_terms:
            break

    if not active:
        return {"fit_percent": 0.0, "descriptors": [], "steps": steps}

    approximation = coefficients @ descriptor_atoms[active]
    fit = float(approximation @ target / max(float(np.linalg.norm(approximation)), 1.0e-12))
    total = max(float(np.sum(coefficients)), 1.0e-12)
    rows = []
    for rank, coefficient_index in enumerate(np.argsort(coefficients)[::-1], start=1):
        coefficient = float(coefficients[int(coefficient_index)])
        if coefficient <= 1.0e-8:
            continue
        rows.append(
            {
                "rank": rank,
                "descriptor": descriptors[active[int(coefficient_index)]],
                "amplitude": round(coefficient, 10),
                "abs_amplitude": round(abs(coefficient), 10),
                "percent": round(coefficient / total * 100.0, 6),
            }
        )
    return {"fit_percent": round(fit * 100.0, 6), "descriptors": rows, "steps": steps}


def profile_matrix(
    characters: list[dict],
    descriptor_index: dict[str, int],
    descriptor_atoms: np.ndarray,
    shared_role_weight_mode: str,
    row_weight_mode: str,
    normalize_character_rows: bool,
) -> tuple[np.ndarray, list[dict]]:
    rows = []
    labels = []
    for character in characters:
        indices = [descriptor_index[tag] for tag in character.get("descriptors", []) if tag in descriptor_index]
        if not indices:
            continue
        row = np.sum(descriptor_atoms[indices], axis=0)
        unweighted_norm = float(np.linalg.norm(row))
        if normalize_character_rows:
            row = row / max(unweighted_norm, 1.0e-12)
        shared_weight = shared_role_weight(character, shared_role_weight_mode)
        popularity_weight = character_row_weight(character, row_weight_mode)
        weight = shared_weight * popularity_weight
        rows.append(row * weight)
        labels.append(
            {
                **character,
                "shared_role_weight": round(shared_weight, 8),
                "popularity_row_weight": round(popularity_weight, 8),
                "row_weight": round(weight, 8),
                "unweighted_row_norm": round(unweighted_norm, 8),
            }
        )
    if not rows:
        return np.zeros((0, descriptor_atoms.shape[1]), dtype=np.float64), []
    return np.vstack(rows), labels


def build_axis_payload(
    rank: int,
    singular_value: float,
    mass_percent: float,
    left_vector: np.ndarray,
    right_vector: np.ndarray,
    character_labels: list[dict],
    descriptor_atoms: np.ndarray,
    descriptors: list[str],
    *,
    relative_to_sv2_percent: float | None,
    broad: bool,
    sv1_fit_target: float,
    sv1_min_terms: int,
    sv1_max_terms: int,
    variation_gain_cutoff: float,
    max_variation_terms: int,
) -> dict:
    if broad:
        decoded = decode_axis(
            right_vector,
            descriptor_atoms,
            descriptors,
            stop_fit=sv1_fit_target,
            min_terms=sv1_min_terms,
            stop_gain=variation_gain_cutoff,
            max_terms=sv1_max_terms,
        )
        return {
            "rank": rank,
            "kind": "broad_cluster",
            "singular_value": round(float(singular_value), 10),
            "mass_percent": round(float(mass_percent), 6),
            "descriptor_fit_percent": decoded["fit_percent"],
            "display_descriptor_components": decoded["descriptors"],
            "descriptor_steps": decoded["steps"],
            "characters": character_rows(left_vector, character_labels),
            "positive_characters": character_rows(left_vector, character_labels, 1),
            "negative_characters": character_rows(left_vector, character_labels, -1),
            "character_participation": participation(left_vector),
        }

    positive_decode = decode_axis(
        right_vector,
        descriptor_atoms,
        descriptors,
        stop_fit=None,
        min_terms=1,
        stop_gain=variation_gain_cutoff,
        max_terms=max_variation_terms,
    )
    negative_decode = decode_axis(
        -right_vector,
        descriptor_atoms,
        descriptors,
        stop_fit=None,
        min_terms=1,
        stop_gain=variation_gain_cutoff,
        max_terms=max_variation_terms,
    )
    return {
        "rank": rank,
        "kind": "variation_axis",
        "singular_value": round(float(singular_value), 10),
        "mass_percent": round(float(mass_percent), 6),
        "relative_to_sv2_percent": round(float(relative_to_sv2_percent or 0.0), 6),
        "positive_pole": {
            "descriptor_fit_percent": positive_decode["fit_percent"],
            "display_descriptor_components": positive_decode["descriptors"],
            "descriptor_steps": positive_decode["steps"],
            "characters": character_rows(left_vector, character_labels, 1),
        },
        "negative_pole": {
            "descriptor_fit_percent": negative_decode["fit_percent"],
            "display_descriptor_components": negative_decode["descriptors"],
            "descriptor_steps": negative_decode["steps"],
            "characters": character_rows(left_vector, character_labels, -1),
        },
    }


def orient_sv1(left_vector: np.ndarray, right_vector: np.ndarray, character_labels: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    favourite_weights = np.asarray(
        [math.log1p(max(float(character.get("favourites") or 0.0), 0.0)) for character in character_labels],
        dtype=np.float64,
    )
    scores = np.abs(left_vector) * favourite_weights
    if float(np.max(scores)) <= 0.0:
        scores = np.abs(left_vector)
    pivot = int(np.argmax(scores))
    if left_vector[pivot] < 0.0:
        return -left_vector, -right_vector
    return left_vector, right_vector


def orient_variation(
    left_vector: np.ndarray,
    right_vector: np.ndarray,
    character_labels: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.abs(left_vector) * np.asarray(
        [math.log1p(max(float(character.get("favourites") or 0.0), 0.0)) for character in character_labels],
        dtype=np.float64,
    )
    if float(np.max(scores)) <= 0.0:
        scores = np.abs(left_vector)
    pivot = int(np.argmax(scores))
    if left_vector[pivot] < 0.0:
        return -left_vector, -right_vector
    return left_vector, right_vector


def main() -> None:
    args = parse_args()
    site_profiles = read_json(args.site_profile_input)
    display_payload = read_json(args.character_display_input)
    display_by_id = {int(key): value for key, value in display_payload.get("characters", {}).items()}
    bangumi_collects_by_id = load_bangumi_collects(args.safe_enrichment, args.bangumi_raw_dir)
    role_payload = read_json(args.role_edges)
    role_character_by_id = {}
    for role in role_payload.get("roles", []):
        character = role.get("character") or {}
        character_id = character.get("character_id")
        if character_id is not None and int(character_id) not in role_character_by_id:
            role_character_by_id[int(character_id)] = character
    tag_payload = read_json(args.tags_input)
    union_payload = read_json(args.descriptor_union)
    descriptor_rows = union_payload["descriptors"]
    descriptor_mask = [
        descriptor_shape_ok(row["tag"], args.descriptor_shape)
        for row in descriptor_rows
    ]
    descriptors = [row["tag"] for row, keep_row in zip(descriptor_rows, descriptor_mask, strict=True) if keep_row]
    descriptor_index = {descriptor: index for index, descriptor in enumerate(descriptors)}

    embeddings = np.load(args.embedding_npz)["embeddings"].astype(np.float64)
    embeddings = embeddings[np.asarray(descriptor_mask, dtype=bool)]
    embeddings = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1.0e-12)
    gram = embeddings @ embeddings.T
    eigenvalues, eigenvectors = np.linalg.eigh((gram + gram.T) * 0.5)
    keep = eigenvalues > gram.shape[0] * np.finfo(np.float64).eps * max(float(eigenvalues.max()), 1.0) * 100.0
    descriptor_atoms = eigenvectors[:, keep] * np.sqrt(eigenvalues[keep])
    descriptor_atoms = descriptor_atoms / np.maximum(np.linalg.norm(descriptor_atoms, axis=1, keepdims=True), 1.0e-12)

    character_descriptors: dict[int, set[str]] = defaultdict(set)
    assignment_count = 0
    with args.descriptor_assignments.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            descriptor = row.get("tag") or ""
            targets = descriptor_assignment_targets(descriptor, descriptor_index)
            if targets:
                character_descriptors[int(row["anilist_character_id"])].update(targets)
                assignment_count += 1
    merged_personality_assignment_count = 0
    for source in tag_payload.get("characters", []):
        character_id = int(source["anilist_character_id"])
        for tag_row in ((source.get("llm_tags") or {}).get("personality") or []):
            targets = descriptor_assignment_targets(tag_row.get("tag") or "", descriptor_index)
            if targets:
                character_descriptors[character_id].update(targets)
                merged_personality_assignment_count += 1

    character_by_id: dict[int, dict] = {}
    seiyuu_to_characters: dict[str, list[dict]] = defaultdict(list)
    excluded_characters = []
    for source in tag_payload.get("characters", []):
        character_id = int(source["anilist_character_id"])
        descriptors_for_character = sorted(character_descriptors.get(character_id, set()))
        if not descriptors_for_character:
            continue
        role_edge_count = len(source.get("seiyuu") or [])
        if role_edge_count > args.max_role_edge_count or is_pop_team_character(source):
            excluded_characters.append(
                {
                    "character_id": character_id,
                    "name": source.get("name") or "",
                    "first_anime": source.get("first_anime") or "",
                    "role_edge_count": role_edge_count,
                    "reason": "pop_team_epic" if is_pop_team_character(source) else "role_edge_count",
                }
            )
            continue
        character = enrich_character(
            {
                "character_id": character_id,
                "name": source.get("name") or "",
                "first_anime": source.get("first_anime") or "",
                "favourites": int(source.get("favourites") or 0),
                "site_url": source.get("site_url") or "",
                "role_edge_count": max(role_edge_count, 1),
                "descriptors": descriptors_for_character,
                **bangumi_collects_by_id.get(character_id, {}),
            },
            display_by_id,
            role_character_by_id,
        )
        character["descriptors"] = descriptors_for_character
        character_by_id[character_id] = character
        for seiyuu in source.get("seiyuu", []):
            for key in name_keys(seiyuu.get("name") or ""):
                seiyuu_to_characters[key].append(character)

    profiles = []
    profile_indices = site_profiles.get("profiles", [])
    if args.max_profiles:
        profile_indices = profile_indices[: args.max_profiles]

    args.profile_dir.mkdir(parents=True, exist_ok=True)
    for profile_index, profile in enumerate(profile_indices, start=1):
        seen = set()
        characters = []
        for key in alias_keys(profile):
            for character in seiyuu_to_characters.get(key, []):
                character_id = int(character["character_id"])
                if character_id in seen:
                    continue
                seen.add(character_id)
                characters.append(character)

        matrix, character_labels = profile_matrix(
            characters,
            descriptor_index,
            descriptor_atoms,
            args.shared_role_weight,
            args.row_weight,
            args.normalize_character_rows,
        )
        if matrix.shape[0] == 0:
            continue
        left, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
        singular_mass = singular_values * singular_values
        total_mass = float(np.sum(singular_mass))
        if total_mass <= 0.0:
            continue

        u1, v1 = orient_sv1(left[:, 0].copy(), vt[0].copy(), character_labels)
        major_lane = build_axis_payload(
            1,
            float(singular_values[0]),
            float(singular_mass[0] / total_mass * 100.0),
            u1,
            v1,
            character_labels,
            descriptor_atoms,
            descriptors,
            relative_to_sv2_percent=None,
            broad=True,
            sv1_fit_target=args.sv1_fit_target,
            sv1_min_terms=args.sv1_min_terms,
            sv1_max_terms=args.sv1_max_terms,
            variation_gain_cutoff=args.variation_gain_cutoff,
            max_variation_terms=args.max_variation_terms,
        )

        variation_axes = []
        if len(singular_values) > 1:
            sv2 = float(singular_values[1])
            for component_index in range(1, len(singular_values)):
                if float(singular_values[component_index]) < args.sv_relative_cutoff * sv2:
                    break
                uk, vk = orient_variation(left[:, component_index].copy(), vt[component_index].copy(), character_labels)
                variation_axes.append(
                    build_axis_payload(
                        component_index + 1,
                        float(singular_values[component_index]),
                        float(singular_mass[component_index] / total_mass * 100.0),
                        uk,
                        vk,
                        character_labels,
                        descriptor_atoms,
                        descriptors,
                        relative_to_sv2_percent=float(singular_values[component_index] / sv2 * 100.0),
                        broad=False,
                        sv1_fit_target=args.sv1_fit_target,
                        sv1_min_terms=args.sv1_min_terms,
                        sv1_max_terms=args.sv1_max_terms,
                        variation_gain_cutoff=args.variation_gain_cutoff,
                        max_variation_terms=args.max_variation_terms,
                    )
                )

        spectrum = [
            {
                "rank": index + 1,
                "singular_value": round(float(value), 10),
                "mass_percent": round(float(singular_mass[index] / total_mass * 100.0), 6),
                "relative_to_sv2_percent": round(float(value / singular_values[1] * 100.0), 6)
                if index >= 1 and len(singular_values) > 1
                else None,
            }
            for index, value in enumerate(singular_values[: min(12, len(singular_values))])
        ]

        profile_path = f"profile_payloads/{slug(profile['name'])}.json"
        payload = {
            **profile,
            "profile_path": profile_path,
            "character_count": len(character_labels),
            "supported_character_count": len(character_labels),
            "major_lane": major_lane,
            "variation_axes": variation_axes,
            "singular_spectrum": spectrum,
            "model": {
                "name": "redundant_descriptor_svd",
                "description": "Uncentered B @ G @ X SVD over the full adjectival personality descriptor pool; SV1 is decoded as the broad role cluster, later singular vectors are shown as variation axes.",
                "descriptor_count": len(descriptors),
                "descriptor_shape": args.descriptor_shape,
                "orthogonal_rank": int(np.sum(keep)),
                "sv_relative_cutoff": args.sv_relative_cutoff,
                "sv1_fit_target": args.sv1_fit_target,
                "sv1_min_terms": args.sv1_min_terms,
                "sv1_max_terms": args.sv1_max_terms,
                "variation_gain_cutoff": args.variation_gain_cutoff,
                "shared_role_weight": args.shared_role_weight,
                "row_weight": args.row_weight,
                "row_weight_field": "sqrt(log1p(AniList favourites + Bangumi collects) + 1), multiplied by shared-role weight",
                "normalize_character_rows": args.normalize_character_rows,
                "bangumi_collects_source": str(args.bangumi_raw_dir),
            },
        }
        write_json(args.output.parent / profile_path, payload)

        profiles.append(
            {
                "seiyuu_id": profile.get("seiyuu_id"),
                "name": profile.get("name") or "",
                "native_name": profile.get("native_name") or "",
                "image": profile.get("image") or "",
                "site_url": profile.get("site_url") or "",
                "role_count": int(profile.get("role_count") or 0),
                "character_count": len(character_labels),
                "first_year": profile.get("first_year"),
                "aliases": sorted(alias_keys(profile)),
                "profile_path": profile_path,
                "sv1_mass_percent": major_lane["mass_percent"],
                "sv1_descriptor_fit_percent": major_lane["descriptor_fit_percent"],
                "variation_axis_count": len(variation_axes),
            }
        )
        if profile_index % 100 == 0:
            print(f"built {profile_index}/{len(profile_indices)} profiles")

    profiles.sort(
        key=lambda row: (
            int(row["character_count"] >= 10),
            float(row["sv1_descriptor_fit_percent"]),
            int(row["character_count"]),
            int(row["role_count"]),
        ),
        reverse=True,
    )

    output_payload = {
        "generated_at": utc_now(),
        "source": "build_redundant_svd_site.py",
        "parameters": {
            "descriptor_union": str(args.descriptor_union),
            "descriptor_assignments": str(args.descriptor_assignments),
            "embedding_npz": str(args.embedding_npz),
            "tags_input": str(args.tags_input),
            "role_edges": str(args.role_edges),
            "descriptor_count": len(descriptors),
            "descriptor_shape": args.descriptor_shape,
            "orthogonal_rank": int(np.sum(keep)),
            "sv_relative_cutoff": args.sv_relative_cutoff,
            "sv1_fit_target": args.sv1_fit_target,
            "sv1_min_terms": args.sv1_min_terms,
            "sv1_max_terms": args.sv1_max_terms,
            "variation_gain_cutoff": args.variation_gain_cutoff,
            "max_variation_terms": args.max_variation_terms,
            "max_role_edge_count": args.max_role_edge_count,
            "shared_role_weight": args.shared_role_weight,
            "row_weight": args.row_weight,
            "row_weight_field": "sqrt(log1p(AniList favourites + Bangumi collects) + 1), multiplied by shared-role weight",
            "normalize_character_rows": args.normalize_character_rows,
            "safe_enrichment": str(args.safe_enrichment),
            "bangumi_raw_dir": str(args.bangumi_raw_dir),
        },
        "counts": {
            "profiles": len(profiles),
            "characters_with_descriptors": len(character_by_id),
            "descriptor_assignments": assignment_count,
            "merged_personality_assignments": merged_personality_assignment_count,
            "excluded_shared_characters": len(excluded_characters),
            "characters_with_bangumi_collects": sum(
                1 for row in character_by_id.values() if int(row.get("bangumi_collects") or 0) > 0
            ),
        },
        "samples": [
            {
                "name": profile["name"],
                "native_name": profile["native_name"],
                "image": profile["image"],
                "role_count": profile["role_count"],
                "character_count": profile["character_count"],
                "sv1_mass_percent": profile["sv1_mass_percent"],
                "sv1_descriptor_fit_percent": profile["sv1_descriptor_fit_percent"],
            }
            for profile in profiles[: args.sample_count]
        ],
        "profiles": profiles,
    }
    write_json(args.output, output_payload)
    print(f"wrote {args.output} with {len(profiles)} profiles")


if __name__ == "__main__":
    main()
