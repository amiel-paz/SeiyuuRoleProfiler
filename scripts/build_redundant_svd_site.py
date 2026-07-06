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
        "--global-canonicalization-input",
        type=Path,
        default=Path("models/global_descriptor_canonicalization/descriptor_canonicalization.json"),
        help="Raw-to-canonical descriptor map. Only canonical targets that pass the current descriptor filter are kept.",
    )
    parser.add_argument(
        "--embedding-npz",
        type=Path,
        default=Path("models/adjectival_personality_nmf/adjectival_personality_embeddings_baai_bge-small-en-v1.5.npz"),
    )
    parser.add_argument(
        "--contextual-personality-scores",
        type=Path,
        default=Path("models/contextual_personality_anchor_scores/contextual_personality_anchor_scores.json"),
        help="Optional cached evidence-context anchor scores used as a second-pass personality descriptor filter.",
    )
    parser.add_argument(
        "--min-contextual-personality-score",
        type=float,
        default=0.005,
        help="Keep descriptors whose evidence-context positive-minus-negative anchor score is at least this value.",
    )
    parser.add_argument(
        "--min-descriptor-character-count",
        type=int,
        default=2,
        help="Keep descriptors only when the contextual score cache sees them on at least this many characters.",
    )
    parser.add_argument("--output", type=Path, default=Path("site/profiles.json"))
    parser.add_argument("--profile-dir", type=Path, default=Path("site/profile_payloads"))
    parser.add_argument("--max-profiles", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=18)
    parser.add_argument("--sv-relative-cutoff", type=float, default=0.85)
    parser.add_argument("--sv1-fit-target", type=float, default=0.95)
    parser.add_argument("--sv1-min-terms", type=int, default=1)
    parser.add_argument("--sv1-max-terms", type=int, default=10)
    parser.add_argument(
        "--sv1-fit-order",
        choices=["residual", "weighted_support"],
        default="weighted_support",
        help="For SV1, either greedily fit residual descriptors or add descriptors by direct weighted support first.",
    )
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
        default="none",
        help="Optional character popularity weighting applied to rows before per-seiyuu SVD.",
    )
    parser.add_argument(
        "--svd-matrix",
        choices=["z", "z_unit"],
        default="z",
        help=(
            "Matrix passed to per-seiyuu SVD. z uses the weighted character x orthogonal descriptor matrix; "
            "z_unit normalizes each character row after construction so every supported character has unit self-overlap."
        ),
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
    parser.add_argument("--positive-lane-neighbors", type=int, default=12)
    parser.add_argument("--positive-lane-candidate-limit", type=int, default=80)
    parser.add_argument("--positive-lane-similarity-floor", type=float, default=0.35)
    parser.add_argument("--positive-lane-orthogonality-penalty", type=float, default=1.0)
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


NATIONALITY_TERMS = {
    "afghan",
    "african",
    "albanian",
    "algerian",
    "american",
    "argentine",
    "argentinian",
    "armenian",
    "asian",
    "australian",
    "austrian",
    "azerbaijani",
    "bangladeshi",
    "belarusian",
    "belgian",
    "bolivian",
    "bosnian",
    "brazilian",
    "british",
    "bulgarian",
    "burmese",
    "cambodian",
    "canadian",
    "chilean",
    "chinese",
    "colombian",
    "croatian",
    "cuban",
    "czech",
    "danish",
    "dutch",
    "egyptian",
    "english",
    "estonian",
    "ethiopian",
    "european",
    "filipino",
    "finnish",
    "french",
    "georgian",
    "german",
    "ghanaian",
    "greek",
    "haitian",
    "hispanic",
    "hungarian",
    "icelandic",
    "indian",
    "indonesian",
    "iranian",
    "iraqi",
    "irish",
    "israeli",
    "italian",
    "jamaican",
    "japanese",
    "jordanian",
    "kazakh",
    "kenyan",
    "korean",
    "kuwaiti",
    "latvian",
    "lebanese",
    "libyan",
    "lithuanian",
    "malaysian",
    "mexican",
    "mongolian",
    "moroccan",
    "nepalese",
    "nigerian",
    "norwegian",
    "pakistani",
    "palestinian",
    "persian",
    "peruvian",
    "polish",
    "portuguese",
    "romanian",
    "russian",
    "scottish",
    "serbian",
    "singaporean",
    "slovak",
    "slovenian",
    "somali",
    "spanish",
    "sudanese",
    "swedish",
    "swiss",
    "syrian",
    "taiwanese",
    "thai",
    "tibetan",
    "turkish",
    "ukrainian",
    "vietnamese",
    "welsh",
}


def is_nationality_descriptor(descriptor: str) -> bool:
    tokens = re.findall(r"[a-z]+", descriptor.lower())
    return any(token in NATIONALITY_TERMS for token in tokens)


def descriptor_shape_ok(descriptor: str, mode: str) -> bool:
    if is_nationality_descriptor(descriptor):
        return False
    if mode == "all":
        return True
    if mode == "single_word_or_hyphenated":
        return re.fullmatch(r"[a-z]+(?:-[a-z]+)*", descriptor.strip()) is not None
    raise ValueError(f"unknown descriptor shape mode: {mode}")


def descriptor_context_ok(
    descriptor: str,
    contextual_scores: dict[str, dict],
    min_score: float,
    min_character_count: int,
) -> bool:
    if not contextual_scores:
        return True
    row = contextual_scores.get(descriptor)
    if row is None:
        return False
    return (
        float(row.get("personality_score_mean") or -999.0) >= min_score
        and int(row.get("character_count") or 0) >= min_character_count
    )


def remove_unhyphenated_duplicates(descriptor_rows: list[dict], base_mask: list[bool]) -> list[bool]:
    compact_hyphenated = {
        str(row["tag"]).replace("-", "")
        for row, keep in zip(descriptor_rows, base_mask, strict=True)
        if keep and "-" in str(row["tag"])
    }
    return [
        keep and not ("-" not in str(row["tag"]) and str(row["tag"]) in compact_hyphenated)
        for row, keep in zip(descriptor_rows, base_mask, strict=True)
    ]


def descriptor_assignment_targets(
    descriptor: str,
    descriptor_index: dict[str, int],
    raw_to_canonical: dict[str, str],
) -> list[str]:
    descriptor = (descriptor or "").strip()
    if descriptor in descriptor_index:
        return [descriptor]
    canonical = raw_to_canonical.get(descriptor)
    if canonical in descriptor_index:
        return [canonical]
    return []


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


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1.0e-12)


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


def descriptor_weighted_support(character_labels: list[dict]) -> dict[str, dict]:
    support: dict[str, dict] = {}
    for character in character_labels:
        weight = float(character.get("row_weight") or 1.0)
        for descriptor in character.get("descriptors") or []:
            row = support.setdefault(
                descriptor,
                {
                    "weighted_support": 0.0,
                    "support": 0,
                    "combined_favourites": 0,
                },
            )
            row["weighted_support"] += weight
            row["support"] += 1
            row["combined_favourites"] += int(character.get("combined_favourites") or character.get("favourites") or 0)
    return support


def decode_axis_by_weighted_support(
    target: np.ndarray,
    descriptor_atoms: np.ndarray,
    descriptors: list[str],
    character_labels: list[dict],
    *,
    stop_fit: float,
    min_terms: int = 1,
    max_terms: int,
) -> dict:
    target = target / max(float(np.linalg.norm(target)), 1.0e-12)
    descriptor_index = {descriptor: index for index, descriptor in enumerate(descriptors)}
    support = descriptor_weighted_support(character_labels)
    ordered = [
        descriptor
        for descriptor, row in sorted(
            support.items(),
            key=lambda item: (
                float(item[1]["weighted_support"]),
                int(item[1]["support"]),
                int(item[1]["combined_favourites"]),
                item[0],
            ),
            reverse=True,
        )
        if descriptor in descriptor_index
    ]

    active: list[int] = []
    steps = []
    coefficients = np.zeros(0, dtype=np.float64)
    fit = 0.0
    for step, descriptor in enumerate(ordered[:max_terms], start=1):
        active.append(descriptor_index[descriptor])
        coefficients = nnls_pg(descriptor_atoms[active], target)
        approximation = coefficients @ descriptor_atoms[active]
        fit = float(approximation @ target / max(float(np.linalg.norm(approximation)), 1.0e-12))
        support_row = support.get(descriptor) or {}
        steps.append(
            {
                "step": step,
                "descriptor": descriptor,
                "weighted_support": round(float(support_row.get("weighted_support") or 0.0), 8),
                "support": int(support_row.get("support") or 0),
                "fit_percent": round(fit * 100.0, 6),
                "selection": "weighted_support",
            }
        )
        if fit >= stop_fit and step >= min_terms:
            break

    if not active:
        return {"fit_percent": 0.0, "descriptors": [], "steps": steps}

    total = max(float(np.sum(coefficients)), 1.0e-12)
    rows = []
    for rank, descriptor in enumerate(ordered[: len(active)], start=1):
        index = descriptor_index[descriptor]
        coefficient_position = active.index(index)
        coefficient = float(coefficients[coefficient_position])
        support_row = support.get(descriptor) or {}
        rows.append(
            {
                "rank": rank,
                "descriptor": descriptor,
                "amplitude": round(coefficient, 10),
                "abs_amplitude": round(abs(coefficient), 10),
                "percent": round(coefficient / total * 100.0, 6) if coefficient > 1.0e-8 else 0.0,
                "weighted_support": round(float(support_row.get("weighted_support") or 0.0), 8),
                "support": int(support_row.get("support") or 0),
            }
        )
    return {"fit_percent": round(fit * 100.0, 6), "descriptors": rows, "steps": steps}


def annotate_descriptor_support(component_rows: list[dict], character_labels: list[dict]) -> list[dict]:
    output = []
    for row in component_rows:
        descriptor = row.get("descriptor")
        supporting = [
            character
            for character in character_labels
            if descriptor and descriptor in set(character.get("descriptors") or [])
        ]
        output.append(
            {
                **row,
                "support": len(supporting),
                "supporting_characters": [
                    {
                        "character_id": character["character_id"],
                        "name": character["name"],
                        "anime": character.get("anime") or "",
                        "image": character.get("image") or "",
                        "site_url": character.get("site_url") or "",
                        "combined_favourites": int(character.get("combined_favourites") or 0),
                    }
                    for character in supporting[:5]
                ],
            }
        )
    return output


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


def direct_descriptor_plane(
    character_labels: list[dict],
    descriptor_index: dict[str, int],
    descriptor_atoms: np.ndarray,
    descriptors: list[str],
    *,
    component_scope: str,
    neighbor_count: int,
    candidate_limit: int,
    similarity_floor: float,
    orthogonality_penalty: float,
) -> dict | None:
    if len(character_labels) < 3:
        return None
    descriptor_support: dict[str, int] = defaultdict(int)
    character_descriptor_sets = []
    for character in character_labels:
        descriptor_set = {descriptor for descriptor in character.get("descriptors") or [] if descriptor in descriptor_index}
        character_descriptor_sets.append(descriptor_set)
        for descriptor in descriptor_set:
            if descriptor in descriptor_index:
                descriptor_support[descriptor] += 1

    character_vectors = []
    for character in character_labels:
        indices = [descriptor_index[tag] for tag in character.get("descriptors", []) if tag in descriptor_index]
        if indices:
            character_vectors.append(np.sum(descriptor_atoms[indices], axis=0))
        else:
            character_vectors.append(np.zeros(descriptor_atoms.shape[1], dtype=np.float64))
    character_matrix = np.vstack(character_vectors)

    component_pool = [
        descriptor_index[descriptor]
        for descriptor in descriptors
        if descriptor in descriptor_index and (component_scope == "universal_515" or descriptor_support.get(descriptor, 0) > 0)
    ]
    if len(component_pool) < 2:
        return None
    component_pool_array = np.asarray(component_pool, dtype=np.int64)
    component_pool_atoms = descriptor_atoms[component_pool_array]
    seed_indices = list(range(len(descriptors))) if component_scope == "universal_515" else component_pool

    def lane_label(components: list[dict]) -> str:
        shown = [component["descriptor"] for component in components[:3]]
        if len(components) > 3:
            shown.append(f"+{len(components) - 3}")
        return " / ".join(shown)

    candidates = []
    for seed_index in seed_indices:
        seed_descriptor = descriptors[seed_index]
        seed_vector = descriptor_atoms[seed_index]
        similarities = component_pool_atoms @ seed_vector
        eligible = np.flatnonzero(similarities >= similarity_floor)
        if not np.any(component_pool_array[eligible] == seed_index):
            eligible = np.unique(np.concatenate([eligible, np.flatnonzero(component_pool_array == seed_index)]))
        if eligible.size == 0:
            continue
        ranked = sorted(
            (
                (
                    int(component_pool_array[position]),
                    float(similarities[position]),
                    int(descriptor_support.get(descriptors[int(component_pool_array[position])], 0)),
                )
                for position in eligible
            ),
            key=lambda item: (-(item[1] * item[1] * math.sqrt(max(item[2], 1))), -item[2], descriptors[item[0]]),
        )[: max(1, neighbor_count)]
        indices = np.asarray([item[0] for item in ranked], dtype=np.int64)
        raw_weights = np.asarray(
            [max(item[1], 1.0e-9) ** 2 * math.sqrt(max(item[2], 1)) for item in ranked],
            dtype=np.float64,
        )
        raw_weights = raw_weights / max(float(np.sum(raw_weights)), 1.0e-12)
        lane_vector = raw_weights @ descriptor_atoms[indices]
        lane_norm = float(np.linalg.norm(lane_vector))
        if lane_norm <= 1.0e-12:
            continue
        lane_vector = lane_vector / lane_norm
        scores_raw = character_matrix @ lane_vector
        scores_centered = scores_raw - float(np.mean(scores_raw))
        score_norm = float(np.linalg.norm(scores_centered))
        if score_norm <= 1.0e-12:
            continue
        variance = float(np.mean(scores_centered * scores_centered))
        components = [
            {
                "descriptor": descriptors[int(index)],
                "weight": round(float(weight), 8),
                "similarity_to_seed": round(float(similarity), 8),
                "support": int(support),
            }
            for index, weight, similarity, support in zip(indices, raw_weights, [item[1] for item in ranked], [item[2] for item in ranked], strict=True)
        ]
        candidates.append(
            {
                "descriptor": lane_label(components),
                "seed_descriptor": seed_descriptor,
                "support": int(sum(1 for descriptor_set in character_descriptor_sets if any(component["descriptor"] in descriptor_set for component in components))),
                "variance": variance,
                "descriptor_index": int(seed_index),
                "components": components,
                "vector": lane_vector,
                "scores_raw": scores_raw,
                "scores_centered": scores_centered,
                "score_norm": score_norm,
            }
        )

    candidates.sort(key=lambda row: (-row["variance"], -row["support"], row["descriptor"]))
    candidates = candidates[: max(2, candidate_limit)]
    if len(candidates) < 2:
        return None

    x_candidate = candidates[0]
    x_component_descriptors = {component["descriptor"] for component in x_candidate["components"]}

    def strip_candidate_components(candidate: dict, excluded_descriptors: set[str]) -> dict | None:
        kept_components = [component for component in candidate["components"] if component["descriptor"] not in excluded_descriptors]
        if not kept_components:
            return None
        raw_weights = np.asarray([float(component["weight"]) for component in kept_components], dtype=np.float64)
        raw_weights = raw_weights / max(float(np.sum(raw_weights)), 1.0e-12)
        indices = np.asarray([descriptor_index[component["descriptor"]] for component in kept_components], dtype=np.int64)
        lane_vector = raw_weights @ descriptor_atoms[indices]
        lane_norm = float(np.linalg.norm(lane_vector))
        if lane_norm <= 1.0e-12:
            return None
        lane_vector = lane_vector / lane_norm
        scores_raw = character_matrix @ lane_vector
        scores_centered = scores_raw - float(np.mean(scores_raw))
        score_norm = float(np.linalg.norm(scores_centered))
        if score_norm <= 1.0e-12:
            return None
        normalized_components = [
            {
                **component,
                "weight": round(float(weight), 8),
            }
            for component, weight in zip(kept_components, raw_weights, strict=True)
        ]
        descriptor_sets_with_component = sum(
            1
            for descriptor_set in character_descriptor_sets
            if any(component["descriptor"] in descriptor_set for component in normalized_components)
        )
        return {
            **candidate,
            "descriptor": lane_label(normalized_components),
            "support": int(descriptor_sets_with_component),
            "variance": float(np.mean(scores_centered * scores_centered)),
            "components": normalized_components,
            "vector": lane_vector,
            "scores_raw": scores_raw,
            "scores_centered": scores_centered,
            "score_norm": score_norm,
            "stripped_component_count": int(len(candidate["components"]) - len(normalized_components)),
        }

    y_rows = []
    for y_candidate in candidates[1:]:
        y_candidate = strip_candidate_components(y_candidate, x_component_descriptors)
        if y_candidate is None:
            continue
        correlation = float((x_candidate["scores_centered"] @ y_candidate["scores_centered"]) / (x_candidate["score_norm"] * y_candidate["score_norm"]))
        lane_cosine = float(x_candidate["vector"] @ y_candidate["vector"])
        positive_correlation = max(correlation, 0.0)
        axis_independence = float(max(0.0, 1.0 - positive_correlation * positive_correlation))
        residual_variance = float(y_candidate["variance"] * axis_independence)
        volume_score = float(x_candidate["variance"] * residual_variance)
        orthogonality_score = float(max(0.0, 1.0 - lane_cosine * lane_cosine))
        objective = float(residual_variance * (orthogonality_score ** orthogonality_penalty))
        y_rows.append(
            (x_candidate, y_candidate, correlation, lane_cosine, volume_score, orthogonality_score, objective, residual_variance)
        )

    def threshold_diagnostics(correlation_threshold: float) -> dict:
        floors = [0.10, 0.25, 0.50]
        rows = {}
        for floor in floors:
            min_variance = floor * float(x_candidate["variance"])
            passing = [pair for pair in y_rows if pair[2] < correlation_threshold and pair[1]["variance"] >= min_variance]
            best = max(
                passing,
                key=lambda pair: (
                    pair[6],
                    pair[7],
                    pair[1]["variance"],
                    -pair[2],
                    -abs(pair[3]),
                    pair[1]["descriptor"],
                ),
                default=None,
            )
            rows[f"{int(floor * 100)}pct"] = {
                "passes": bool(passing),
                "passing_count": int(len(passing)),
                "minimum_y_projection_variance": round(float(min_variance), 10),
                "best_y_descriptor": best[1]["descriptor"] if best else None,
                "best_y_variance": round(float(best[1]["variance"]), 10) if best else None,
                "best_correlation": round(float(best[2]), 10) if best else None,
                "best_lane_cosine": round(float(best[3]), 10) if best else None,
            }
        return {
            "correlation_threshold": correlation_threshold,
            "variance_floors": rows,
        }

    best_orthogonal_variance_y = max(
        y_rows,
        key=lambda pair: (
            pair[1]["variance"] * (pair[5] ** orthogonality_penalty),
            pair[1]["variance"],
            pair[5],
            pair[1]["descriptor"],
        ),
        default=None,
    )
    orthogonal_variance_summary = (
        {
            "descriptor": best_orthogonal_variance_y[1]["descriptor"],
            "variance": round(float(best_orthogonal_variance_y[1]["variance"]), 10),
            "correlation": round(float(best_orthogonal_variance_y[2]), 10),
            "abs_correlation": round(abs(float(best_orthogonal_variance_y[2])), 10),
            "lane_cosine": round(float(best_orthogonal_variance_y[3]), 10),
            "orthogonality_score": round(float(best_orthogonal_variance_y[5]), 10),
            "objective": round(
                float(best_orthogonal_variance_y[1]["variance"] * (best_orthogonal_variance_y[5] ** orthogonality_penalty)),
                10,
            ),
            "components": best_orthogonal_variance_y[1]["components"],
        }
        if best_orthogonal_variance_y
        else None
    )

    threshold_summary = threshold_diagnostics(0.7)
    min_y_variance_fraction = 0.25
    min_y_variance = min_y_variance_fraction * float(x_candidate["variance"])
    eligible_pairs = [pair for pair in y_rows if pair[2] < 0.5 and pair[1]["variance"] >= min_y_variance]
    if not eligible_pairs:
        x_scores_raw = x_candidate["scores_raw"]
        x_mean = float(np.mean(x_scores_raw))
        character_rows_payload = []
        for character, x_score in zip(character_labels, x_scores_raw, strict=True):
            descriptor_set = set(character.get("descriptors") or [])
            x_centered = float(x_score - x_mean)
            character_rows_payload.append(
                {
                    **character,
                    "x_raw_score": round(float(x_score), 10),
                    "x_score": round(x_centered, 10),
                    "amplitude": round(x_centered, 10),
                    "x_direct": any(component["descriptor"] in descriptor_set for component in x_candidate["components"]),
                }
            )
        best_failed_y = max(y_rows, key=lambda pair: pair[6]) if y_rows else None
        best_low_correlation_y = max(
            (pair for pair in y_rows if pair[2] < 0.5),
            key=lambda pair: pair[1]["variance"],
            default=None,
        )
        best_high_variance_y = min(
            (pair for pair in y_rows if pair[1]["variance"] >= min_y_variance),
            key=lambda pair: pair[2],
            default=None,
        )
        return {
            "dimension_count": 1,
            "x_axis": {
                "descriptor": x_candidate["descriptor"],
                "support": x_candidate["support"],
                "variance": round(x_candidate["variance"], 10),
                "seed_descriptor": x_candidate["seed_descriptor"],
                "components": x_candidate["components"],
            },
            "axis_selection": {
                "criterion": "No exact-component-disjoint second lane had character-score correlation below 0.5 while retaining at least 25% of lane 1 variation, so this seiyuu is shown on the single highest-variance positive descriptor lane.",
                "component_scope": component_scope,
                "correlation_threshold_for_2d": 0.5,
                "minimum_y_variance_fraction_of_x": min_y_variance_fraction,
                "x_projection_variance": round(float(x_candidate["variance"]), 10),
                "minimum_y_projection_variance": round(float(min_y_variance), 10),
                "best_rejected_correlation": round(float(best_failed_y[2]), 10) if best_failed_y else None,
                "best_rejected_lane_cosine": round(float(best_failed_y[3]), 10) if best_failed_y else None,
                "best_low_correlation_y_variance": round(float(best_low_correlation_y[1]["variance"]), 10) if best_low_correlation_y else None,
                "best_high_variance_y_correlation": round(float(best_high_variance_y[2]), 10) if best_high_variance_y else None,
                "candidate_count": int(len(candidates)),
                "x_component_count": int(len(x_component_descriptors)),
                "y_candidates_after_component_exclusion": int(len(y_rows)),
                "threshold_diagnostics": threshold_summary,
                "orthogonal_variance_lane": orthogonal_variance_summary,
                "component_exclusion": "Lane 2 candidate mixtures are recomputed after stripping exact descriptor components used by lane 1.",
            },
            "characters": character_rows_payload,
            "center": {"x_raw_mean": round(x_mean, 10)},
            "method": "Each character is the unnormalized sum of its canonical descriptor vectors; x is the dot product onto the highest-variance positive descriptor-mixture lane, centered by subtracting this seiyuu's mean projection.",
        }

    best_pair = max(
        eligible_pairs,
        key=lambda pair: (
            pair[6],
            pair[7],
            pair[1]["variance"],
            -pair[2],
            -abs(pair[3]),
            pair[1]["descriptor"],
        ),
    )
    x_axis, y_axis, axis_correlation, lane_cosine, volume_score, orthogonality_score, objective, residual_variance = best_pair
    x_scores_raw = x_axis["scores_raw"]
    y_scores_raw = y_axis["scores_raw"]
    x_mean = float(np.mean(x_scores_raw))
    y_mean = float(np.mean(y_scores_raw))

    scored_characters = []
    for character, x_score, y_score in zip(character_labels, x_scores_raw, y_scores_raw, strict=True):
        descriptor_set = set(character.get("descriptors") or [])
        scored_characters.append(
            {
                **character,
                "x_raw_score": float(x_score),
                "y_raw_score": float(y_score),
                "x_direct": any(component["descriptor"] in descriptor_set for component in x_axis["components"]),
                "y_direct": any(component["descriptor"] in descriptor_set for component in y_axis["components"]),
            }
        )
    character_rows_payload = []
    for character in scored_characters:
        x_centered = float(character["x_raw_score"] - x_mean)
        y_centered = float(character["y_raw_score"] - y_mean)
        character_rows_payload.append(
            {
                **character,
                "x_raw_score": round(float(character["x_raw_score"]), 10),
                "y_raw_score": round(float(character["y_raw_score"]), 10),
                "x_score": round(x_centered, 10),
                "y_score": round(y_centered, 10),
            }
        )

    return {
        "dimension_count": 2,
        "x_axis": {
            "descriptor": x_axis["descriptor"],
            "support": x_axis["support"],
            "variance": round(x_axis["variance"], 10),
            "seed_descriptor": x_axis["seed_descriptor"],
            "components": x_axis["components"],
        },
        "y_axis": {
            "descriptor": y_axis["descriptor"],
            "support": y_axis["support"],
            "variance": round(y_axis["variance"], 10),
            "seed_descriptor": y_axis["seed_descriptor"],
            "components": y_axis["components"],
        },
        "axis_selection": {
            "criterion": "Two strictly positive descriptor-mixture lanes are selected by maximizing centered 2D volume while penalizing semantic non-orthogonality.",
            "component_scope": component_scope,
            "correlation": round(axis_correlation, 10),
            "lane_cosine": round(lane_cosine, 10),
            "volume_score": round(volume_score, 10),
            "orthogonality_score": round(orthogonality_score, 10),
            "objective": round(objective, 10),
            "y_residual_variance_after_x": round(float(residual_variance), 10),
            "correlation_threshold_for_2d": 0.5,
            "minimum_y_variance_fraction_of_x": min_y_variance_fraction,
            "minimum_y_projection_variance": round(float(min_y_variance), 10),
            "x_projection_variance": round(x_axis["variance"], 10),
            "y_projection_variance": round(y_axis["variance"], 10),
            "neighbor_count": int(neighbor_count),
            "candidate_count": int(len(candidates)),
            "x_component_count": int(len(x_component_descriptors)),
            "y_candidates_after_component_exclusion": int(len(y_rows)),
            "threshold_diagnostics": threshold_summary,
            "orthogonal_variance_lane": orthogonal_variance_summary,
            "component_exclusion": "Lane 2 candidate mixtures are recomputed after stripping exact descriptor components used by lane 1.",
            "selection_order": "x is the highest-variance positive lane; y is chosen by lowest feasible correlation tier, then residual variance, then semantic orthogonality.",
            "similarity_floor": round(float(similarity_floor), 6),
            "orthogonality_penalty": round(float(orthogonality_penalty), 6),
        },
        "characters": character_rows_payload,
        "center": {"x_raw_mean": round(x_mean, 10), "y_raw_mean": round(y_mean, 10)},
        "method": "Each character is the unnormalized sum of its canonical descriptor vectors; x/y are dot products onto positive descriptor-mixture lanes, then centered by subtracting this seiyuu's mean x/y projection.",
    }


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
    sv1_fit_order: str,
    variation_gain_cutoff: float,
    max_variation_terms: int,
) -> dict:
    if broad:
        if sv1_fit_order == "weighted_support":
            decoded = decode_axis_by_weighted_support(
                right_vector,
                descriptor_atoms,
                descriptors,
                character_labels,
                stop_fit=sv1_fit_target,
                min_terms=sv1_min_terms,
                max_terms=sv1_max_terms,
            )
        else:
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
            "display_descriptor_components": annotate_descriptor_support(decoded["descriptors"], character_labels),
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
    canonicalization_payload = read_json(args.global_canonicalization_input) if args.global_canonicalization_input.exists() else {}
    raw_to_canonical = {
        str(raw): str(canonical)
        for raw, canonical in (canonicalization_payload.get("raw_to_canonical") or {}).items()
    }
    contextual_payload = read_json(args.contextual_personality_scores) if args.contextual_personality_scores.exists() else {}
    contextual_scores = {
        str(descriptor): row
        for descriptor, row in (contextual_payload.get("scores_by_descriptor") or {}).items()
    }
    descriptor_rows = union_payload["descriptors"]
    base_descriptor_mask = [
        descriptor_shape_ok(row["tag"], args.descriptor_shape)
        and descriptor_context_ok(
            row["tag"],
            contextual_scores,
            args.min_contextual_personality_score,
            args.min_descriptor_character_count,
        )
        for row in descriptor_rows
    ]
    descriptor_mask = remove_unhyphenated_duplicates(descriptor_rows, base_descriptor_mask)
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
            targets = descriptor_assignment_targets(descriptor, descriptor_index, raw_to_canonical)
            if targets:
                character_descriptors[int(row["anilist_character_id"])].update(targets)
                assignment_count += 1
    merged_personality_assignment_count = 0
    for source in tag_payload.get("characters", []):
        character_id = int(source["anilist_character_id"])
        for tag_row in ((source.get("llm_tags") or {}).get("personality") or []):
            targets = descriptor_assignment_targets(tag_row.get("tag") or "", descriptor_index, raw_to_canonical)
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
        if args.svd_matrix == "z_unit":
            matrix = normalize_rows(matrix)
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
            sv1_fit_order=args.sv1_fit_order,
            variation_gain_cutoff=args.variation_gain_cutoff,
            max_variation_terms=args.max_variation_terms,
        )
        direct_descriptor_planes = {
            "local_supported": direct_descriptor_plane(
                character_labels,
                descriptor_index,
                descriptor_atoms,
                descriptors,
                component_scope="local_supported",
                neighbor_count=args.positive_lane_neighbors,
                candidate_limit=args.positive_lane_candidate_limit,
                similarity_floor=args.positive_lane_similarity_floor,
                orthogonality_penalty=args.positive_lane_orthogonality_penalty,
            ),
            "universal_515": direct_descriptor_plane(
                character_labels,
                descriptor_index,
                descriptor_atoms,
                descriptors,
                component_scope="universal_515",
                neighbor_count=args.positive_lane_neighbors,
                candidate_limit=args.positive_lane_candidate_limit,
                similarity_floor=args.positive_lane_similarity_floor,
                orthogonality_penalty=args.positive_lane_orthogonality_penalty,
            ),
        }
        major_lane["direct_descriptor_planes"] = direct_descriptor_planes
        major_lane["direct_descriptor_plane"] = direct_descriptor_planes["local_supported"]

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
                        sv1_fit_order=args.sv1_fit_order,
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
                "sv1_fit_order": args.sv1_fit_order,
                "variation_gain_cutoff": args.variation_gain_cutoff,
                "shared_role_weight": args.shared_role_weight,
                "row_weight": args.row_weight,
                "svd_matrix": args.svd_matrix,
                "row_weight_field": "ignored by z_unit after final row normalization; otherwise multiplied by shared-role weight",
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
            "global_canonicalization_input": str(args.global_canonicalization_input),
            "contextual_personality_scores": str(args.contextual_personality_scores),
            "embedding_npz": str(args.embedding_npz),
            "tags_input": str(args.tags_input),
            "role_edges": str(args.role_edges),
            "descriptor_count": len(descriptors),
            "descriptor_shape": args.descriptor_shape,
            "contextual_personality_filter": {
                "enabled": bool(contextual_scores),
                "min_score": args.min_contextual_personality_score,
                "min_descriptor_character_count": args.min_descriptor_character_count,
                "scoring": contextual_payload.get("parameters", {}).get("score") if contextual_payload else None,
            },
            "orthogonal_rank": int(np.sum(keep)),
            "sv_relative_cutoff": args.sv_relative_cutoff,
            "sv1_fit_target": args.sv1_fit_target,
            "sv1_min_terms": args.sv1_min_terms,
            "sv1_max_terms": args.sv1_max_terms,
            "sv1_fit_order": args.sv1_fit_order,
            "variation_gain_cutoff": args.variation_gain_cutoff,
            "max_variation_terms": args.max_variation_terms,
            "max_role_edge_count": args.max_role_edge_count,
            "shared_role_weight": args.shared_role_weight,
            "row_weight": args.row_weight,
            "svd_matrix": args.svd_matrix,
            "row_weight_field": "ignored by z_unit after final row normalization; otherwise multiplied by shared-role weight",
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
