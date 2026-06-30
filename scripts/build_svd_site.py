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

ADJECTIVE_POS_TAGS = {"JJ", "JJR", "JJS"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the static SV1-first seiyuu profiler payload.")
    parser.add_argument("--site-profile-input", type=Path, default=Path("site/profiles.json"))
    parser.add_argument("--character-display-input", type=Path, default=Path("site/character_display.json"))
    parser.add_argument("--role-edges", type=Path, default=Path("data/role_edges.json"))
    parser.add_argument(
        "--tags-input",
        type=Path,
        default=Path("data/external/merged/all_characters_llm_vndb_personality_tags.json"),
    )
    parser.add_argument(
        "--matrix-metadata",
        type=Path,
        default=Path("models/tag_descriptor_matrices_llm_only/all_characters_llm_only_personality_traits_matrix_metadata.json"),
    )
    parser.add_argument(
        "--basis-json",
        type=Path,
        default=Path(
            "models/global_descriptor_basis/"
            "global_qwen_gloss_descriptor_pivoted_cholesky_filterpure_adjective_centernone_priorityrow_sum_r384_trace1e-06_pivot1e-12.json"
        ),
    )
    parser.add_argument(
        "--basis-npz",
        type=Path,
        default=Path(
            "models/global_descriptor_basis/"
            "global_qwen_gloss_descriptor_pivoted_cholesky_filterpure_adjective_centernone_priorityrow_sum_r384_trace1e-06_pivot1e-12.npz"
        ),
    )
    parser.add_argument(
        "--glosses-json",
        type=Path,
        default=Path(
            "models/global_ollama_descriptor_glosses/"
            "all_characters_llm_only_personality_traits_qwen3_5_4b_personality_traits_filtered_all_ollama_glosses.json"
        ),
    )
    parser.add_argument(
        "--glosses-npz",
        type=Path,
        default=Path(
            "models/global_ollama_descriptor_glosses/"
            "all_characters_llm_only_personality_traits_qwen3_5_4b_personality_traits_filtered_all_ollama_glosses.npz"
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("site/profiles.json"))
    parser.add_argument("--profile-dir", type=Path, default=Path("site/profile_payloads"))
    parser.add_argument("--max-profiles", type=int, default=0, help="Optional cap for quick local smoke tests.")
    parser.add_argument("--sample-count", type=int, default=18)
    parser.add_argument(
        "--row-weight",
        choices=["none", "sqrt_log_favourites", "log_favourites", "sqrt_favourites"],
        default="none",
        help="Optional character popularity weighting applied to rows before per-seiyuu SVD.",
    )
    parser.add_argument(
        "--profile-center",
        choices=["none", "global_unweighted", "global_weighted"],
        default="none",
        help="Optional global character centroid subtraction before per-seiyuu SVD.",
    )
    parser.add_argument(
        "--character-space",
        choices=["raw", "unit", "centered_unit"],
        default="raw",
        help=(
            "Character row space for per-seiyuu SVD. centered_unit uses "
            "row_normalize((B @ G @ X) - global_character_centroid)."
        ),
    )
    parser.add_argument(
        "--descriptor-scope",
        choices=["kitchen_sink", "no_roles"],
        default="kitchen_sink",
        help=(
            "Descriptor sources to include. no_roles keeps LLM personality/traits and VNDB non-role tags, "
            "and skips role/categoryless raw fallback tags."
        ),
    )
    parser.add_argument(
        "--max-role-edge-count",
        type=int,
        default=20,
        help=(
            "Exclude characters credited to more than this many seiyuu. This filters shared/gimmick "
            "character nodes such as Pop Team Epic variants and omnibus narrators."
        ),
    )
    parser.add_argument(
        "--shared-role-weight",
        choices=["none", "inverse_sqrt", "inverse"],
        default="inverse_sqrt",
        help="Downweight characters credited to multiple seiyuu after hard shared-role filtering.",
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
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "value"


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


def percent_rows(values: np.ndarray, labels: list[dict]) -> list[dict]:
    norm2 = float(np.sum(values * values))
    rows = []
    for index in np.argsort(values * values)[::-1]:
        amplitude = float(values[int(index)])
        share = (amplitude * amplitude / norm2) if norm2 > 0 else 0.0
        rows.append(
            {
                **labels[int(index)],
                "amplitude": round(amplitude, 10),
                "abs_amplitude": round(abs(amplitude), 10),
                "percent": round(share * 100.0, 6),
            }
        )
    return rows


def signed_percent_rows(values: np.ndarray, labels: list[dict], sign: int) -> list[dict]:
    if sign not in {-1, 1}:
        raise ValueError("sign must be -1 or 1")
    rows = percent_rows(values, labels)
    if sign > 0:
        return [row for row in rows if float(row["amplitude"]) > 0.0]
    return [row for row in rows if float(row["amplitude"]) < 0.0]


def attach_component_characters(
    component_rows: list[dict],
    row_matrix: np.ndarray,
    component_vector: np.ndarray,
    character_labels: list[dict],
    index_key: str,
    max_characters: int = 3,
    epsilon: float = 1.0e-9,
) -> list[dict]:
    output = []
    for row in component_rows:
        component_index = row.get(index_key)
        if component_index is None:
            output.append(row)
            continue
        component_index = int(component_index)
        direction = 1.0 if float(component_vector[component_index]) >= 0 else -1.0
        aligned_values = row_matrix[:, component_index] * direction
        order = np.argsort(aligned_values)[::-1]
        top_characters = []
        for character_index in order:
            projection = float(aligned_values[int(character_index)])
            if projection <= epsilon:
                continue
            top_characters.append(
                {
                    **character_labels[int(character_index)],
                    "projection": round(projection, 10),
                    "raw_projection": round(float(row_matrix[int(character_index), component_index]), 10),
                }
            )
            if len(top_characters) >= max_characters:
                break
        output.append({**row, "top_characters": top_characters})
    return output


def sparse_descriptor_components(
    character_axis_values: np.ndarray,
    character_descriptor_indices: list[list[int]],
    descriptor_similarity: np.ndarray,
    descriptor_thresholds: np.ndarray,
    display_descriptor_indices: set[int],
    original_descriptor_labels: list[dict],
    character_labels: list[dict],
    max_rows: int = 25,
    max_characters: int = 3,
    cluster_similarity: float = 0.90,
    coverage_target: float = 0.95,
    full_support_effective_characters: float = 3.0,
    epsilon: float = 1.0e-9,
) -> list[dict]:
    positive_weights = np.maximum(np.asarray(character_axis_values, dtype=np.float64), 0.0)
    if float(np.sum(positive_weights)) <= epsilon:
        positive_weights = np.abs(np.asarray(character_axis_values, dtype=np.float64))

    display_index_array = np.asarray(sorted(display_descriptor_indices), dtype=np.int64)
    if display_index_array.size == 0:
        return []

    scores: dict[int, float] = defaultdict(float)
    evidence: dict[int, list[dict]] = defaultdict(list)
    for character_index, (weight, descriptor_indices) in enumerate(
        zip(positive_weights, character_descriptor_indices, strict=True)
    ):
        if weight <= epsilon or not descriptor_indices:
            continue
        owned = np.asarray(descriptor_indices, dtype=np.int64)
        similarities = descriptor_similarity[display_index_array][:, owned]
        best_owned_position = np.argmax(similarities, axis=1)
        best_similarities = similarities[np.arange(display_index_array.size), best_owned_position]
        best_owned_indices = owned[best_owned_position]
        target_thresholds = descriptor_thresholds[display_index_array]
        source_thresholds = descriptor_thresholds[best_owned_indices]
        supported_positions = np.flatnonzero(
            (best_similarities >= target_thresholds) & (best_similarities >= source_thresholds)
        )
        character_candidates = []
        for descriptor_position in supported_positions:
            descriptor_index = int(display_index_array[int(descriptor_position)])
            similarity = float(best_similarities[int(descriptor_position)])
            target_threshold = float(target_thresholds[int(descriptor_position)])
            source_threshold = float(source_thresholds[int(descriptor_position)])
            target_margin = (similarity - target_threshold) / max(1.0 - target_threshold, epsilon)
            source_margin = (similarity - source_threshold) / max(1.0 - source_threshold, epsilon)
            support_strength = max(0.0, min(target_margin, source_margin))
            if support_strength <= epsilon:
                continue
            matched_descriptor_index = int(owned[int(best_owned_position[int(descriptor_position)])])
            character_candidates.append(
                (
                    descriptor_index,
                    support_strength,
                    similarity,
                    matched_descriptor_index,
                )
            )

        candidate_total = float(sum(candidate[1] for candidate in character_candidates))
        if candidate_total <= epsilon:
            continue
        for descriptor_index, support_strength, similarity, matched_descriptor_index in character_candidates:
            projection = float(weight * support_strength / candidate_total)
            scores[descriptor_index] += projection
            evidence[descriptor_index].append(
                {
                    **character_labels[character_index],
                    "projection": round(projection, 10),
                    "similarity": round(similarity, 6),
                    "matched_descriptor": original_descriptor_labels[matched_descriptor_index]["descriptor"],
                }
            )

    raw_total = float(sum(scores.values()))
    if raw_total <= epsilon:
        return []

    clusters = []
    for descriptor_index, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        best_cluster_index = None
        best_similarity = -1.0
        for cluster_index, cluster in enumerate(clusters):
            similarity = float(descriptor_similarity[descriptor_index, cluster["representative_index"]])
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster_index = cluster_index
        if best_cluster_index is not None and best_similarity >= cluster_similarity:
            cluster = clusters[best_cluster_index]
            cluster["members"].append(descriptor_index)
            cluster["score"] += score
        else:
            clusters.append(
                {
                    "representative_index": descriptor_index,
                    "members": [descriptor_index],
                    "score": float(score),
                }
            )

    for cluster in clusters:
        member_indices = sorted(cluster["members"], key=lambda index: scores[index], reverse=True)
        evidence_by_character: dict[int, dict] = {}
        for member_index in member_indices:
            for row in evidence[member_index]:
                character_id = int(row["character_id"])
                current = evidence_by_character.get(character_id)
                if current is None:
                    evidence_by_character[character_id] = {
                        **row,
                        "projection": 0.0,
                        "similarity": 0.0,
                        "matched_descriptors": [],
                    }
                current = evidence_by_character[character_id]
                current["projection"] += float(row["projection"])
                current["similarity"] = max(float(current["similarity"]), float(row.get("similarity") or 0.0))
                matched_descriptor = row.get("matched_descriptor")
                if matched_descriptor and matched_descriptor not in current["matched_descriptors"]:
                    current["matched_descriptors"].append(matched_descriptor)

        projections = np.asarray(
            [float(row["projection"]) for row in evidence_by_character.values() if float(row["projection"]) > epsilon],
            dtype=np.float64,
        )
        effective_support = 0.0
        if projections.size:
            effective_support = float(np.sum(projections) ** 2 / max(np.sum(projections * projections), epsilon))
        support_factor = min(1.0, effective_support / max(full_support_effective_characters, epsilon))
        cluster["member_indices"] = member_indices
        cluster["evidence_by_character"] = evidence_by_character
        cluster["effective_support"] = effective_support
        cluster["support_factor"] = support_factor
        cluster["adjusted_score"] = float(cluster["score"]) * support_factor

    display_total = float(sum(cluster["adjusted_score"] for cluster in clusters))
    if display_total <= epsilon:
        return []

    clusters.sort(key=lambda cluster: cluster["score"], reverse=True)
    clusters.sort(key=lambda cluster: cluster["adjusted_score"], reverse=True)
    output = []
    covered = 0.0
    for rank, cluster in enumerate(clusters, start=1):
        member_indices = cluster["member_indices"]
        score = float(cluster["adjusted_score"])
        covered += score
        evidence_by_character = cluster["evidence_by_character"]

        top_characters = sorted(
            evidence_by_character.values(),
            key=lambda row: row["projection"],
            reverse=True,
        )
        top_characters = [row for row in top_characters if row["projection"] > epsilon][:max_characters]
        for row in top_characters:
            row["projection"] = round(float(row["projection"]), 10)
            row["similarity"] = round(float(row["similarity"]), 6)
            if row["matched_descriptors"]:
                row["matched_descriptor"] = " / ".join(row["matched_descriptors"][:3])
            row.pop("matched_descriptors", None)

        member_labels = [original_descriptor_labels[index]["descriptor"] for index in member_indices]
        display_members = member_labels[:4]
        descriptor = " / ".join(display_members)
        if len(member_labels) > len(display_members):
            descriptor = f"{descriptor} / +{len(member_labels) - len(display_members)}"

        output.append(
            {
                **original_descriptor_labels[int(cluster["representative_index"])],
                "descriptor": descriptor,
                "rank": rank,
                "amplitude": round(float(score), 10),
                "abs_amplitude": round(float(abs(score)), 10),
                "raw_amplitude": round(float(cluster["score"]), 10),
                "percent": round(float(score / display_total * 100.0), 6),
                "raw_percent": round(float(cluster["score"] / raw_total * 100.0), 6),
                "coverage_percent": round(float(covered / display_total * 100.0), 6),
                "member_count": len(member_labels),
                "members": member_labels,
                "support": len(evidence_by_character),
                "effective_support": round(float(cluster["effective_support"]), 6),
                "support_factor": round(float(cluster["support_factor"]), 6),
                "top_characters": top_characters,
            }
        )
        if covered / display_total >= coverage_target or len(output) >= max_rows:
            break
    return output


def participation_metrics(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    norm2 = float(np.sum(values * values))
    if norm2 <= 0:
        return {
            "effective_count": 0.0,
            "normalized_entanglement": 0.0,
            "top5_mass_percent": 0.0,
            "top10_mass_percent": 0.0,
        }
    probabilities = (values * values) / norm2
    effective_count = 1.0 / max(float(np.sum(probabilities * probabilities)), 1.0e-12)
    sorted_probabilities = np.sort(probabilities)[::-1]
    return {
        "effective_count": round(float(effective_count), 6),
        "normalized_entanglement": round(float(effective_count / len(probabilities)), 6),
        "top5_mass_percent": round(float(np.sum(sorted_probabilities[:5]) * 100.0), 6),
        "top10_mass_percent": round(float(np.sum(sorted_probabilities[:10]) * 100.0), 6),
    }


def effective_rank(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[values > 1.0e-12]
    if len(values) == 0:
        return 0.0
    probabilities = values / float(np.sum(values))
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def covariance_range_metrics(matrix: np.ndarray) -> dict:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[0] < 2:
        return {
            "role_range": 0.0,
            "semantic_spread": 0.0,
            "axis_concentration_percent": 0.0,
            "lane_count": 0.0,
            "direction_range": 0.0,
            "direction_spread": 0.0,
            "direction_axis_concentration_percent": 0.0,
            "direction_lane_count": 0.0,
        }

    local = matrix - matrix.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(local, full_matrices=False, compute_uv=False)
    eigenvalues = (singular_values * singular_values) / max(matrix.shape[0] - 1, 1)
    trace = float(np.sum(eigenvalues))

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    unit_matrix = matrix / np.maximum(norms, 1.0e-12)
    unit_local = unit_matrix - unit_matrix.mean(axis=0, keepdims=True)
    unit_singular_values = np.linalg.svd(unit_local, full_matrices=False, compute_uv=False)
    unit_eigenvalues = (unit_singular_values * unit_singular_values) / max(matrix.shape[0] - 1, 1)
    unit_trace = float(np.sum(unit_eigenvalues))

    return {
        "role_range": round(float(np.sqrt(eigenvalues[0])) if trace > 0 else 0.0, 6),
        "semantic_spread": round(float(np.sqrt(trace)) if trace > 0 else 0.0, 6),
        "axis_concentration_percent": round(float(eigenvalues[0] / trace * 100.0) if trace > 0 else 0.0, 6),
        "lane_count": round(effective_rank(eigenvalues), 6),
        "direction_range": round(float(np.sqrt(unit_eigenvalues[0])) if unit_trace > 0 else 0.0, 6),
        "direction_spread": round(float(np.sqrt(unit_trace)) if unit_trace > 0 else 0.0, 6),
        "direction_axis_concentration_percent": round(
            float(unit_eigenvalues[0] / unit_trace * 100.0) if unit_trace > 0 else 0.0,
            6,
        ),
        "direction_lane_count": round(effective_rank(unit_eigenvalues), 6),
    }


def descriptor_label(row: dict, rank: int) -> dict:
    return {
        "rank": rank,
        "descriptor": row["descriptor"],
        "descriptor_index": int(row["descriptor_index"]),
    }


def normalized_descriptor(value: str) -> str:
    return re.sub(r"[^a-z0-9+-]+", " ", value.lower()).strip()


def coordinated_descriptor_parts(value: str) -> list[str]:
    value = re.sub(r"\s+", " ", value.strip())
    value = re.sub(r"^(?:both|either)\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*(?:,|;|&|\band\b|\bor\b)\s*", "|", value, flags=re.IGNORECASE)
    return [part.strip(" |") for part in value.split("|") if part.strip(" |")]


def known_descriptor_values(value: str, descriptor_index: dict[str, int]) -> list[str]:
    parts = [normalized_descriptor(part) for part in coordinated_descriptor_parts(value)]
    if len(parts) >= 2 and all(part in descriptor_index for part in parts):
        return list(dict.fromkeys(parts))

    descriptor = normalized_descriptor(value)
    if descriptor in descriptor_index:
        return [descriptor]
    return []


def parse_json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


def cached_raw_tags(character: dict) -> dict:
    cache_path = character.get("llm_raw_cache")
    if not cache_path:
        return {}
    path = Path(cache_path)
    if not path.exists():
        return {}
    cached = read_json(path)
    content = cached.get("response", {}).get("message", {}).get("content", "")
    return parse_json_object(content)


def cached_raw_tag_values(character: dict) -> list[str]:
    cache_path = character.get("llm_raw_cache")
    if not cache_path:
        return []
    path = Path(cache_path)
    if not path.exists():
        return []
    cached = read_json(path)
    content = cached.get("response", {}).get("message", {}).get("content", "")
    return re.findall(r'"tag"\s*:\s*"([^"]+)"', content)


def add_if_known(output: list[str], value: str, descriptor_index: dict[str, int]) -> None:
    output.extend(known_descriptor_values(value, descriptor_index))


def descriptors_from_tag_row(character: dict, descriptor_index: dict[str, int], descriptor_scope: str) -> list[str]:
    descriptors = []
    llm_categories = ["role", "personality", "traits"] if descriptor_scope == "kitchen_sink" else ["personality", "traits"]
    for category in llm_categories:
        for tag in character.get("llm_tags", {}).get(category, []):
            add_if_known(descriptors, str(tag.get("tag") or ""), descriptor_index)

    raw_tags = cached_raw_tags(character)
    for category in llm_categories:
        for tag in raw_tags.get(category, []):
            if not isinstance(tag, dict):
                continue
            add_if_known(descriptors, str(tag.get("tag") or ""), descriptor_index)
    if descriptor_scope == "kitchen_sink" and not raw_tags:
        for value in cached_raw_tag_values(character):
            add_if_known(descriptors, value, descriptor_index)

    for tag in character.get("merged_descriptor_sources", {}).get("accepted_vndb", []):
        group = str(tag.get("group") or tag.get("category") or "").lower()
        if descriptor_scope == "no_roles" and group == "role":
            continue
        add_if_known(descriptors, str(tag.get("tag") or tag.get("name") or ""), descriptor_index)

    return sorted(set(descriptors))


def descriptor_scope_description(descriptor_scope: str) -> str:
    if descriptor_scope == "no_roles":
        return (
            "vocabulary-matched descriptors from LLM personality/traits and accepted VNDB non-role tags; "
            "LLM role tags, rejected VNDB tags, and categoryless raw fallbacks are excluded"
        )
    return (
        "role-inclusive vocabulary-matched descriptors from LLM role/personality/traits, "
        "raw LLM cache fallbacks, and accepted VNDB tags"
    )


def import_nltk() -> Any:
    try:
        import nltk
        from nltk.corpus import wordnet as wn
    except ImportError as error:
        raise RuntimeError("Install nltk to build adjective-only display descriptors.") from error
    nltk.pos_tag(["a", "shy", "character"])
    return nltk, wn


def descriptor_tokens(value: str) -> list[str]:
    return re.findall(r"[a-z]+(?:-[a-z]+)*", value.lower())


def wordnet_has_adjective(wn: Any, token: str) -> bool:
    normalized = token.replace("-", "_")
    return bool(wn.synsets(normalized, pos=wn.ADJ) or wn.synsets(normalized, pos=wn.ADJ_SAT))


def wordnet_has_noun(wn: Any, token: str) -> bool:
    return bool(wn.synsets(token.replace("-", "_"), pos=wn.NOUN))


def pure_adjective_descriptor(value: str, nltk: Any, wn: Any) -> bool:
    tokens = descriptor_tokens(value)
    if not tokens:
        return False
    tagged = nltk.pos_tag(["a", *tokens, "character"])[1:-1]
    if len(tokens) == 1 and wordnet_has_adjective(wn, tokens[0]):
        return True
    if len(tokens) == 1 and "-" in tokens[0] and tagged[0][1] in ADJECTIVE_POS_TAGS:
        return True
    if not all(tag in ADJECTIVE_POS_TAGS for _, tag in tagged):
        return False
    if not all(wordnet_has_adjective(wn, token) or "-" in token for token in tokens):
        return False
    return not wordnet_has_noun(wn, tokens[-1])


def enrich_character_display(character: dict, display: dict | None) -> dict:
    return {
        "character_id": int(character["character_id"]),
        "name": character.get("name") or "",
        "native_name": (display or {}).get("native_name") or "",
        "anime": character.get("first_anime") or (display or {}).get("anime_title") or "",
        "image": (display or {}).get("image") or "",
        "site_url": character.get("site_url") or (display or {}).get("site_url") or "",
        "anime_url": (display or {}).get("anime_url") or "",
        "role": (display or {}).get("role") or "",
        "favourites": int(character.get("favourites") or 0),
    }


def character_row_weight(character: dict, mode: str) -> float:
    favourites = max(float(character.get("favourites") or 0.0), 0.0)
    if mode == "none":
        return 1.0
    if mode == "sqrt_log_favourites":
        return math.sqrt(math.log1p(favourites) + 1.0)
    if mode == "log_favourites":
        return math.log1p(favourites) + 1.0
    if mode == "sqrt_favourites":
        return math.sqrt(favourites + 1.0)
    raise ValueError(f"unknown row-weight mode: {mode}")


def shared_role_weight(character: dict, mode: str) -> float:
    role_edge_count = max(int(character.get("role_edge_count") or 1), 1)
    if mode == "none":
        return 1.0
    if mode == "inverse_sqrt":
        return 1.0 / math.sqrt(role_edge_count)
    if mode == "inverse":
        return 1.0 / role_edge_count
    raise ValueError(f"unknown shared-role-weight mode: {mode}")


def is_pop_team_character(character: dict) -> bool:
    anime = norm_name(character.get("first_anime") or "")
    name = norm_name(character.get("name") or "")
    return (
        "pop team epic" in anime
        or "poputepipikku" in anime
        or name in {"pipimi", "popuko"}
    )


def profile_sort_score(profile: dict) -> tuple[float, float, int, int]:
    range_metrics = profile["major_lane"]["range_metrics"]
    supported_characters = int(profile["character_count"])
    support_gate = 1.0 if supported_characters >= 10 else 0.0
    return (
        support_gate,
        range_metrics["role_range"],
        supported_characters,
        int(profile["role_count"]),
    )


def projected_character_row(
    character: dict,
    descriptor_to_basis_index: dict[str, int],
    projection: np.ndarray,
) -> tuple[np.ndarray, list[int]] | None:
    descriptor_indices = [
        descriptor_to_basis_index[descriptor]
        for descriptor in character.get("descriptors", [])
        if descriptor in descriptor_to_basis_index
    ]
    if not descriptor_indices:
        return None
    return np.sum(projection[descriptor_indices], axis=0), descriptor_indices


def sv1_profile(
    profile: dict,
    characters: list[dict],
    descriptor_to_basis_index: dict[str, int],
    projection: np.ndarray,
    basis_labels: list[dict],
    original_descriptor_labels: list[dict],
    display_descriptor_indices: set[int],
    descriptor_similarity: np.ndarray,
    descriptor_thresholds: np.ndarray,
    display_by_character_id: dict[int, dict],
    row_weight_mode: str,
    shared_role_weight_mode: str,
    center_vector: np.ndarray | None,
    profile_center: str,
    character_space: str,
) -> dict | None:
    rows = []
    row_weights = []
    supported_descriptor_indices = set()
    descriptor_support: dict[int, list[str]] = defaultdict(list)
    for character in characters:
        projected = projected_character_row(character, descriptor_to_basis_index, projection)
        if projected is None:
            continue
        row_vector, descriptor_indices = projected
        if center_vector is not None:
            row_vector = row_vector - center_vector
        for descriptor_index in descriptor_indices:
            if descriptor_index in display_descriptor_indices:
                supported_descriptor_indices.add(descriptor_index)
                descriptor_support[descriptor_index].append(character.get("name") or "")
        rows.append((character, row_vector, descriptor_indices))
        row_weights.append(
            character_row_weight(character, row_weight_mode)
            * shared_role_weight(character, shared_role_weight_mode)
        )
    if not rows:
        return None

    unweighted_matrix = np.vstack([row[1] for row in rows])
    range_metrics = covariance_range_metrics(unweighted_matrix)
    if character_space in {"unit", "centered_unit"}:
        norms = np.linalg.norm(unweighted_matrix, axis=1, keepdims=True)
        unweighted_matrix = unweighted_matrix / np.maximum(norms, 1.0e-12)
    weight_vector = np.asarray(row_weights, dtype=np.float64)
    matrix = unweighted_matrix * weight_vector.reshape(-1, 1)
    left, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    singular_mass = singular_values * singular_values
    singular_mass_total = float(np.sum(singular_mass))
    u = left[:, 0].copy()
    v = vt[0].copy()
    favourite_weights = np.asarray(
        [np.log1p(max(float(character.get("favourites") or 0.0), 0.0)) for character, _, _ in rows],
        dtype=np.float64,
    )
    salient_scores = np.abs(u) * favourite_weights
    if float(np.max(salient_scores)) <= 0.0:
        salient_scores = np.abs(u)
    orientation_pivot = int(np.argmax(salient_scores))
    if u[orientation_pivot] < 0:
        u *= -1.0
        v *= -1.0

    character_labels = []
    for (character, _, _), weight in zip(rows, row_weights, strict=True):
        character_labels.append(
            {
                **enrich_character_display(character, display_by_character_id.get(int(character["character_id"]))),
                "row_weight": round(float(weight), 8),
            }
        )
    character_rows = percent_rows(u, character_labels)
    display_component_rows = sparse_descriptor_components(
        u,
        [descriptor_indices for _, _, descriptor_indices in rows],
        descriptor_similarity,
        descriptor_thresholds,
        display_descriptor_indices,
        original_descriptor_labels,
        character_labels,
    )
    opposite_display_component_rows = sparse_descriptor_components(
        -u,
        [descriptor_indices for _, _, descriptor_indices in rows],
        descriptor_similarity,
        descriptor_thresholds,
        display_descriptor_indices,
        original_descriptor_labels,
        character_labels,
    )
    return {
        "rank": 1,
        "singular_value": round(float(singular_values[0]), 10),
        "singular_spectrum": [
            {
                "rank": index + 1,
                "singular_value": round(float(value), 10),
                "mass_percent": round(
                    float(singular_mass[index] / singular_mass_total * 100.0) if singular_mass_total > 0 else 0.0,
                    6,
                ),
            }
            for index, value in enumerate(singular_values[: min(12, len(singular_values))])
        ],
        "row_weight_mode": row_weight_mode,
        "shared_role_weight_mode": shared_role_weight_mode,
        "profile_center": profile_center,
        "character_space": character_space,
        "basis_participation": participation_metrics(v),
        "character_participation": participation_metrics(u),
        "range_metrics": range_metrics,
        "explained_sv_l2_percent": round(
            float(singular_mass[0] / singular_mass_total * 100.0) if singular_mass_total > 0 else 0.0,
            6,
        ),
        "component_basis": (
            "per-seiyuu semantic descriptor clusters built from sparse original descriptors supported by "
            "mutual local 99th-percentile semantic matches to character-owned descriptors; each character's "
            "display contribution is conserved, clusters are penalized when support comes from too few "
            "effective characters, and clusters are reported to 95% descriptor-mass coverage"
        ),
        "display_descriptor_components": display_component_rows,
        "opposite_display_descriptor_components": opposite_display_component_rows,
        "characters": character_rows,
        "positive_characters": signed_percent_rows(u, character_labels, 1),
        "negative_characters": signed_percent_rows(u, character_labels, -1),
    }


def main() -> None:
    args = parse_args()
    old_payload = read_json(args.site_profile_input)
    character_display_payload = read_json(args.character_display_input) if args.character_display_input.exists() else {}
    tag_payload = read_json(args.tags_input)
    role_payload = read_json(args.role_edges) if args.role_edges.exists() else {}
    basis_payload = read_json(args.basis_json)
    basis_npz = np.load(args.basis_npz)
    gloss_payload = read_json(args.glosses_json)
    gloss_embeddings = np.load(args.glosses_npz)["variant_embeddings"].astype(np.float64)
    nltk, wn = import_nltk()

    old_profiles = old_payload["profiles"]
    if args.max_profiles > 0:
        old_profiles = old_profiles[: args.max_profiles]

    old_character_display: dict[int, dict] = {
        int(character_id): character
        for character_id, character in (character_display_payload.get("characters") or {}).items()
    }
    favourite_metadata = {
        "source": "AniList",
        "field": "character.favourites",
        "cached_at": role_payload.get("generated_at") or tag_payload.get("generated_at") or "",
        "tag_cache_generated_at": tag_payload.get("generated_at") or "",
    }
    if role_payload:
        for role in role_payload.get("roles", []):
            character = role.get("character") or {}
            character_id = character.get("character_id")
            if character_id is None:
                continue
            anime = role.get("anime") or []
            first_anime = anime[0] if anime else {}
            old_character_display[int(character_id)] = {
                "character_id": int(character_id),
                "name": character.get("name") or "",
                "native_name": character.get("native_name") or "",
                "image": character.get("image") or "",
                "site_url": character.get("site_url") or "",
                "anime_title": character.get("first_anime") or first_anime.get("title") or "",
                "anime_url": first_anime.get("mal_url") or first_anime.get("site_url") or "",
                "favourites": character.get("favourites") or 0,
                "role": role.get("character_role") or "",
            }
    for old_profile in old_payload.get("profiles", []):
        for lane in old_profile.get("lanes", []):
            for character in lane.get("characters", []):
                character_id = character.get("character_id")
                if character_id is not None and character_id not in old_character_display:
                    old_character_display[int(character_id)] = character

    all_descriptor_rows = gloss_payload["rows"]
    all_descriptors = [row["descriptor"] for row in all_descriptor_rows]
    descriptor_to_basis_index = {descriptor: index for index, descriptor in enumerate(all_descriptors)}
    seiyuu_to_characters: dict[str, list[dict]] = defaultdict(list)
    source_characters: list[dict] = []
    source_character_ids = set()
    excluded_shared_characters = []
    for source in tag_payload.get("characters", []):
        role_edge_count = len(source.get("seiyuu") or [])
        if role_edge_count > args.max_role_edge_count or is_pop_team_character(source):
            excluded_shared_characters.append(
                {
                    "character_id": int(source["anilist_character_id"]),
                    "name": source.get("name") or "",
                    "first_anime": source.get("first_anime") or "",
                    "role_edge_count": role_edge_count,
                    "reason": "pop_team_epic" if is_pop_team_character(source) else "role_edge_count",
                }
            )
            continue
        descriptors = descriptors_from_tag_row(source, descriptor_to_basis_index, args.descriptor_scope)
        if not descriptors:
            continue
        character = {
            "character_id": int(source["anilist_character_id"]),
            "name": source.get("name") or "",
            "first_anime": source.get("first_anime") or "",
            "favourites": int(source.get("favourites") or 0),
            "site_url": source.get("site_url") or "",
            "descriptors": descriptors,
            "role_edge_count": max(role_edge_count, 1),
        }
        character_id = int(character["character_id"])
        if character_id not in source_character_ids:
            source_character_ids.add(character_id)
            source_characters.append(character)
        for seiyuu in source.get("seiyuu", []):
            for key in name_keys(seiyuu.get("name") or ""):
                seiyuu_to_characters[key].append(character)

    original_descriptor_labels = [{} for _ in all_descriptors]
    display_descriptor_indices = set()
    for descriptor_index, row in enumerate(all_descriptor_rows):
        label = descriptor_label({"descriptor": row["descriptor"], "descriptor_index": descriptor_index}, descriptor_index + 1)
        original_descriptor_labels[descriptor_index] = label
        if pure_adjective_descriptor(row["descriptor"], nltk, wn):
            display_descriptor_indices.add(descriptor_index)
    pivot_indices = basis_npz["pivot_indices"].astype(np.int64)
    source_descriptor_indices = basis_npz["source_descriptor_indices"].astype(np.int64)
    embeddings = gloss_embeddings.mean(axis=1)
    embeddings = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1.0e-12)
    descriptor_similarity = np.clip(
        embeddings.astype(np.float32) @ embeddings.astype(np.float32).T,
        -1.0,
        1.0,
    ).astype(np.float32)
    threshold_reference = descriptor_similarity.copy()
    np.fill_diagonal(threshold_reference, -1.0)
    descriptor_thresholds = np.quantile(threshold_reference, 0.99, axis=1).astype(np.float32)
    del threshold_reference
    basis_centering = basis_payload.get("parameters", {}).get("basis_centering", "none")
    filtered_basis_embeddings = basis_npz["E_raw"].astype(np.float64)
    if basis_centering == "mean":
        basis_mean = filtered_basis_embeddings.mean(axis=0, keepdims=True)
        projection_embeddings = embeddings - basis_mean
        pivot_embedding_source = filtered_basis_embeddings - basis_mean
    elif basis_centering == "none":
        projection_embeddings = embeddings
        pivot_embedding_source = filtered_basis_embeddings
    else:
        raise ValueError(f"unknown basis_centering in basis payload: {basis_centering}")
    pivot_source_indices = source_descriptor_indices[pivot_indices]
    pivot_embeddings = pivot_embedding_source[pivot_indices]
    overlap_to_pivots = projection_embeddings @ pivot_embeddings.T
    pivot_overlap = pivot_embeddings @ pivot_embeddings.T
    eigenvalues, eigenvectors = np.linalg.eigh((pivot_overlap + pivot_overlap.T) * 0.5)
    keep = eigenvalues > max(pivot_overlap.shape) * np.finfo(np.float64).eps * max(float(eigenvalues.max()), 1.0) * 100.0
    inverse_sqrt = eigenvectors[:, keep] @ np.diag(1.0 / np.sqrt(eigenvalues[keep])) @ eigenvectors[:, keep].T
    projection = overlap_to_pivots @ inverse_sqrt
    basis_labels = [
        {**descriptor_label(pivot, rank + 1), "basis_index": rank}
        for rank, pivot in enumerate(basis_payload["pivots"])
    ]
    center_vector = None
    effective_profile_center = (
        "global_unweighted"
        if args.character_space == "centered_unit" and args.profile_center == "none"
        else args.profile_center
    )
    if effective_profile_center != "none":
        center_rows = []
        center_weights = []
        for character in source_characters:
            projected = projected_character_row(character, descriptor_to_basis_index, projection)
            if projected is None:
                continue
            center_rows.append(projected[0])
            if effective_profile_center == "global_weighted":
                center_weights.append(
                    character_row_weight(character, args.row_weight)
                    * shared_role_weight(character, args.shared_role_weight)
                )
            else:
                center_weights.append(1.0)
        center_matrix = np.vstack(center_rows)
        center_weight_vector = np.asarray(center_weights, dtype=np.float64)
        center_vector = np.average(center_matrix, axis=0, weights=center_weight_vector)

    profiles = []
    for profile in old_profiles:
        seen = set()
        characters = []
        for key in alias_keys(profile):
            for character in seiyuu_to_characters.get(key, []):
                character_id = int(character["character_id"])
                if character_id in seen:
                    continue
                seen.add(character_id)
                characters.append(character)
        major_lane = sv1_profile(
            profile,
            characters,
            descriptor_to_basis_index,
            projection,
            basis_labels,
            original_descriptor_labels,
            display_descriptor_indices,
            descriptor_similarity,
            descriptor_thresholds,
            old_character_display,
            args.row_weight,
            args.shared_role_weight,
            center_vector,
            effective_profile_center,
            args.character_space,
        )
        if not major_lane:
            continue
        profile_slug = f"{slug(profile.get('name') or str(profile.get('seiyuu_id') or 'profile'))}.json"
        full_profile = {
            "seiyuu_id": profile.get("seiyuu_id"),
            "name": profile.get("name") or "",
            "native_name": profile.get("native_name") or "",
            "image": profile.get("image") or "",
            "site_url": profile.get("site_url") or "",
            "role_count": profile.get("role_count") or 0,
            "character_count": len(major_lane["characters"]),
            "first_year": profile.get("first_year"),
            "aliases": sorted(alias_keys(profile)),
            "favourite_metadata": favourite_metadata,
            "major_lane": major_lane,
        }
        write_json(args.profile_dir / profile_slug, full_profile)
        profiles.append({**full_profile, "profile_path": f"profile_payloads/{profile_slug}"})

    profiles.sort(key=profile_sort_score, reverse=True)
    profile_index = [
        {
            "seiyuu_id": profile["seiyuu_id"],
            "name": profile["name"],
            "native_name": profile["native_name"],
            "image": profile["image"],
            "site_url": profile["site_url"],
            "role_count": profile["role_count"],
            "character_count": profile["character_count"],
            "first_year": profile["first_year"],
            "aliases": profile["aliases"],
            "profile_path": profile["profile_path"],
            "range_metrics": profile["major_lane"]["range_metrics"],
        }
        for profile in profiles
    ]
    samples = [
        {
            "name": profile["name"],
            "native_name": profile["native_name"],
            "image": profile["image"],
            "role_count": profile["role_count"],
            "character_count": profile["character_count"],
            "range_metrics": profile["major_lane"]["range_metrics"],
        }
        for profile in profiles[: args.sample_count]
    ]
    write_json(
        args.output,
        {
            "generated_at": utc_now(),
            "source": "build_svd_site.py",
            "parameters": {
                "tags_input": str(args.tags_input),
                "basis_json": str(args.basis_json),
                "basis_npz": str(args.basis_npz),
                "basis_centering": basis_centering,
                "glosses_json": str(args.glosses_json),
                "glosses_npz": str(args.glosses_npz),
                "profile_dir": str(args.profile_dir),
                "descriptor_scope": args.descriptor_scope,
                "descriptor_input": descriptor_scope_description(args.descriptor_scope),
                "model": "B @ G_P @ K^(-1/2), with G_P projecting all descriptor embeddings into the 384-descriptor Cholesky pivot basis; per-seiyuu profile components are combined with covariance range diagnostics",
                "row_weight": args.row_weight,
                "shared_role_weight": args.shared_role_weight,
                "max_role_edge_count": args.max_role_edge_count,
                "excluded_shared_character_count": len(excluded_shared_characters),
                "excluded_shared_characters": excluded_shared_characters[:25],
                "profile_center": effective_profile_center,
                "character_space": args.character_space,
                "character_favourites": favourite_metadata,
                "display_descriptors": "basis descriptors and supported character descriptors projected from the dominant role/personality profile",
                "display_descriptor_support": (
                    "human-facing rows are per-seiyuu semantic clusters of original descriptors; descriptor "
                    "evidence is gated by mutual local 99th-percentile embedding similarity thresholds, each "
                    "character's display contribution is conserved, singleton-heavy clusters are penalized, "
                    "and clusters are reported until 95% of display descriptor mass is covered"
                ),
                "percent": "squared unit-vector amplitude times 100",
            },
            "counts": {
                "profiles": len(profiles),
                "basis_size": len(basis_labels),
                "descriptor_rows": len(all_descriptors),
                "display_descriptor_rows": len(display_descriptor_indices),
                "source_characters": len(tag_payload.get("characters", [])),
                "included_source_characters": len(source_characters),
                "excluded_shared_characters": len(excluded_shared_characters),
            },
            "samples": samples,
            "profiles": profile_index,
        },
    )
    print(f"wrote {args.output}")
    print(json.dumps({"profiles": len(profiles), "samples": [sample["name"] for sample in samples[:5]]}, indent=2))


if __name__ == "__main__":
    main()
