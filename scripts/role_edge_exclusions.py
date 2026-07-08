from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_EXCLUSIONS_PATH = Path("data/role_edge_exclusions.json")


def load_role_edge_exclusions(path: Path = DEFAULT_EXCLUSIONS_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("exclusions") or [])


def role_matches_exclusion(role: dict[str, Any], exclusion: dict[str, Any]) -> bool:
    seiyuu = role.get("seiyuu") or {}
    character = role.get("character") or {}
    if int(seiyuu.get("seiyuu_id") or 0) != int(exclusion.get("seiyuu_id") or 0):
        return False
    if int(character.get("character_id") or 0) != int(exclusion.get("character_id") or 0):
        return False

    anime_id = exclusion.get("anime_id")
    if anime_id is None:
        return True
    return any(int(anime.get("anime_id") or 0) == int(anime_id) for anime in role.get("anime") or [])


def filter_excluded_role_edges(
    roles: list[dict[str, Any]],
    exclusions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept = []
    removed = []
    for role in roles:
        matched = next((exclusion for exclusion in exclusions if role_matches_exclusion(role, exclusion)), None)
        if matched is None:
            kept.append(role)
        else:
            removed.append({"role": role, "exclusion": matched})
    return kept, removed
