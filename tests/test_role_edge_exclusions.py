from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from role_edge_exclusions import filter_excluded_role_edges, role_matches_exclusion  # noqa: E402


def test_fate_prototype_artoria_collision_is_excluded() -> None:
    role = {
        "seiyuu": {"seiyuu_id": 95079, "name": "Takahiro Sakurai"},
        "character": {"character_id": 497, "name": "Artoria Pendragon"},
        "anime": [{"anime_id": 12565, "title": "Fate/Prototype"}],
    }
    exclusion = {
        "seiyuu_id": 95079,
        "character_id": 497,
        "anime_id": 12565,
        "reason": "AniList Prototype Arthur collision",
    }

    assert role_matches_exclusion(role, exclusion)
    kept, removed = filter_excluded_role_edges([role], [exclusion])
    assert kept == []
    assert removed[0]["role"] == role


def test_same_character_other_seiyuu_is_not_excluded() -> None:
    role = {
        "seiyuu": {"seiyuu_id": 95035, "name": "Ayako Kawasumi"},
        "character": {"character_id": 497, "name": "Artoria Pendragon"},
        "anime": [{"anime_id": 356, "title": "Fate/stay night"}],
    }
    exclusion = {"seiyuu_id": 95079, "character_id": 497, "anime_id": 12565}

    assert not role_matches_exclusion(role, exclusion)
    kept, removed = filter_excluded_role_edges([role], [exclusion])
    assert kept == [role]
    assert removed == []
