#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from role_edge_exclusions import DEFAULT_EXCLUSIONS_PATH, filter_excluded_role_edges, load_role_edge_exclusions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a targeted full-career role cache into the existing compact role cache."
    )
    parser.add_argument("--base", type=Path, default=Path("data/role_edges.json"))
    parser.add_argument("--targeted", type=Path, default=Path("data/role_edges_left_censored_full.json"))
    parser.add_argument("--output", type=Path, default=Path("data/role_edges_current_seiyuu_expanded.json"))
    parser.add_argument("--target-staff-ids", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS_PATH)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_ids(path: Path) -> set[int]:
    return {
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def recompute_counts(roles: list[dict]) -> dict[str, int]:
    return {
        "credit_edges": sum(int(role.get("credit_count") or 1) for role in roles),
        "seiyuu_character_pairs": len(roles),
        "seiyuu": len({int(role["seiyuu"]["seiyuu_id"]) for role in roles}),
        "characters": len({int(role["character"]["character_id"]) for role in roles}),
    }


def main() -> None:
    args = parse_args()
    base = read_json(args.base)
    targeted = read_json(args.targeted)
    target_ids = load_ids(args.target_staff_ids)

    base_roles = base.get("roles") or []
    targeted_roles = targeted.get("roles") or []
    kept_base = [
        role
        for role in base_roles
        if int((role.get("seiyuu") or {}).get("seiyuu_id") or 0) not in target_ids
    ]
    replacement = [
        role
        for role in targeted_roles
        if int((role.get("seiyuu") or {}).get("seiyuu_id") or 0) in target_ids
    ]
    exclusions = load_role_edge_exclusions(args.exclusions)
    roles, removed = filter_excluded_role_edges(kept_base + replacement, exclusions)
    roles = sorted(roles, key=lambda row: (row["seiyuu"]["name"], row["character"]["name"]))

    output = {
        **base,
        "generated_at": utc_now(),
        "source": "merge_role_edges_with_targeted_full.py",
        "parameters": {
            "base": str(args.base),
            "targeted": str(args.targeted),
            "target_staff_ids": str(args.target_staff_ids),
            "exclusions": str(args.exclusions),
            "merge_rule": "replace all base rows for target staff ids with full-career targeted rows",
        },
        "counts": recompute_counts(roles),
        "roles": roles,
    }
    output["counts"]["target_staff_ids"] = len(target_ids)
    output["counts"]["base_roles_removed"] = len(base_roles) - len(kept_base)
    output["counts"]["targeted_roles_added"] = len(replacement)
    output["counts"]["role_edges_excluded"] = len(removed)
    output["role_edge_exclusions"] = [
        {
            "seiyuu_id": item["role"]["seiyuu"]["seiyuu_id"],
            "seiyuu_name": item["role"]["seiyuu"]["name"],
            "character_id": item["role"]["character"]["character_id"],
            "character_name": item["role"]["character"]["name"],
            "anime_ids": [anime.get("anime_id") for anime in item["role"].get("anime") or []],
            "reason": item["exclusion"].get("reason") or "",
        }
        for item in removed
    ]
    write_json(args.output, output)
    print(json.dumps(output["counts"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
