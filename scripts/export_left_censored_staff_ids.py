#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export current profiler staff ids whose observed first year equals the old role-cache floor."
    )
    parser.add_argument("--profiles", type=Path, default=Path("site/profiles.json"))
    parser.add_argument("--floor-year", type=int, default=2007)
    parser.add_argument("--output", type=Path, default=Path("run/left_censored_staff_ids.txt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.profiles.read_text(encoding="utf-8"))
    rows = payload.get("profiles") or []
    ids = sorted(
        {
            int(row["seiyuu_id"])
            for row in rows
            if row.get("seiyuu_id") is not None and int(row.get("first_year") or 0) == args.floor_year
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(f"{staff_id}\n" for staff_id in ids), encoding="utf-8")
    print(f"wrote {len(ids)} ids to {args.output}")


if __name__ == "__main__":
    main()
