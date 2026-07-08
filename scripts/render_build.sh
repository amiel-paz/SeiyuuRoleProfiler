#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

required_files=(
  "site/index.html"
  "site/mvp_visualizer.html"
  "site/mvp_visualizer/index.json"
  "site/mvp_visualizer/rankings_unit.json"
  "site/mvp_visualizer/rankings_favorites_weighted.json"
)

for file in "${required_files[@]}"; do
  if [[ ! -s "$file" ]]; then
    echo "missing required static artifact: $file" >&2
    exit 1
  fi
done

unit_profiles="$(find site/mvp_visualizer/profiles/unit -type f -name '*.json' | wc -l | tr -d ' ')"
weighted_profiles="$(find site/mvp_visualizer/profiles/favorites_weighted -type f -name '*.json' | wc -l | tr -d ' ')"

if [[ "$unit_profiles" -eq 0 || "$weighted_profiles" -eq 0 ]]; then
  echo "missing precached profile payloads" >&2
  exit 1
fi

if [[ "$unit_profiles" -ne "$weighted_profiles" ]]; then
  echo "profile payload mismatch: unit=$unit_profiles favorites_weighted=$weighted_profiles" >&2
  exit 1
fi

python3 - <<'PY'
import json
from pathlib import Path

root = Path("site/mvp_visualizer")
index = json.loads((root / "index.json").read_text())
unit = json.loads((root / "rankings_unit.json").read_text())
weighted = json.loads((root / "rankings_favorites_weighted.json").read_text())

if not index.get("profiles"):
    raise SystemExit("index.json has no profiles")
if not unit.get("seiyuu") or not weighted.get("seiyuu"):
    raise SystemExit("ranking payloads are empty")
if unit.get("descriptors") != weighted.get("descriptors"):
    raise SystemExit("ranking descriptor vocabularies differ by mode")

for mode in ("unit", "favorites_weighted"):
    missing = [
        row["profile_path"]
        for row in index["profiles"]
        if not (Path("site") / row["profile_path"].format(mode=mode)).exists()
    ]
    if missing:
        raise SystemExit(f"{mode} profile paths missing: {missing[:3]}")

print(
    "verified static profiler payload:",
    len(index["profiles"]),
    "profiles;",
    len(unit["descriptors"]),
    "descriptors",
)
PY

if command -v node >/dev/null 2>&1; then
  node - <<'NODE'
const fs = require('fs');
const html = fs.readFileSync('site/mvp_visualizer.html', 'utf8');
for (const [i, match] of [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].entries()) {
  new Function(match[1]);
  console.log(`script ${i + 1} parses`);
}
NODE
fi

echo "Render static build verification complete."
