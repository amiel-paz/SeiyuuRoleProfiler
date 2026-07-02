#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from export_safe_enrichment_llm_pending import main as export_pending_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a minimal GPU-worker bundle for safe-enrichment LLM tagging.")
    parser.add_argument("--bundle-dir", type=Path, default=Path("run/gpu_llm_tagging/bundle"))
    parser.add_argument("--archive", type=Path, default=Path("run/gpu_llm_tagging/seiyuu_gpu_llm_tagging_bundle.tar.gz"))
    parser.add_argument("--pending", type=Path, default=Path("run/gpu_llm_tagging/pending_safe_enrichment.jsonl"))
    parser.add_argument("--pending-manifest", type=Path, default=Path("run/gpu_llm_tagging/pending_manifest.json"))
    parser.add_argument("--include-errors", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ollama-model", default="qwen3.5:4b")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--think", choices=("false", "true", "low", "medium", "high", "max"), default="false")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_export(args: argparse.Namespace) -> None:
    command = [
        ".venv/bin/python",
        "scripts/export_safe_enrichment_llm_pending.py",
        "--output",
        str(args.pending),
        "--manifest",
        str(args.pending_manifest),
    ]
    if args.include_errors:
        command.append("--include-errors")
    if args.limit > 0:
        command.extend(["--limit", str(args.limit)])
    subprocess.run(command, check=True)


def remote_run_script(model: str, timeout: float, think: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p output/raw logs

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is not installed. Install it first, then rerun this script." >&2
  echo "Common Linux install: curl -fsSL https://ollama.com/install.sh | sh" >&2
  exit 2
fi

if ! pgrep -x ollama >/dev/null 2>&1; then
  nohup ollama serve > logs/ollama.log 2>&1 &
  sleep 5
fi

ollama pull {model}

python3 scripts/watch_safe_enrichment_llm_tags.py \\
  --input data/pending_safe_enrichment.jsonl \\
  --output output/character_tags.jsonl \\
  --errors output/errors.jsonl \\
  --raw-cache-dir output/raw \\
  --status output/status.json \\
  --ollama-model {model} \\
  --timeout {timeout} \\
  --think {think} \\
  --once \\
  2>&1 | tee logs/tagger.log

tar -czf output/results.tar.gz -C output character_tags.jsonl errors.jsonl status.json raw
echo "Done. Retrieve output/results.tar.gz"
"""


def readme(model: str) -> str:
    return f"""# GPU LLM Tagging Bundle

This bundle contains only the untagged safe-enrichment rows plus the local tagging runner.

## Remote setup

On an L4/A40 machine:

```bash
tar -xzf seiyuu_gpu_llm_tagging_bundle.tar.gz
cd seiyuu_gpu_llm_tagging_bundle
bash run_remote_tagging.sh
```

The script expects `ollama` to be installed and will run:

```bash
ollama pull {model}
python3 scripts/watch_safe_enrichment_llm_tags.py --once ...
```

Results are written to:

```text
output/results.tar.gz
```

## Pull results back locally

After copying `results.tar.gz` back to the repo, unpack it somewhere like:

```bash
mkdir -p run/gpu_llm_tagging/returned
tar -xzf results.tar.gz -C run/gpu_llm_tagging/returned
```

Then merge:

```bash
.venv/bin/python scripts/merge_safe_enrichment_llm_results.py \\
  --remote-tags run/gpu_llm_tagging/returned/character_tags.jsonl \\
  --remote-errors run/gpu_llm_tagging/returned/errors.jsonl \\
  --remote-raw-dir run/gpu_llm_tagging/returned/raw \\
  --replace-errors-with-success
```
"""


def main() -> None:
    args = parse_args()
    run_export(args)

    if args.bundle_dir.exists():
        shutil.rmtree(args.bundle_dir)
    (args.bundle_dir / "scripts").mkdir(parents=True)
    (args.bundle_dir / "data").mkdir(parents=True)

    shutil.copy2(args.pending, args.bundle_dir / "data" / "pending_safe_enrichment.jsonl")
    shutil.copy2(args.pending_manifest, args.bundle_dir / "pending_manifest.json")
    shutil.copy2("scripts/watch_safe_enrichment_llm_tags.py", args.bundle_dir / "scripts" / "watch_safe_enrichment_llm_tags.py")
    write_text(args.bundle_dir / "run_remote_tagging.sh", remote_run_script(args.ollama_model, args.timeout, args.think))
    (args.bundle_dir / "run_remote_tagging.sh").chmod(0o755)
    write_text(args.bundle_dir / "README_GPU_TAGGING.md", readme(args.ollama_model))
    write_json(
        args.bundle_dir / "bundle_manifest.json",
        {
            "generated_at": utc_now(),
            "ollama_model": args.ollama_model,
            "think": args.think,
            "timeout": args.timeout,
            "pending_manifest": json.loads(args.pending_manifest.read_text(encoding="utf-8")),
        },
    )

    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive, "w:gz") as archive:
        archive.add(args.bundle_dir, arcname="seiyuu_gpu_llm_tagging_bundle")
    print(f"wrote {args.bundle_dir}")
    print(f"wrote {args.archive}")


if __name__ == "__main__":
    main()
