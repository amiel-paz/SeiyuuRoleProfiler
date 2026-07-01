#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the proven A100 batched Transformers tagging bundle.")
    parser.add_argument("--bundle-dir", type=Path, default=Path("run/a100_batch_transformers_tagging/bundle"))
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("run/a100_batch_transformers_tagging/seiyuu_a100_batch_transformers_bundle.tar.gz"),
    )
    parser.add_argument("--pending", type=Path, default=Path("run/gpu_llm_tagging/pending_safe_enrichment.jsonl"))
    parser.add_argument("--pending-manifest", type=Path, default=Path("run/gpu_llm_tagging/pending_manifest.json"))
    parser.add_argument("--include-errors", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=640)
    parser.add_argument("--max-tags-per-category", type=int, default=3)
    parser.add_argument("--max-source-chars", type=int, default=5000)
    parser.add_argument("--smoke-rows", type=int, default=128)
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
        sys.executable,
        "scripts/export_safe_enrichment_llm_pending.py",
        "--output",
        str(args.pending),
        "--manifest",
        str(args.pending_manifest),
        "--max-source-chars",
        str(args.max_source_chars),
    ]
    if args.include_errors:
        command.append("--include-errors")
    if args.limit > 0:
        command.extend(["--limit", str(args.limit)])
    subprocess.run(command, check=True)


def remote_run_script(args: argparse.Namespace) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MODEL="${{MODEL:-{args.model}}}"
BATCH_SIZE="${{BATCH_SIZE:-{args.batch_size}}}"
MAX_NEW_TOKENS="${{MAX_NEW_TOKENS:-{args.max_new_tokens}}}"
MAX_TAGS_PER_CATEGORY="${{MAX_TAGS_PER_CATEGORY:-{args.max_tags_per_category}}}"
MAX_SOURCE_CHARS="${{MAX_SOURCE_CHARS:-{args.max_source_chars}}}"
SMOKE_ROWS="${{SMOKE_ROWS:-{args.smoke_rows}}}"
RUN_FULL="${{RUN_FULL:-0}}"
AUTO_FULL_AFTER_SMOKE="${{AUTO_FULL_AFTER_SMOKE:-0}}"
CACHE_ROOT="${{CACHE_ROOT:-/dev/shm/seiyuu_transformers_cache}}"

mkdir -p output/artifacts output/batch_transformers_prod output/batch_transformers_smoke logs "$CACHE_ROOT"/hf "$CACHE_ROOT"/pip "$CACHE_ROOT"/tmp
export HF_HOME="$CACHE_ROOT/hf"
export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/hf/hub"
export HF_HUB_DISABLE_XET=1
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export TMPDIR="$CACHE_ROOT/tmp"

echo "== preflight =="
date -u +%Y-%m-%dT%H:%M:%SZ
df -h / /dev/shm 2>/dev/null || true
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
python3 - <<'PY'
import sys
try:
    import torch
except Exception as exc:
    raise SystemExit(f"PyTorch is required in the pod image: {{exc}}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch. Use a CUDA/PyTorch A100 image.")
print("torch", torch.__version__, "cuda", torch.version.cuda)
PY

python3 - <<'PY' >/dev/null 2>&1 || python3 -m pip install --upgrade "transformers==5.12.1" accelerate safetensors sentencepiece
import transformers, accelerate, safetensors, sentencepiece
PY

run_tagger() {{
  local name="$1"
  local limit_args=()
  local force_args=()
  if [ "$name" = "smoke" ]; then
    limit_args=(--limit "$SMOKE_ROWS")
    force_args=(--force --refresh-raw)
  fi
  python3 -u scripts/watch_safe_enrichment_transformers_batch.py \\
    --input data/pending_safe_enrichment.jsonl \\
    --output "output/batch_transformers_${{name}}/character_tags.jsonl" \\
    --errors "output/batch_transformers_${{name}}/errors.jsonl" \\
    --raw-cache-dir "output/batch_transformers_${{name}}/raw" \\
    --status "output/batch_transformers_${{name}}/status.json" \\
    --model "$MODEL" \\
    --cache-dir "$CACHE_ROOT/hf" \\
    --batch-size "$BATCH_SIZE" \\
    --max-new-tokens "$MAX_NEW_TOKENS" \\
    --max-tags-per-category "$MAX_TAGS_PER_CATEGORY" \\
    --max-source-chars "$MAX_SOURCE_CHARS" \\
    --compact-prompt \\
    "${{limit_args[@]}}" \\
    "${{force_args[@]}}"
}}

checkpoint_loop() {{
  while true; do
    tar -czf output/artifacts/batch_transformers_prod_latest.tar.gz -C output batch_transformers_prod 2>/dev/null || true
    date -u +%Y-%m-%dT%H:%M:%SZ > output/artifacts/last_checkpoint.txt
    wc -l output/batch_transformers_prod/character_tags.jsonl output/batch_transformers_prod/errors.jsonl \\
      > output/artifacts/line_counts.txt 2>/dev/null || true
    sleep 180
  done
}}

echo "== smoke =="
run_tagger smoke 2>&1 | tee logs/smoke.log
tar -czf output/artifacts/batch_transformers_smoke_results.tar.gz -C output batch_transformers_smoke

if [ "$RUN_FULL" != "1" ] && [ "$AUTO_FULL_AFTER_SMOKE" != "1" ]; then
  echo "Smoke complete. Set RUN_FULL=1 or AUTO_FULL_AFTER_SMOKE=1 for production."
  exit 0
fi

echo "== production =="
checkpoint_loop &
checkpoint_pid=$!
trap 'kill "$checkpoint_pid" 2>/dev/null || true' EXIT

run_tagger prod 2>&1 | tee logs/prod.log
kill "$checkpoint_pid" 2>/dev/null || true
trap - EXIT

tar -czf output/artifacts/batch_transformers_prod_results.tar.gz -C output batch_transformers_prod
echo "Done. Pull output/artifacts/batch_transformers_prod_results.tar.gz"
"""


def readme(args: argparse.Namespace) -> str:
    return f"""# A100 Batched Transformers Tagging Bundle

This is the reproducible path that succeeded on the A100 pod.

## Proven settings

- Model: `{args.model}`
- Runtime: one CUDA Transformers process, not vLLM/Ollama
- Batch size: `{args.batch_size}`
- Max new tokens: `{args.max_new_tokens}`
- Tags per category: `{args.max_tags_per_category}`
- Prompt mode: compact JSON extraction
- Observed A100-SXM4 80GB run: 2,511 rows in 1,656s, about 90.9 rows/minute

## Run on a new A100 pod

Use a CUDA/PyTorch image if possible. The script will install lightweight dependencies
(`transformers`, `accelerate`, `safetensors`, `sentencepiece`) if missing, but it will
not install PyTorch/CUDA.

```bash
tar -xzf seiyuu_a100_batch_transformers_bundle.tar.gz
cd seiyuu_a100_batch_transformers_bundle
AUTO_FULL_AFTER_SMOKE=1 bash run_a100_batch_tagging.sh
```

For smoke only:

```bash
bash run_a100_batch_tagging.sh
```

For production after a smoke:

```bash
RUN_FULL=1 bash run_a100_batch_tagging.sh
```

## Pull results

```bash
scp -P <port> -i ~/.ssh/id_ed25519 \\
  root@<host>:/root/seiyuu_a100_batch_transformers_bundle/output/artifacts/batch_transformers_prod_results.tar.gz \\
  run/gpu_llm_tagging/
```

The runner writes append-only JSONL outputs and periodic checkpoint archives under
`output/artifacts/`, so interrupted runs can be recovered from the pod filesystem.
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
    for script_name in ("watch_safe_enrichment_llm_tags.py", "watch_safe_enrichment_transformers_batch.py"):
        shutil.copy2(f"scripts/{script_name}", args.bundle_dir / "scripts" / script_name)

    write_text(args.bundle_dir / "run_a100_batch_tagging.sh", remote_run_script(args))
    (args.bundle_dir / "run_a100_batch_tagging.sh").chmod(0o755)
    write_text(args.bundle_dir / "README_A100_BATCH_TAGGING.md", readme(args))
    write_json(
        args.bundle_dir / "bundle_manifest.json",
        {
            "generated_at": utc_now(),
            "model": args.model,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "max_tags_per_category": args.max_tags_per_category,
            "max_source_chars": args.max_source_chars,
            "smoke_rows": args.smoke_rows,
            "pending_manifest": json.loads(args.pending_manifest.read_text(encoding="utf-8")),
        },
    )

    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive, "w:gz") as archive:
        archive.add(args.bundle_dir, arcname="seiyuu_a100_batch_transformers_bundle")
    print(f"wrote {args.bundle_dir}")
    print(f"wrote {args.archive}")


if __name__ == "__main__":
    main()
