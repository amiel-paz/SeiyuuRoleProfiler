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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a vLLM GPU worker bundle for safe-enrichment LLM tagging.")
    parser.add_argument("--bundle-dir", type=Path, default=Path("run/vllm_tagging/bundle"))
    parser.add_argument("--archive", type=Path, default=Path("run/vllm_tagging/seiyuu_vllm_tagging_bundle.tar.gz"))
    parser.add_argument("--pending", type=Path, default=Path("run/vllm_tagging/pending_safe_enrichment.jsonl"))
    parser.add_argument("--pending-manifest", type=Path, default=Path("run/vllm_tagging/pending_manifest.json"))
    parser.add_argument("--include-errors", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--structured-mode", choices=("none", "json_object", "json_schema"), default="json_object")
    parser.add_argument("--max-source-chars", type=int, default=5000)
    parser.add_argument("--max-tags-per-category", type=int, default=8)
    parser.add_argument("--smoke-rows", type=int, default=5)
    parser.add_argument("--min-root-free-gb", type=int, default=45)
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
mkdir -p output/raw logs checkpoints

MODEL="${{MODEL:-{args.model}}}"
CONCURRENCY="${{CONCURRENCY:-{args.concurrency}}}"
STRUCTURED_MODE="${{STRUCTURED_MODE:-{args.structured_mode}}}"
MAX_MODEL_LEN="${{MAX_MODEL_LEN:-8192}}"
PORT="${{PORT:-8000}}"
SMOKE_ROWS="${{SMOKE_ROWS:-{args.smoke_rows}}}"
RUN_FULL="${{RUN_FULL:-0}}"
AUTO_FULL_AFTER_SMOKE="${{AUTO_FULL_AFTER_SMOKE:-0}}"
MIN_ROOT_FREE_GB="${{MIN_ROOT_FREE_GB:-{args.min_root_free_gb}}}"
CACHE_ROOT="${{CACHE_ROOT:-}}"

if [ -z "$CACHE_ROOT" ]; then
  if [ -d /dev/shm ] && [ "$(df -Pk /dev/shm | awk 'NR==2 {{printf "%d", $4/1024/1024}}')" -ge 30 ]; then
    CACHE_ROOT="/dev/shm/seiyuu_vllm_cache"
  elif [ -d /workspace ]; then
    CACHE_ROOT="/workspace/seiyuu_vllm_cache"
  else
    CACHE_ROOT="$PWD/cache"
  fi
fi
mkdir -p "$CACHE_ROOT"/hf "$CACHE_ROOT"/vllm "$CACHE_ROOT"/tmp "$CACHE_ROOT"/pycache
export HF_HOME="$CACHE_ROOT/hf"
export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/hf/hub"
export HF_HUB_DISABLE_XET=1
export VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"
export TMPDIR="$CACHE_ROOT/tmp"
export PYTHONPYCACHEPREFIX="$CACHE_ROOT/pycache"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"

echo "== preflight =="
date -u +%Y-%m-%dT%H:%M:%SZ
df -h / /workspace /dev/shm 2>/dev/null || true
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "cache root: $CACHE_ROOT"
root_free_gb="$(df -Pk / | awk 'NR==2 {{printf "%d", $4/1024/1024}}')"
if [ "$MIN_ROOT_FREE_GB" -gt 0 ] && [ "$root_free_gb" -lt "$MIN_ROOT_FREE_GB" ]; then
  echo "Refusing to install/run: root filesystem has ${{root_free_gb}}GB free, need at least ${{MIN_ROOT_FREE_GB}}GB." >&2
  echo "Use a pod/container disk of ~80GB, or a vLLM image with enough persistent model cache." >&2
  exit 78
fi

if python3 -c "import vllm" >/dev/null 2>&1; then
  echo "Using system Python: vLLM is already importable."
  PYTHON="python3"
else
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip wheel setuptools
  if python -c "import vllm" >/dev/null 2>&1; then
    echo "vLLM already importable in bundle venv."
  else
    echo "Installing vLLM. This is the slow part; prefer a vLLM image when available."
    python -m pip install --pre --upgrade vllm --extra-index-url https://wheels.vllm.ai/nightly 2>&1 | tee logs/pip_vllm_install.log
  fi
  PYTHON="python"
fi

if ! pgrep -f "vllm serve .*${{MODEL}}" >/dev/null 2>&1; then
  nohup "$PYTHON" -m vllm.entrypoints.cli.main serve "${{MODEL}}" \\
    --host 127.0.0.1 \\
    --port "${{PORT}}" \\
    --tensor-parallel-size 1 \\
    --max-model-len "${{MAX_MODEL_LEN}}" \\
    --dtype bfloat16 \\
    --reasoning-parser qwen3 \\
    --language-model-only \\
    > logs/vllm.log 2>&1 &
  echo $! > logs/vllm.pid
fi

echo "Waiting for vLLM at http://127.0.0.1:${{PORT}}/v1 ..."
for i in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${{PORT}}/v1/models" >/dev/null 2>&1; then
    break
  fi
  sleep 5
  if ! kill -0 "$(cat logs/vllm.pid)" 2>/dev/null; then
    echo "vLLM process exited before readiness." >&2
    tail -n 240 logs/vllm.log >&2 || true
    exit 1
  fi
done
if ! curl -fsS "http://127.0.0.1:${{PORT}}/v1/models" > logs/vllm_models.json; then
  echo "vLLM did not become ready." >&2
  tail -n 240 logs/vllm.log >&2 || true
  exit 1
fi

checkpoint_loop() {{
  while true; do
    tar -czf checkpoints/results_latest.tar.gz -C output . 2>/dev/null || true
    date -u +%Y-%m-%dT%H:%M:%SZ > checkpoints/last_checkpoint.txt
    wc -l output/character_tags.jsonl output/errors.jsonl 2>/dev/null > checkpoints/line_counts.txt || true
    sleep 60
  done
}}
checkpoint_loop &
echo $! > logs/checkpoint.pid

if [ "$RUN_FULL" != "1" ]; then
  echo "RUN_FULL is not 1; running smoke test only (${{SMOKE_ROWS}} rows)."
  "$PYTHON" -u scripts/watch_safe_enrichment_vllm_tags.py \\
    --input data/pending_safe_enrichment.jsonl \\
    --output output/smoke_character_tags.jsonl \\
    --errors output/smoke_errors.jsonl \\
    --raw-cache-dir output/smoke_raw \\
    --status output/smoke_status.json \\
    --vllm-url "http://127.0.0.1:${{PORT}}/v1" \\
    --model "${{MODEL}}" \\
    --concurrency 2 \\
    --max-rows-per-pass "${{SMOKE_ROWS}}" \\
    --timeout {args.timeout} \\
    --structured-mode "${{STRUCTURED_MODE}}" \\
    --max-tags-per-category {args.max_tags_per_category} \\
    --max-source-chars {args.max_source_chars} \\
    --force \\
    --refresh-raw \\
    --once 2>&1 | tee -a logs/smoke_tagger.log
  tar -czf checkpoints/smoke_results_latest.tar.gz -C output smoke_character_tags.jsonl smoke_errors.jsonl smoke_status.json smoke_raw 2>/dev/null || true
  if [ "$AUTO_FULL_AFTER_SMOKE" != "1" ]; then
    if [ -f logs/checkpoint.pid ]; then kill "$(cat logs/checkpoint.pid)" 2>/dev/null || true; fi
    echo "Smoke done. If it looks good, rerun with RUN_FULL=1."
    exit 0
  fi
  echo "Smoke succeeded; AUTO_FULL_AFTER_SMOKE=1, continuing to full queue."
fi

set +e
"$PYTHON" -u scripts/watch_safe_enrichment_vllm_tags.py \\
  --input data/pending_safe_enrichment.jsonl \\
  --output output/character_tags.jsonl \\
  --errors output/errors.jsonl \\
  --raw-cache-dir output/raw \\
  --status output/status.json \\
  --vllm-url "http://127.0.0.1:${{PORT}}/v1" \\
  --model "${{MODEL}}" \\
  --concurrency "${{CONCURRENCY}}" \\
  --timeout {args.timeout} \\
  --structured-mode "${{STRUCTURED_MODE}}" \\
  --max-tags-per-category {args.max_tags_per_category} \\
  --max-source-chars {args.max_source_chars} \\
  --once 2>&1 | tee -a logs/tagger.log
status=${{PIPESTATUS[0]}}
set -e

tar -czf checkpoints/results_latest.tar.gz -C output . 2>/dev/null || true
tar -czf output/results.tar.gz -C output character_tags.jsonl errors.jsonl status.json raw 2>/dev/null || true
if [ -f logs/checkpoint.pid ]; then kill "$(cat logs/checkpoint.pid)" 2>/dev/null || true; fi
echo "Done. Retrieve checkpoints/results_latest.tar.gz or output/results.tar.gz"
exit "$status"
"""


def smoke_script(args: argparse.Namespace) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ -x .venv/bin/python ]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="python3"
fi
"$PYTHON" -u scripts/watch_safe_enrichment_vllm_tags.py \\
  --input data/pending_safe_enrichment.jsonl \\
  --output output/smoke_character_tags.jsonl \\
  --errors output/smoke_errors.jsonl \\
  --raw-cache-dir output/smoke_raw \\
  --status output/smoke_status.json \\
  --vllm-url "http://127.0.0.1:${{PORT:-8000}}/v1" \\
  --model "${{MODEL:-{args.model}}}" \\
  --concurrency 2 \\
  --max-rows-per-pass {args.smoke_rows} \\
  --timeout {args.timeout} \\
  --structured-mode "${{STRUCTURED_MODE:-{args.structured_mode}}}" \\
  --max-tags-per-category {args.max_tags_per_category} \\
  --max-source-chars {args.max_source_chars} \\
  --force \\
  --refresh-raw \\
  --once
"""


def readme(args: argparse.Namespace) -> str:
    return f"""# vLLM Safe-Enrichment Tagging Bundle

This bundle is intended for a GPU worker. It keeps the current tagging prompt/schema shape, but replaces serial Ollama calls with concurrent vLLM chat-completions-compatible requests.

Important: `run_remote_vllm_tagging.sh` runs a **{args.smoke_rows}-row smoke test by default**. It only runs the full queue when `RUN_FULL=1`.

Pinned defaults:

```text
model: {args.model}
temperature: 0
seed: 13
top_p: 0.95
top_k: 20
presence_penalty: 1.5
thinking: disabled through chat_template_kwargs.enable_thinking=false
structured_mode: {args.structured_mode}
concurrency: {args.concurrency}
smoke_rows: {args.smoke_rows}
```

Recommended pod setup:

```text
GPU: L4 or better for smoke testing
container/root disk: 80GB preferred, 50GB minimum-ish
image: vLLM image preferred; otherwise CUDA/PyTorch with enough disk
```

Smoke run:

```bash
tar -xzf seiyuu_vllm_tagging_bundle.tar.gz
cd seiyuu_vllm_tagging_bundle
bash run_remote_vllm_tagging.sh
```

Full run after smoke works:

```bash
RUN_FULL=1 CONCURRENCY=16 bash run_remote_vllm_tagging.sh
```

Useful overrides:

```bash
SMOKE_ROWS=10 CONCURRENCY=16 MAX_MODEL_LEN=8192 STRUCTURED_MODE=json_object bash run_remote_vllm_tagging.sh
```

Outputs:

```text
output/character_tags.jsonl
output/errors.jsonl
output/raw/*.json
checkpoints/results_latest.tar.gz
```

Pull and merge locally:

```bash
scp -P <PORT> -i ~/.ssh/IDENTITY_FILE root@<HOST>:/root/seiyuu_vllm_work/checkpoints/results_latest.tar.gz run/vllm_tagging/results_latest.tar.gz
mkdir -p run/vllm_tagging/returned
tar -xzf run/vllm_tagging/results_latest.tar.gz -C run/vllm_tagging/returned
.venv/bin/python scripts/merge_safe_enrichment_llm_results.py \\
  --remote-tags run/vllm_tagging/returned/character_tags.jsonl \\
  --remote-errors run/vllm_tagging/returned/errors.jsonl \\
  --remote-raw-dir run/vllm_tagging/returned/raw \\
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
    for script in (
        "watch_safe_enrichment_llm_tags.py",
        "watch_safe_enrichment_vllm_tags.py",
        "merge_safe_enrichment_llm_results.py",
    ):
        shutil.copy2(Path("scripts") / script, args.bundle_dir / "scripts" / script)
    write_text(args.bundle_dir / "run_remote_vllm_tagging.sh", remote_run_script(args))
    write_text(args.bundle_dir / "run_smoke_vllm_tagging.sh", smoke_script(args))
    (args.bundle_dir / "run_remote_vllm_tagging.sh").chmod(0o755)
    (args.bundle_dir / "run_smoke_vllm_tagging.sh").chmod(0o755)
    write_text(args.bundle_dir / "README_VLLM_TAGGING.md", readme(args))
    write_json(
        args.bundle_dir / "bundle_manifest.json",
        {
            "generated_at": utc_now(),
            "runtime": "vllm",
            "model": args.model,
            "concurrency": args.concurrency,
            "structured_mode": args.structured_mode,
            "timeout": args.timeout,
            "max_source_chars": args.max_source_chars,
            "max_tags_per_category": args.max_tags_per_category,
            "pending_manifest": json.loads(args.pending_manifest.read_text(encoding="utf-8")),
        },
    )

    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive, "w:gz") as archive:
        archive.add(args.bundle_dir, arcname="seiyuu_vllm_tagging_bundle")
    print(f"wrote {args.bundle_dir}")
    print(f"wrote {args.archive}")


if __name__ == "__main__":
    main()
