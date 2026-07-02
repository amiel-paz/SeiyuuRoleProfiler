#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   REMOTE='root@HOST' PORT='22' KEY="$HOME/.ssh/IDENTITY_FILE" bash scripts/remote_vllm_tagging_commands.sh upload
#   REMOTE='root@HOST' PORT='22' KEY="$HOME/.ssh/IDENTITY_FILE" bash scripts/remote_vllm_tagging_commands.sh start
#   REMOTE='root@HOST' PORT='22' KEY="$HOME/.ssh/IDENTITY_FILE" RUN_FULL=1 bash scripts/remote_vllm_tagging_commands.sh start
#   REMOTE='root@HOST' PORT='22' KEY="$HOME/.ssh/IDENTITY_FILE" bash scripts/remote_vllm_tagging_commands.sh status
#   REMOTE='root@HOST' PORT='22' KEY="$HOME/.ssh/IDENTITY_FILE" bash scripts/remote_vllm_tagging_commands.sh pull

REMOTE="${REMOTE:?Set REMOTE, e.g. root@1.2.3.4}"
PORT="${PORT:-22}"
KEY="${KEY:-$HOME/.ssh/IDENTITY_FILE}"
WORKDIR="${WORKDIR:-/root/seiyuu_vllm_work}"
ARCHIVE="${ARCHIVE:-run/vllm_tagging/seiyuu_vllm_tagging_bundle.tar.gz}"
LOCAL_RETURN="${LOCAL_RETURN:-run/vllm_tagging/returned}"

ssh_remote() {
  ssh -p "$PORT" -i "$KEY" -o StrictHostKeyChecking=accept-new "$REMOTE" "$@"
}

case "${1:-}" in
  upload)
    scp -P "$PORT" -i "$KEY" "$ARCHIVE" "$REMOTE:/root/seiyuu_vllm_tagging_bundle.tar.gz"
    ssh_remote "rm -rf '$WORKDIR' && mkdir -p '$WORKDIR' && tar -xzf /root/seiyuu_vllm_tagging_bundle.tar.gz -C '$WORKDIR' --strip-components=1 && ls -lah '$WORKDIR'"
    ;;
  start)
    RUN_FULL_REMOTE="${RUN_FULL:-0}"
    CONCURRENCY_REMOTE="${CONCURRENCY:-}"
    SMOKE_ROWS_REMOTE="${SMOKE_ROWS:-}"
    MIN_ROOT_FREE_GB_REMOTE="${MIN_ROOT_FREE_GB:-}"
    CACHE_ROOT_REMOTE="${CACHE_ROOT:-}"
    AUTO_FULL_AFTER_SMOKE_REMOTE="${AUTO_FULL_AFTER_SMOKE:-}"
    ssh_remote "cd '$WORKDIR' && mkdir -p logs && (RUN_FULL='$RUN_FULL_REMOTE' AUTO_FULL_AFTER_SMOKE='$AUTO_FULL_AFTER_SMOKE_REMOTE' CONCURRENCY='$CONCURRENCY_REMOTE' SMOKE_ROWS='$SMOKE_ROWS_REMOTE' MIN_ROOT_FREE_GB='$MIN_ROOT_FREE_GB_REMOTE' CACHE_ROOT='$CACHE_ROOT_REMOTE' nohup bash run_remote_vllm_tagging.sh > logs/remote_master.log 2>&1 & echo \$! > logs/remote_master.pid && echo started \$(cat logs/remote_master.pid))"
    ;;
  status)
    ssh_remote "cd '$WORKDIR' && echo procs && ps -eo pid,ppid,stat,etime,pcpu,pmem,cmd | grep -E 'vllm|watch_safe|run_remote' | grep -v grep || true; echo counts && wc -l output/character_tags.jsonl output/errors.jsonl output/smoke_character_tags.jsonl output/smoke_errors.jsonl 2>/dev/null || true; echo checkpoint && cat checkpoints/last_checkpoint.txt 2>/dev/null || true; cat checkpoints/line_counts.txt 2>/dev/null || true; echo smoke && tail -n 20 logs/smoke_tagger.log 2>/dev/null || true; echo recent && tail -n 20 logs/tagger.log 2>/dev/null || true; echo master && tail -n 40 logs/remote_master.log 2>/dev/null || true; echo vllm && tail -n 80 logs/vllm.log 2>/dev/null || true; echo gpu && nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader || true"
    ;;
  pull)
    mkdir -p "$LOCAL_RETURN"
    ssh_remote "cd '$WORKDIR' && if [ -f checkpoints/results_latest.tar.gz ]; then echo checkpoints/results_latest.tar.gz; elif [ -f checkpoints/smoke_results_latest.tar.gz ]; then echo checkpoints/smoke_results_latest.tar.gz; else exit 1; fi" > run/vllm_tagging/remote_result_path.txt
    REMOTE_RESULT="$(cat run/vllm_tagging/remote_result_path.txt)"
    scp -P "$PORT" -i "$KEY" "$REMOTE:$WORKDIR/$REMOTE_RESULT" run/vllm_tagging/results_latest.tar.gz
    find "$LOCAL_RETURN" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
    tar -xzf run/vllm_tagging/results_latest.tar.gz -C "$LOCAL_RETURN"
    wc -l "$LOCAL_RETURN/character_tags.jsonl" "$LOCAL_RETURN/errors.jsonl" "$LOCAL_RETURN/smoke_character_tags.jsonl" "$LOCAL_RETURN/smoke_errors.jsonl" 2>/dev/null || true
    ;;
  stop)
    ssh_remote "cd '$WORKDIR' 2>/dev/null || exit 0; if [ -f logs/remote_master.pid ]; then kill \\$(cat logs/remote_master.pid) 2>/dev/null || true; fi; if [ -f logs/vllm.pid ]; then kill \\$(cat logs/vllm.pid) 2>/dev/null || true; fi; if [ -f logs/checkpoint.pid ]; then kill \\$(cat logs/checkpoint.pid) 2>/dev/null || true; fi; pkill -f 'watch_safe_enrichment_vllm_tags.py' 2>/dev/null || true; pkill -f 'vllm serve' 2>/dev/null || true"
    ;;
  *)
    echo "Usage: $0 upload|start|status|pull|stop" >&2
    exit 2
    ;;
esac
