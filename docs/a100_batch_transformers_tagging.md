# A100 Batch Transformers Tagging

This is the cached, reproducible path for fast safe-enrichment LLM tagging on an A100 pod.

## Why This Path

The successful run used one CUDA Transformers process with real batching. The failed/slow paths were:

- vLLM installed live on the paid pod, which hit environment and install friction.
- multiple batch-size-1 Transformers workers, which kept the GPU busy but decoded slowly.

## Proven Configuration

- GPU: A100-SXM4 80GB
- Model: `Qwen/Qwen3.5-4B`
- Runtime: `transformers` + CUDA/PyTorch
- Batch size: `64`
- Max new tokens: `640`
- Max tags per category: `3`
- Prompt: compact JSON extraction
- Observed run: `2,511` rows in `1,656` seconds, about `90.9` rows/minute

The runner includes a small parser repair for unescaped quote characters inside evidence spans.

## Build The Bundle

From the repo root:

```bash
python3 scripts/prepare_a100_batch_transformers_bundle.py
```

This writes:

```text
run/a100_batch_transformers_tagging/seiyuu_a100_batch_transformers_bundle.tar.gz
```

`run/` is intentionally gitignored; rebuild the bundle from source when needed.

## Run On A New A100 Pod

Use a pod image that already has CUDA PyTorch working. The script installs lightweight Python packages if missing, but it does not install PyTorch/CUDA.

```bash
scp -P <port> -i ~/.ssh/id_ed25519 \
  run/a100_batch_transformers_tagging/seiyuu_a100_batch_transformers_bundle.tar.gz \
  root@<host>:/root/

ssh -p <port> -i ~/.ssh/id_ed25519 root@<host>
tar -xzf seiyuu_a100_batch_transformers_bundle.tar.gz
cd seiyuu_a100_batch_transformers_bundle
AUTO_FULL_AFTER_SMOKE=1 bash run_a100_batch_tagging.sh
```

Smoke only:

```bash
bash run_a100_batch_tagging.sh
```

Production after smoke:

```bash
RUN_FULL=1 bash run_a100_batch_tagging.sh
```

## Monitor

```bash
cat output/batch_transformers_prod/status.json
wc -l output/batch_transformers_prod/character_tags.jsonl output/batch_transformers_prod/errors.jsonl
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader
tail -n 20 logs/prod.log
```

## Pull Results

```bash
scp -P <port> -i ~/.ssh/id_ed25519 \
  root@<host>:/root/seiyuu_a100_batch_transformers_bundle/output/artifacts/batch_transformers_prod_results.tar.gz \
  run/gpu_llm_tagging/
```

The production runner also writes periodic checkpoint archives under `output/artifacts/`.
