#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

TARGET_STAFF_IDS="${TARGET_STAFF_IDS:-run/left_censored_staff_ids.txt}"
TARGETED_ROLE_OUTPUT="${TARGETED_ROLE_OUTPUT:-data/role_edges_left_censored_full.json}"
ROLE_OUTPUT="${ROLE_OUTPUT:-data/role_edges_current_seiyuu_expanded.json}"
ROLE_RAW_DIR="${ROLE_RAW_DIR:-data/external/anilist_staff_roles}"
SAFE_OUTPUT_DIR="${SAFE_OUTPUT_DIR:-data/external/safe_enrichment}"
SAFE_LLM_OUTPUT="${SAFE_LLM_OUTPUT:-data/external/safe_enrichment_llm/character_tags.jsonl}"
SAFE_LLM_ERRORS="${SAFE_LLM_ERRORS:-data/external/safe_enrichment_llm/errors.jsonl}"
SAFE_LLM_RAW_DIR="${SAFE_LLM_RAW_DIR:-data/external/safe_enrichment_llm/raw}"
SAFE_LLM_STATUS="${SAFE_LLM_STATUS:-data/external/safe_enrichment_llm/status.json}"
ROLE_SLEEP="${ROLE_SLEEP:-1.2}"
SAFE_SLEEP="${SAFE_SLEEP:-0.35}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3.5:4b}"

echo "[descriptor-refresh] $(date -u +%Y-%m-%dT%H:%M:%SZ) export left-censored staff ids"
python3 scripts/export_left_censored_staff_ids.py \
  --profiles site/profiles.json \
  --floor-year 2007 \
  --output "$TARGET_STAFF_IDS"

echo "[descriptor-refresh] $(date -u +%Y-%m-%dT%H:%M:%SZ) targeted full-career role crawl"
python3 scripts/cache_current_seiyuu_full_anilist_roles.py \
  --output "$TARGETED_ROLE_OUTPUT" \
  --raw-dir "$ROLE_RAW_DIR" \
  --staff-ids-file "$TARGET_STAFF_IDS" \
  --sleep "$ROLE_SLEEP" \
  --retries 10 \
  --per-page 25 \
  --sorts START_DATE

echo "[descriptor-refresh] $(date -u +%Y-%m-%dT%H:%M:%SZ) merge targeted roles"
python3 scripts/merge_role_edges_with_targeted_full.py \
  --base data/role_edges.json \
  --targeted "$TARGETED_ROLE_OUTPUT" \
  --target-staff-ids "$TARGET_STAFF_IDS" \
  --output "$ROLE_OUTPUT"

echo "[descriptor-refresh] $(date -u +%Y-%m-%dT%H:%M:%SZ) safe enrichment"
python3 scripts/cache_safe_character_enrichment.py \
  --roles-input "$ROLE_OUTPUT" \
  --output-dir "$SAFE_OUTPUT_DIR" \
  --min-favourites 100 \
  --sleep-seconds "$SAFE_SLEEP"

echo "[descriptor-refresh] $(date -u +%Y-%m-%dT%H:%M:%SZ) LLM tags"
python3 scripts/watch_safe_enrichment_llm_tags.py \
  --input "$SAFE_OUTPUT_DIR/character_safe_enrichment.jsonl" \
  --output "$SAFE_LLM_OUTPUT" \
  --errors "$SAFE_LLM_ERRORS" \
  --raw-cache-dir "$SAFE_LLM_RAW_DIR" \
  --status "$SAFE_LLM_STATUS" \
  --ollama-model "$OLLAMA_MODEL" \
  --temperature 0 \
  --think false \
  --once

echo "[descriptor-refresh] $(date -u +%Y-%m-%dT%H:%M:%SZ) descriptor canonicalization"
python3 scripts/cache_global_descriptor_canonicalization.py

echo "[descriptor-refresh] $(date -u +%Y-%m-%dT%H:%M:%SZ) production personality basis"
python3 scripts/build_production_personality_basis.py

echo "[descriptor-refresh] $(date -u +%Y-%m-%dT%H:%M:%SZ) done"
