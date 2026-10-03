#!/usr/bin/env bash
set -euo pipefail

: "${BOXPWNR_DIR:?Set BOXPWNR_DIR to the pinned BoxPwnr checkout}"
: "${LOCAL_OPENAI_BASE_URL:?Set LOCAL_OPENAI_BASE_URL to the private /v1 endpoint}"

export LOCAL_OPENAI_API_KEY="${LOCAL_OPENAI_API_KEY:-local}"
export LOCAL_OPENAI_MAX_TOKENS="${LOCAL_OPENAI_MAX_TOKENS:-4096}"
export LOCAL_OPENAI_CONTEXT_WINDOW="${LOCAL_OPENAI_CONTEXT_WINDOW:-360000}"
export LOCAL_OPENAI_TIMEOUT="${LOCAL_OPENAI_TIMEOUT:-900}"
export LOCAL_OPENAI_REASONING_EFFORT="${LOCAL_OPENAI_REASONING_EFFORT:-low}"
export LOCAL_OPENAI_THINKING_BUDGET="${LOCAL_OPENAI_THINKING_BUDGET:-0}"

cd "${BOXPWNR_DIR}"
uv run boxpwnr \
  --platform cybench \
  --target "[Very Easy] Delulu" \
  --solver single_loop \
  --model local-openai/apex-flash-1-abliterated \
  --image apex-cyber-attackbox:20261004 \
  --max-turns 100 \
  --max-time 45 \
  --default-execution-timeout 60 \
  --max-execution-timeout 300 \
  --reasoning-effort low \
  --debug
