#!/usr/bin/env bash
# ============================================================
# mistral-edge-voice — vLLM stack launcher (ASR + LLM + TTS)
# ============================================================
#
#   ./launch_servers.sh
#
# Brings up all three vLLM inference servers and tears them down on a
# single Ctrl+C. Sequential startup (parallel triggers vLLM memory-
# profiling races). Each server runs in its own process group via
# setsid, so a single signal on the negative pgid takes the whole
# vllm subprocess tree down — no orphaned model workers left holding
# GPU memory.
#
#   :8001  ASR    Voxtral-Mini-4B-Realtime-2602   (WebSocket)
#   :8002  LLM    Ministral-3-14B-Instruct-2512   (HTTP / SSE)
#   :8003  TTS    Voxtral-4B-TTS-2603             (WebSocket, vllm-omni)
#
# Two venvs are required (do NOT merge them — vllm-omni's stage config
# breaks plain vllm):
#
#   MAIN_VENV  vllm with the ASR + LLM models
#   TTS_VENV   vllm-omni for the TTS launcher
#
# Resolution precedence for each path:
#   1. Explicit env var ($MAIN_VENV / $TTS_VENV) if set
#   2. Project-local venv at .venv/main / .venv/tts (created by setup_venvs.sh)
#   3. Fallback to $HOME/.venv / $HOME/tts
#
# So `./setup_venvs.sh && ./launch_servers.sh` works with no exports,
# while existing $HOME/.venv setups keep working unchanged. Override
# inline if neither default fits:
#
#   MAIN_VENV=/path/to/main TTS_VENV=/path/to/tts ./launch_servers.sh
#
# Logs land in tmp/{asr,llm,tts}.log; tail in another terminal.
# ============================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/tmp"
mkdir -p "$LOG_DIR"

# Venv resolution: explicit env var wins; otherwise prefer the project-
# local venvs created by setup_venvs.sh; otherwise fall back to the
# $HOME defaults. Lets setup_venvs.sh "just work" without an export
# ritual while leaving existing $HOME/.venv setups untouched.
if [ -z "${MAIN_VENV:-}" ]; then
    if [ -f "$SCRIPT_DIR/.venv/main/bin/activate" ]; then
        MAIN_VENV="$SCRIPT_DIR/.venv/main"
    else
        MAIN_VENV="$HOME/.venv"
    fi
fi
if [ -z "${TTS_VENV:-}" ]; then
    if [ -f "$SCRIPT_DIR/.venv/tts/bin/activate" ]; then
        TTS_VENV="$SCRIPT_DIR/.venv/tts"
    else
        TTS_VENV="$HOME/tts"
    fi
fi

STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-600}"   # seconds per server

export MAIN_VENV TTS_VENV SCRIPT_DIR

# Preflight: refuse to start if either venv path doesn't look like a venv.
# Pointing the user at setup_venvs.sh is friendlier than a confusing vllm
# "command not found" deep inside the per-server subshell.
preflight_check_venvs() {
    local missing=()
    [ -f "$MAIN_VENV/bin/activate" ] || missing+=("MAIN_VENV at $MAIN_VENV")
    [ -f "$TTS_VENV/bin/activate" ]  || missing+=("TTS_VENV at $TTS_VENV")
    if [ ${#missing[@]} -gt 0 ]; then
        echo "[launcher] no venv found at:" >&2
        for m in "${missing[@]}"; do echo "[launcher]   $m" >&2; done
        echo >&2
        echo "[launcher] Either run ./setup_venvs.sh to create project-local venvs," >&2
        echo "[launcher] or set MAIN_VENV / TTS_VENV to point at existing ones:" >&2
        echo "[launcher]   MAIN_VENV=/path/to/main TTS_VENV=/path/to/tts ./launch_servers.sh" >&2
        exit 1
    fi
}
preflight_check_venvs

# HF_TOKEN preflight (non-fatal). Without it, the first-time model
# download will fail; once weights are cached locally subsequent runs
# don't strictly need it. Warn but proceed.
if [ -z "${HF_TOKEN:-}" ]; then
    echo "[launcher] WARNING: HF_TOKEN is not set." >&2
    echo "[launcher]   If the model weights aren't already cached, vllm will fail to" >&2
    echo "[launcher]   download them. Export HF_TOKEN (see README → Installation →" >&2
    echo "[launcher]   Authentication) before running, or proceed if you've already" >&2
    echo "[launcher]   downloaded the weights." >&2
fi

# Captured pgids of children. `kill -SIG -<pgid>` hits the whole
# process group, including vllm's own subprocess tree.
declare -a CHILD_PGIDS=()
declare -a CHILD_LABELS=()

cleanup() {
    local rc=$?
    # Disarm the trap so signals during cleanup don't recurse.
    trap '' INT TERM EXIT
    echo
    echo "[launcher] tearing down (rc=$rc)"
    for i in "${!CHILD_PGIDS[@]}"; do
        local pgid="${CHILD_PGIDS[$i]}"
        local label="${CHILD_LABELS[$i]}"
        if kill -0 "-$pgid" 2>/dev/null; then
            echo "[launcher] kill -TERM -$pgid ($label)"
            kill -TERM "-$pgid" 2>/dev/null || true
        fi
    done
    # Grace period for graceful exit (vllm flushes KV cache, etc.)
    sleep 3
    for i in "${!CHILD_PGIDS[@]}"; do
        local pgid="${CHILD_PGIDS[$i]}"
        local label="${CHILD_LABELS[$i]}"
        if kill -0 "-$pgid" 2>/dev/null; then
            echo "[launcher] kill -KILL -$pgid ($label)"
            kill -KILL "-$pgid" 2>/dev/null || true
        fi
    done
    exit "$rc"
}
trap cleanup INT TERM EXIT

start_pg() {
    # Usage: start_pg <label> <logfile> <bash-body>
    # Starts the body as a new session/process group leader so we can
    # tree-kill the whole vllm worker stack later.
    local label="$1"; shift
    local logfile="$1"; shift
    local body="$1"; shift
    setsid bash -c "$body" > "$logfile" 2>&1 < /dev/null &
    local pid=$!
    # `setsid` makes the new process its own pgrp leader → pgid == pid.
    CHILD_PGIDS+=("$pid")
    CHILD_LABELS+=("$label")
    echo "[launcher] $label started (pid/pgid=$pid, log=$logfile)"
}

wait_for_health() {
    # Args: <pgid> <port> <label>
    local pgid="$1" port="$2" label="$3"
    local url="http://localhost:${port}/health"
    local deadline=$(( $(date +%s) + STARTUP_TIMEOUT ))
    echo "[launcher] waiting up to ${STARTUP_TIMEOUT}s for $label on :$port ..."
    while (( $(date +%s) < deadline )); do
        if ! kill -0 "-$pgid" 2>/dev/null; then
            echo "[launcher] $label (pgid $pgid) exited before becoming healthy"
            return 1
        fi
        if curl -sf -o /dev/null --max-time 2 "$url"; then
            echo "[launcher] $label healthy on :$port"
            return 0
        fi
        sleep 2
    done
    echo "[launcher] $label did not become healthy within ${STARTUP_TIMEOUT}s"
    return 1
}

# ── Clean slate ───────────────────────────────────────────────
echo "[launcher] killing any existing vllm processes …"
pkill -f vllm 2>/dev/null || true
sleep 3

# ── 1. ASR: Voxtral Realtime (port 8001) ──────────────────────
start_pg "asr" "$LOG_DIR/asr.log" '
    source "$MAIN_VENV/bin/activate"
    VLLM_DISABLE_COMPILE_CACHE=1 exec vllm serve \
        mistralai/Voxtral-Mini-4B-Realtime-2602 \
        --port 8001 \
        --gpu-memory-utilization 0.22 \
        --compilation_config "{\"cudagraph_mode\": \"PIECEWISE\"}" \
        --max-model-len 8192 \
        --max-num-seqs 4 \
        --max-num-batched-tokens 8192
'
wait_for_health "${CHILD_PGIDS[-1]}" 8001 "ASR" || exit 1

# ── 2. LLM: Ministral 3 14B Instruct (port 8002) ──────────────
start_pg "llm" "$LOG_DIR/llm.log" '
    source "$MAIN_VENV/bin/activate"
    exec vllm serve \
        mistralai/Ministral-3-14B-Instruct-2512 \
        --port 8002 \
        --gpu-memory-utilization 0.40 \
        --tokenizer_mode mistral \
        --config_format mistral \
        --max-model-len 8192 \
        --max-num-seqs 2 \
        --enable-prefix-caching
'
wait_for_health "${CHILD_PGIDS[-1]}" 8002 "LLM" || exit 1

# ── 3. TTS: Voxtral TTS via launch_tts.py (port 8003) ─────────
# `python -u` forces unbuffered stdout/stderr. Without it, a TTS crash
# during startup can lose its traceback to Python's block-buffer when
# the process dies — leaves tmp/tts.log near-empty and the failure mode
# undiagnosable. ASR/LLM go through vllm's CLI which handles this itself.
start_pg "tts" "$LOG_DIR/tts.log" '
    source "$TTS_VENV/bin/activate"
    exec python -u "$SCRIPT_DIR/launch_tts.py"
'
wait_for_health "${CHILD_PGIDS[-1]}" 8003 "TTS" || exit 1

echo
echo "──────────────────────────────────────────────────────────"
echo "[launcher] all servers healthy:"
echo "  ASR  :8001  pgid ${CHILD_PGIDS[0]}  log: $LOG_DIR/asr.log"
echo "  LLM  :8002  pgid ${CHILD_PGIDS[1]}  log: $LOG_DIR/llm.log"
echo "  TTS  :8003  pgid ${CHILD_PGIDS[2]}  log: $LOG_DIR/tts.log"
echo
echo "  Run the agent (new terminal):  python voice_agent.py"
echo "  Tail logs:                     tail -F $LOG_DIR/{asr,llm,tts}.log"
echo "  List audio devices:            python -m sounddevice"
echo "  Ctrl+C here to stop all."
echo "──────────────────────────────────────────────────────────"
echo

# Block until any server's process group leader exits. The trap then
# tears down the rest and exits non-zero.
wait -n
echo "[launcher] a server process exited — tearing down the others"
