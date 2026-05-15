#!/usr/bin/env bash
# ============================================================
# mistral-edge-voice — one-time venv bootstrap
# ============================================================
#
#   ./setup_venvs.sh
#
# Creates two PROJECT-LOCAL Python virtual environments inside this
# repo and installs their dependencies. Never touches $HOME/.venv or
# any other system / user-global venv, so your existing Python
# environments stay untouched.
#
#   .venv/main   vLLM + ASR (Voxtral Realtime) + LLM (Ministral 3 14B)
#                + client-side deps for voice_agent.py
#   .venv/tts    vllm-omni for Voxtral TTS
#
# Both are .gitignored. After bootstrap, launch_servers.sh auto-detects
# these paths and uses them — no env-var ritual needed. To use your own
# venvs instead, skip this script and set MAIN_VENV / TTS_VENV when
# launching:
#
#   MAIN_VENV=/path/to/main TTS_VENV=/path/to/tts ./launch_servers.sh
#
# Requirements: Python 3.10+, a CUDA-capable GPU, `git`, and `uv` on
# PATH (install with `curl -LsSf https://astral.sh/uv/install.sh | sh`
# if missing). Per-model install specifics (CUDA wheel selection,
# model download) live on each model's HuggingFace page — see README
# "Third-Party Components".
#
# Idempotent: re-running skips venv creation if a venv already exists
# and re-runs pip install (no-op or upgrade as appropriate). To redo
# from scratch, `rm -rf .venv/` first.
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_VENV_PATH="$SCRIPT_DIR/.venv/main"
TTS_VENV_PATH="$SCRIPT_DIR/.venv/tts"

PYTHON="${PYTHON:-python3}"

create_venv() {
    local target="$1"
    if [ -f "$target/bin/activate" ]; then
        echo "[setup] $target already exists — skipping creation"
    else
        echo "[setup] creating $target"
        "$PYTHON" -m venv "$target"
    fi
}

echo "──────────────────────────────────────────────────────────"
echo "[setup] mistral-edge-voice — bootstrap project-local venvs"
echo "        (your \$HOME venvs and other Python envs are not touched)"
echo
echo "        MAIN  → $MAIN_VENV_PATH"
echo "        TTS   → $TTS_VENV_PATH"
echo "──────────────────────────────────────────────────────────"
echo

# HF_TOKEN preflight (non-fatal). Mistral model weights are gated, so
# without a token the eventual launch_servers.sh run will fail to
# download them. Setup itself doesn't need the token (only pip), so we
# warn and continue rather than abort.
if [ -z "${HF_TOKEN:-}" ]; then
    cat >&2 <<'EOF'
[setup] WARNING: HF_TOKEN is not set in this shell.
[setup] Mistral model weights are gated; launch_servers.sh will hit auth
[setup] errors when the servers try to download them. You can finish
[setup] this script without HF_TOKEN, then export it before launching.
[setup] See README → Installation → Authentication.

EOF
fi

create_venv "$MAIN_VENV_PATH"
create_venv "$TTS_VENV_PATH"

# Both venvs use `uv` (per the Voxtral TTS install guide). If `uv` is
# not on your PATH, install with:
#   curl -LsSf https://astral.sh/uv/install.sh | sh

verify_mistral_common() {
    # Sanity-check that vllm pulled in mistral_common >= 1.10.0. The
    # Voxtral TTS guide notes that installing vllm >= 0.18.0 auto-
    # installs the right mistral_common; this catches the case where
    # an older vllm got resolved.
    python3 - <<'PY'
import sys, mistral_common
v = tuple(int(x) for x in mistral_common.__version__.split('.')[:3])
if v < (1, 10, 0):
    sys.exit(
        f"mistral_common {mistral_common.__version__} is too old "
        f"(need >= 1.10.0; install vllm >= 0.18.0)"
    )
print(f"[setup]   mistral_common {mistral_common.__version__}: ok")
PY
}

# ── MAIN_VENV: vllm + ASR audio deps + transformers + client deps ──
# Per the Voxtral-Mini Realtime install guide:
#   uv pip install -U vllm
#   uv pip install soxr librosa soundfile
#   uv pip install --upgrade transformers
# Plus client-side imports for voice_agent.py from requirements.txt
# (soxr overlap with the ASR audio deps is intentional and harmless).
echo
echo "[setup] installing into MAIN venv: vllm + audio libs + transformers + client deps"
(
    # shellcheck source=/dev/null
    source "$MAIN_VENV_PATH/bin/activate"
    uv pip install -U vllm
    uv pip install soxr librosa soundfile
    uv pip install --upgrade transformers
    uv pip install -r "$SCRIPT_DIR/requirements.txt"
    verify_mistral_common
)

# ── TTS_VENV: vllm + vllm-omni (per the Voxtral TTS install guide) ──
# https://huggingface.co/mistralai/Voxtral-4B-TTS-2603 directs users
# to `uv pip install -U vllm` plus `uv pip install vllm-omni --upgrade`
# (>= 0.18.0). vllm-omni is a separate package from mainline vllm and
# carries the OmniOpenAIServingSpeech endpoint launch_tts.py imports.
echo
echo "[setup] installing into TTS venv: vllm + vllm-omni (per Voxtral TTS guide)"
(
    # shellcheck source=/dev/null
    source "$TTS_VENV_PATH/bin/activate"
    uv pip install -U vllm
    uv pip install vllm-omni --upgrade
    verify_mistral_common
)

cat <<EOF

──────────────────────────────────────────────────────────
[setup] done.

Both venvs are ready. launch_servers.sh will detect them
automatically — no env-var export needed:

    ./launch_servers.sh

After all three servers are healthy, in another terminal:

    source "$MAIN_VENV_PATH/bin/activate"
    python voice_agent.py

Model weights for Voxtral Realtime / Ministral 3 14B / Voxtral TTS
will download from HuggingFace on first server boot (~1-2 min idle
in the launcher logs as that happens).
──────────────────────────────────────────────────────────
EOF
