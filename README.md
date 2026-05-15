# mistral-edge-voice

Full-duplex local voice agent stack for Mistral models (Voxtral Realtime, Ministral 3 14B, Voxtral TTS) on a single edge GPU.

> [!NOTE]
> **Artistic-research code, not production.** APIs and defaults may change as the project evolves; pin a specific commit if you need stability.

> Developed for **Intimate Triage**, presented at **Ars Electronica Festival 2026**.

---

## About

`mistral-edge-voice` is an open-weights, on-device voice agent stack. It runs realtime ASR, an instruction-tuned LLM, and neural TTS together on a single edge GPU, with acoustic echo cancellation enabling full-duplex interaction — the agent can hear while it speaks.

The stack was developed in the context of the artistic-research project *Intimate Triage* (working title), exploring human–robot interaction through voice. It is released here as a general-purpose foundation for similar work.

*As shipped, the stack is for non-commercial use only — one bundled model (Voxtral TTS) is CC BY-NC 4.0. The source code itself is Apache 2.0. See the [License](#license) section for the per-component breakdown.*

## Architecture

```
  mic ──▶ [AEC] ──▶ [Voxtral Realtime] ──▶ [Ministral 3 14B] ──▶ [Voxtral TTS] ──▶ speaker
            ▲                                                              │
            └──────────────── echo reference ──────────────────────────────┘
```

- **ASR:** Voxtral Realtime — streaming speech recognition
- **LLM:** Ministral 3 14B, served via vLLM
- **TTS:** Voxtral TTS — low-latency neural synthesis
- **AEC:** acoustic echo cancellation enabling barge-in / full-duplex

## Hardware

A single CUDA-capable GPU runs all three models concurrently. The launcher allocates VRAM as roughly **22% ASR + 40% LLM + ~15% TTS** (configurable in `launch_servers.sh`), so any card with enough headroom for that split works in principle.

**Reference card:** NVIDIA RTX 6000 Pro Max-Q Workstation GPU, granted through the **NVIDIA Academic AI Grant** program (NVIDIA Corporation). The default VRAM proportions in `launch_servers.sh` are tuned to this card.

*TBD — audio I/O reference setup will be pinned with verified test runs.*

## Installation

The stack uses **two Python virtual environments** to keep vLLM and vllm-omni isolated — they have conflicting dependencies and should not share an interpreter:

| Venv | Contents |
|------|----------|
| `MAIN_VENV` | vLLM with the ASR (Voxtral Realtime) and LLM (Ministral 3 14B) servers, plus the client-side imports for `voice_agent.py` |
| `TTS_VENV`  | vllm-omni with the Voxtral TTS server |

**Quick path — bootstrap both venvs in one command:**

```bash
./setup_venvs.sh
```

This creates project-local venvs at `.venv/main/` and `.venv/tts/`, installs vLLM + `requirements.txt` into the first and vllm-omni into the second. It never touches `$HOME/.venv` or other system Python environments. The launcher auto-detects these paths — no env-var export needed afterwards.

**Manual path** — if you'd rather install into your own venvs, set `MAIN_VENV` / `TTS_VENV` when launching:

```bash
MAIN_VENV=/path/to/main TTS_VENV=/path/to/tts ./launch_servers.sh
```

The launcher resolves the venv paths in this order: explicit env var → project-local `.venv/main` / `.venv/tts` → fallback to `$HOME/.venv` / `$HOME/tts`. A preflight check refuses to start with a pointer at `setup_venvs.sh` if none of those exist.

System requirements:

- Linux with PipeWire 1.x and `pactl` (`module-echo-cancel` with `aec_method=webrtc` available)
- CUDA-capable GPU (see [Hardware](#hardware))
- Python 3.10+, `git`, and [`uv`](https://docs.astral.sh/uv/) on `PATH` (install with `curl -LsSf https://astral.sh/uv/install.sh | sh` if missing) — `setup_venvs.sh` uses `uv pip` per the Voxtral TTS install guide

### Authentication

vLLM downloads the Mistral model weights from HuggingFace at first server boot. Before running either setup path:

1. Accept each model's terms on its HuggingFace page (linked in [Third-Party Components](#third-party-components)).
2. Create an access token at <https://huggingface.co/settings/tokens> (read-only is sufficient).
3. Export it in any shell that runs `setup_venvs.sh` or `launch_servers.sh`:

   ```bash
   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

   Persist by adding the line to your shell's rc file (`~/.bashrc` / `~/.zshrc` / similar).

Other env vars worth knowing:

| Env var | Purpose |
|---------|---------|
| `HF_HOME` | HuggingFace cache directory (default `~/.cache/huggingface`). Override if your home partition is small — combined model weights are ~30 GB. |
| `CUDA_VISIBLE_DEVICES` | Which GPU(s) to use if you have several. Default: all visible. |
| `STARTUP_TIMEOUT` | Per-server health-check deadline in `launch_servers.sh` (default 600 s). Raise on slow networks where first-time model downloads dominate. |

Per-model installation specifics and vLLM serving instructions live on each model's HuggingFace page (linked in [Third-Party Components](#third-party-components) below).

## Quickstart

```bash
# Terminal 1: start the three vLLM servers (ASR, LLM, TTS). Sequential
# startup takes ~1-2 min. Ctrl+C here stops all three and releases GPU.
./launch_servers.sh

# Terminal 2: activate MAIN_VENV (where the client deps live) and run
# the agent. If you used setup_venvs.sh, that's .venv/main:
source .venv/main/bin/activate
python voice_agent.py
```

Speak; the agent replies through the speaker. Barge-in is supported — start talking while the agent is speaking and it cuts itself off.

Common options:

```bash
python voice_agent.py --no-speculation              # A/B baseline: no speculative LLM+TTS during user speech
python voice_agent.py --voice de_female             # German voice (and language)
python voice_agent.py --mic "Wireless GO"           # pick mic by name substring
python voice_agent.py --prompt-file my_prompt.txt   # custom system prompt
```

## Configuration

- **`systemprompt.txt`** — system prompt loaded at startup. Edit in place, or point at a different file with `--prompt-file path/to/other.txt`.
- **CLI flags** — `python voice_agent.py --help` for the full list (mic, speaker, voice, prompt file, speculation toggle).
- **Tunable constants** — the top of `voice_agent.py` exposes turn-detection windows, speculation thresholds, and AEC warmup, each documented inline.
- **Server VRAM allocation** — adjust `--gpu-memory-utilization` in `launch_servers.sh` to fit your card.

## Third-Party Components

This project integrates the following third-party software and models. Each remains under its original license; consult the linked sources for full terms.

| Component | Role | License |
|-----------|------|---------|
| [`mistralai/Voxtral-Mini-4B-Realtime-2602`](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) | Streaming ASR | Apache 2.0 |
| [`mistralai/Ministral-3-14B-Instruct-2512`](https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512) | LLM | Apache 2.0 |
| [`mistralai/Voxtral-4B-TTS-2603`](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) | Neural TTS | **CC BY-NC 4.0** *(non-commercial)* |
| [vLLM](https://github.com/vllm-project/vllm) | ASR + LLM inference engine | Apache 2.0 |
| [`vllm-omni`](https://github.com/vllm-project/vllm-omni) | Multimodal inference for TTS | Apache 2.0 |
| [PipeWire `module-echo-cancel`](https://pipewire.org/) | Acoustic echo cancellation (WebRTC AEC3 backend) | LGPL-2.1+ (with BSD-3-Clause AEC3 via `webrtc-audio-processing`) |

Model weights are downloaded from their official sources; this repository does not redistribute them.

## License

Copyright (c) 2026 Emanuel Gollob. Developed as external contracted work for the Open Innovation in Science Center (Ludwig Boltzmann Gesellschaft) and the Department of Creative Robotics (Kunstuniversität Linz). See [NOTICE](NOTICE) for full attribution.

Source code is released under the Apache License 2.0 — see [LICENSE](LICENSE).

Third-party model weights are governed by their respective licenses (see "Third-Party Components" above).

**Note:** while the source code and two of the three model weights are permissively licensed (Apache 2.0), the Voxtral TTS weights are CC BY-NC 4.0 — *non-commercial use only*. The stack as-shipped is therefore non-commercial; commercial deployment requires either swapping the TTS component or obtaining a separate license from Mistral.

## Acknowledgements

- **Mistral AI** — for releasing Voxtral Realtime, Ministral 3 14B, and Voxtral TTS as open-weights models.
- **NVIDIA Corporation** — for the NVIDIA RTX 6000 Pro Max-Q Workstation GPU used during development, awarded through the NVIDIA Academic AI Grant program.
- **Open Innovation in Science Center, Ludwig Boltzmann Gesellschaft** (Vienna, Austria).
- **Department of Creative Robotics, Kunstuniversität Linz** (Linz, Austria).
- **Ars Electronica Festival 2026** — presentation context for *Intimate Triage*.

## Citation

Machine-readable citation metadata is in [`CITATION.cff`](CITATION.cff); GitHub also exposes a "Cite this repository" button on the repo page that reads from it. A formal paper / festival entry citation will be added here once the accompanying publication is available.
