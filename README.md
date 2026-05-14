# mistral-edge-voice

Full-duplex local voice agent stack for Mistral models (Voxtral Realtime, Ministral 3 14B, Voxtral TTS) on a single edge GPU.

> Developed for **Intimate Triage**, presented at **Ars Electronica Festival 2026**.

---

## About

`mistral-edge-voice` is an open-weights, on-device voice agent stack. It runs realtime ASR, an instruction-tuned LLM, and neural TTS together on a single edge GPU, with acoustic echo cancellation enabling full-duplex interaction — the agent can hear while it speaks.

The stack was developed in the context of the artistic-research project *Intimate Triage* (working title), exploring human–robot interaction through voice. It is released here as a general-purpose foundation for similar work.

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

| Venv | Default path | Contents |
|------|--------------|----------|
| `MAIN_VENV` | `$HOME/.venv` | vLLM with the ASR (Voxtral Realtime) and LLM (Ministral 3 14B) servers |
| `TTS_VENV`  | `$HOME/tts`   | vllm-omni with the Voxtral TTS server |

Override the paths via env vars:

```bash
MAIN_VENV=/path/to/main TTS_VENV=/path/to/tts ./launch_servers.sh
```

System requirements:

- Linux with PipeWire 1.x and `pactl` (`module-echo-cancel` with `aec_method=webrtc` available)
- CUDA-capable GPU (see [Hardware](#hardware))

Python client dependencies for `voice_agent.py` are listed in [`requirements.txt`](requirements.txt). Install with `pip install -r requirements.txt` into any standard venv (these are the agent-side imports — they're independent of `MAIN_VENV` / `TTS_VENV` which carry the inference servers).

Per-model installation, dependency requirements, and vLLM serving instructions live on each model's HuggingFace page (linked in [Third-Party Components](#third-party-components) below). A pinned-versions end-to-end recipe specific to this repo is TBD pending a verified test run.

## Quickstart

```bash
# Terminal 1: start the three vLLM servers (ASR, LLM, TTS). Sequential
# startup takes ~1-2 min. Ctrl+C here stops all three and releases GPU.
./launch_servers.sh

# Terminal 2: run the voice agent.
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
| [`mistralai/Voxtral-Mini-4B-Realtime-2602`](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) | Streaming ASR | *TBD — confirm Mistral release license* |
| [`mistralai/Ministral-3-14B-Instruct-2512`](https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512) | LLM | *TBD — confirm Mistral release license* |
| [`mistralai/Voxtral-4B-TTS-2603`](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) | Neural TTS | *TBD — confirm Mistral release license* |
| [vLLM](https://github.com/vllm-project/vllm) | ASR + LLM inference engine | Apache 2.0 |
| `vllm-omni` | Multimodal inference for TTS | *TBD — confirm release license* |
| [PipeWire `module-echo-cancel`](https://pipewire.org/) | Acoustic echo cancellation (WebRTC AEC3 backend) | LGPL-2.1+ (with BSD-3-Clause AEC3 via `webrtc-audio-processing`) |

Model weights are downloaded from their official sources; this repository does not redistribute them.

## License

Copyright (c) 2026 Emanuel Gollob. Developed as external contracted work for the Open Innovation in Science Center (Ludwig Boltzmann Gesellschaft) and the Department of Creative Robotics (Kunstuniversität Linz). See [NOTICE](NOTICE) for full attribution.

Source code is released under the Apache License 2.0 — see [LICENSE](LICENSE).

Third-party model weights are governed by their respective licenses (see "Third-Party Components" above).

## Acknowledgements

- **Mistral AI** — for releasing Voxtral Realtime, Ministral 3 14B, and Voxtral TTS as open-weights models.
- **NVIDIA Corporation** — for the NVIDIA RTX 6000 Pro Max-Q Workstation GPU used during development, awarded through the NVIDIA Academic AI Grant program.
- **Open Innovation in Science Center, Ludwig Boltzmann Gesellschaft** (Vienna, Austria).
- **Department of Creative Robotics, Kunstuniversität Linz** (Linz, Austria).
- **Ars Electronica Festival 2026** — presentation context for *Intimate Triage*.

## Citation

*TBD — citation block will be added when the accompanying festival entry / publication is available.*
