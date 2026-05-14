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

*TBD — target single edge GPU (VRAM budget, reference card) and audio I/O setup will be pinned as the stack stabilizes.*

## Installation

*TBD — installation instructions will be added as the stack stabilizes.*

```bash
git clone https://github.com/<user>/mistral-edge-voice.git
cd mistral-edge-voice
# install deps...
```

## Quickstart

*TBD — launch instructions for the ASR, LLM, and TTS services and the full-duplex orchestrator.*

## Configuration

*TBD — configuration schema for model paths, audio devices, AEC parameters, and conversation policy.*

## Third-Party Components

This project integrates the following third-party software and models. Each remains under its original license; consult the linked sources for full terms.

| Component | Role | License |
|-----------|------|---------|
| [Voxtral Realtime](https://mistral.ai) | Streaming ASR | *TBD — confirm Mistral release license* |
| [Ministral 3 14B](https://mistral.ai) | LLM | *TBD — confirm Mistral release license* |
| [Voxtral TTS](https://mistral.ai) | Neural TTS | *TBD — confirm Mistral release license* |
| [vLLM](https://github.com/vllm-project/vllm) | LLM inference engine | Apache 2.0 |
| AEC backend | Acoustic echo cancellation | *TBD — backend not yet selected* |

Model weights are downloaded from their official sources; this repository does not redistribute them.

## License

Copyright (c) 2026 Emanuel Gollob. Developed as external contracted work for the Open Innovation in Science Center (Ludwig Boltzmann Gesellschaft) and the Department of Creative Robotics (Kunstuniversität Linz). See [NOTICE](NOTICE) for full attribution.

Source code is released under the Apache License 2.0 — see [LICENSE](LICENSE).

Third-party model weights are governed by their respective licenses (see "Third-Party Components" above).

## Acknowledgements

- **Mistral AI** — for releasing Voxtral Realtime, Ministral 3 14B, and Voxtral TTS as open-weights models.
- **Open Innovation in Science Center, Ludwig Boltzmann Gesellschaft** (Vienna, Austria).
- **Department of Creative Robotics, Kunstuniversität Linz** (Linz, Austria).
- **Ars Electronica Festival 2026** — presentation context for *Intimate Triage*.

## Citation

*TBD — citation block will be added when the accompanying festival entry / publication is available.*
