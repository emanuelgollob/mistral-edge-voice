"""
Voxtral TTS launcher.

Runs the Voxtral-4B-TTS model under vllm-omni with two patches:

1. Caps GPU memory utilisation lower than vLLM's default. The TTS pipeline
   has two stages (text → audio tokens, then audio decoder); both are
   short-context and don't need the default ~90 % VRAM cap. Lowering frees
   headroom for the ASR + LLM servers sharing the same GPU.

2. Aliases ``_generate_pcm_chunks`` to ``_generate_audio_chunks``. The
   installed vllm-omni's HTTP handler for /v1/audio/speech/stream calls
   the former, but the class defines the latter (default
   response_format="pcm"). Drop this aliasing once upstream is consistent.

Must run inside the vllm-omni venv (see launch_servers.sh).
"""

import os
import sys

# Safety flags: prevent PyTorch deadlocks and NCCL silent hangs.
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
# TORCH_NCCL_BLOCKING_WAIT is the current name; NCCL_BLOCKING_WAIT still
# works but emits a DeprecationWarning on every launch. Set both so older
# wheels also pick it up.
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_BLOCKING_WAIT"] = "1"


def launch():
    # Import vLLM only inside this function so the parent process stays
    # clean of CUDA / Torch init before the spawn fork.
    import vllm.config
    _original_init = vllm.config.CacheConfig.__init__

    def _patched_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)
        # Stage 0 (text → audio-token LLM): cap at 0.15 of total VRAM.
        # Sentence-level synth never needs more KV; the lower cap leaves
        # headroom for the ASR + LLM servers sharing the GPU.
        if getattr(self, "gpu_memory_utilization", 0.9) > 0.3:
            self.gpu_memory_utilization = 0.15
        # Stage 1 (audio decoder): tiny model, trivial KV footprint.
        else:
            self.gpu_memory_utilization = 0.05

    vllm.config.CacheConfig.__init__ = _patched_init

    # vllm-omni's /v1/audio/speech/stream handler calls
    # OmniOpenAIServingSpeech._generate_pcm_chunks, but the class only
    # defines _generate_audio_chunks (default response_format="pcm").
    # Alias them so streaming speech works. Remove once upstream is
    # consistent.
    from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
    if not hasattr(OmniOpenAIServingSpeech, "_generate_pcm_chunks"):
        OmniOpenAIServingSpeech._generate_pcm_chunks = (
            OmniOpenAIServingSpeech._generate_audio_chunks
        )

    # CUDA graphs + torch.compile are enabled on the audio stage (no
    # --enforce-eager) to reduce TTS synthesis latency by ~0.7-1.5s per
    # reply. First request after startup pays a one-time "Capturing CUDA
    # graphs" cost of ~5-15s.
    #
    # If TTS produces garbled audio, hangs, or OOMs, add "--enforce-eager"
    # as the last sys.argv element below and restart.
    sys.argv = [
        "vllm", "serve", "mistralai/Voxtral-4B-TTS-2603",
        "--port", "8003",
        "--omni",
    ]

    print("🚀 Starting Voxtral TTS (vllm-omni, capped VRAM) on :8003 …")
    from vllm_omni.entrypoints.cli.main import main
    sys.exit(main())


if __name__ == "__main__":
    launch()
