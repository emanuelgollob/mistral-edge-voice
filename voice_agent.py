#!/usr/bin/env python3
"""
mistral-edge-voice — Full-duplex voice agent
==============================================
Local voice agent on top of three Mistral models: Voxtral Realtime
(streaming ASR), Ministral 3 14B (LLM via vLLM), Voxtral TTS
(streaming neural synthesis). Full-duplex enabled by WebRTC AEC3
through PipeWire's module-echo-cancel — the agent can hear while
it speaks, so open speakers are fine (headphones still work too).

Speculation is enabled by default: while the user is still speaking,
the agent opportunistically fires LLM+TTS on each transcript snapshot
that has grown by SPECULATION_MIN_GROWTH characters since the last
fire, accumulating the resulting audio in a private hold buffer. On
end-of-turn, if the final transcript matches the latest snapshot, the
held audio is released to the speaker — yielding near-zero-latency
replies on hits. On miss, the held audio is discarded and a fresh
reply is fired. Toggle off with `--no-speculation` for an A/B baseline.

Servers: ./launch_servers.sh must be running first (ASR on :8001,
LLM on :8002, TTS on :8003).

Usage:
    python voice_agent.py                              # defaults
    python voice_agent.py --mic 14                     # override mic by index
    python voice_agent.py --mic <name>                 # override mic by name
    python voice_agent.py --voice de_female            # German voice
    python voice_agent.py --no-speculation             # disable speculation
    python voice_agent.py --prompt-file my_prompt.txt  # custom system prompt

Trade-offs:
    - Barge-in during the first AEC_WARMUP seconds of each TTS reply
      is suppressed while AEC3 converges on the new playback reference.
    - ~1-2 s of underrun distortion at the start of each reply while
      PipeWire spins up its end of the AEC chain (mitigated by using
      default latency, not "low").

Available TTS voices (Voxtral-4B-TTS-2603, queried via
`curl http://localhost:8003/v1/audio/voices`). The voice also
determines the spoken language — pick one that matches the language
the LLM is responding in:

    English (neutral)   neutral_female, neutral_male,
                        casual_female,  casual_male,
                        cheerful_female, cheerful_male
    German              de_female, de_male
    French              fr_female, fr_male
    Spanish             es_female, es_male
    Italian             it_female, it_male
    Dutch               nl_female, nl_male
    Portuguese          pt_female, pt_male
    Hindi               hi_female, hi_male
    Arabic              ar_male

Ministral 14B is multilingual; it answers in whatever language you
speak in, so usually you just pick the voice for that language.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import re
import shutil
import string
import subprocess
import threading
import time
from pathlib import Path
from typing import AsyncIterator

import httpx
import numpy as np
import sounddevice as sd
import soxr
import websockets


# ── Server URLs ────────────────────────────────────────────────
ASR_URL    = "ws://localhost:8001/v1/realtime"
LLM_URL    = "http://localhost:8002/v1/chat/completions"
TTS_WS_URL = "ws://localhost:8003/v1/audio/speech/stream"

ASR_MODEL = "mistralai/Voxtral-Mini-4B-Realtime-2602"
LLM_MODEL = "mistralai/Ministral-3-14B-Instruct-2512"
TTS_MODEL = "mistralai/Voxtral-4B-TTS-2603"

# ── Audio ──────────────────────────────────────────────────────
SAMPLE_RATE   = 16000
BLOCKSIZE     = 1600            # 100 ms @ 16 kHz
TTS_STREAM_SR = 24000
TTS_DEVICE_SR = 48000

# ── LLM ────────────────────────────────────────────────────────
LLM_MAX_TOKENS  = 150           # ~3-4 sentences typical
LLM_TEMPERATURE = 0.7
LLM_STOP        = []            # let the model decide; no early period-cut
MAX_HISTORY     = 30

DEFAULT_PROMPT_FILE = "systemprompt.txt"

# ── Turn detection ─────────────────────────────────────────────
# Voxtral's transcription.done is the primary end-of-turn signal;
# if it doesn't fire, fall back to this stability window. 1.50s
# tolerates natural mid-sentence pauses without fragmenting turns.
END_OF_TURN_STABILITY = 1.50
TURN_MIN_CHARS        = 4
# Lower than TURN_MIN_CHARS so interrupts fire on the first
# transcribed character or two, while spurious single-char turns
# still get filtered.
BARGEIN_MIN_CHARS     = 2
GRACE_AFTER_TURN      = 0.20
GRACE_AFTER_TTS       = 0.30

# ── Speculation ────────────────────────────────────────────────
# Don't speculate on tiny burps.
SPECULATION_MIN_CHARS  = 8
# Re-fire only when transcript has grown by this much since last fire.
# 16 chars ≈ 2-3 words. Keeps churn on the TTS server low while still
# landing a fresh snapshot near the end of every turn — most spec hits
# are the LAST fire anyway.
SPECULATION_MIN_GROWTH = 16
# Safety cap: discard speculation whose hold buffer exceeds this many
# bytes (≈ 30 s of 24 kHz int16 mono after 24→48 k upsample).
SPECULATION_MAX_HOLD_BYTES = 30 * TTS_DEVICE_SR * 2

# ── AEC ────────────────────────────────────────────────────────
# Time AEC3 needs to converge on the new playback reference signal.
# We gate ASR for this many seconds from the moment the speaker stream
# opens for a reply, so residual echo doesn't trigger false barge-ins
# during convergence.
AEC_WARMUP = 0.5

# Names registered with PipeWire when we load module-echo-cancel.
EC_SOURCE_NAME = "ec_source"
EC_SINK_NAME   = "ec_sink"


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("voice_agent")


def _device_label(arg: int | str | None, kind: str) -> str:
    """Human-readable label for a sounddevice reference. arg=None means
    'use system default for this direction' — sounddevice resolves that
    via PortAudio's PulseAudio host API, which on PipeWire 1.x reflects
    whatever's selected in the system sound settings. arg may also be
    a name (e.g. 'ec_source') for substring resolution."""
    try:
        info = sd.query_devices(kind=kind) if arg is None else sd.query_devices(arg)
    except Exception:
        return f"{arg!r} (query failed)"
    name = info.get("name", "?")
    idx  = info.get("index", "?")
    if arg is None:
        return f"system default '{name}' (#{idx})"
    return f"#{idx} '{name}'"


# ── Echo-cancel module lifecycle ───────────────────────────────

class EchoCancelModule:
    """Loads PipeWire's WebRTC echo-cancel module on startup, unloads
    on exit. Also unloads any pre-existing `module-echo-cancel`
    instances at startup so repeated runs don't stack devices."""

    def __init__(self, source_name: str = EC_SOURCE_NAME, sink_name: str = EC_SINK_NAME):
        self.source_name = source_name
        self.sink_name   = sink_name
        self.module_id: int | None = None
        # Captured from `pactl info` at load() time so we can restore
        # them on unload(). PortAudio's ALSA host API doesn't enumerate
        # individual PulseAudio sources by name (only an aggregated
        # "default" entry), so the way to route through the AEC is to
        # make ec_source/ec_sink the system default for the script's
        # lifetime, then restore the user's previous defaults.
        self._prev_default_source: str | None = None
        self._prev_default_sink:   str | None = None

    def load(self) -> None:
        if shutil.which("pactl") is None:
            raise RuntimeError("pactl not found in PATH — is pipewire-pulse / pulseaudio installed?")

        # Capture defaults BEFORE the unload+load sequence so we see the
        # user's real preferred devices, not whatever PulseAudio just
        # auto-promoted to fill the gap left by unloading the prior
        # module-echo-cancel (which often ends up being our own freshly
        # created ec_source/ec_sink).
        self._capture_previous_defaults()

        # If a previous run crashed without restoring, the captured
        # defaults will themselves be ec_* names — don't try to restore
        # to those on exit, and don't pin them as masters.
        if self._prev_default_source and self._prev_default_source.startswith("ec_"):
            log.warning("Captured source default %r looks like a stuck AEC source; "
                        "ignoring (will not restore on exit, will not pin as master).",
                        self._prev_default_source)
            self._prev_default_source = None
        if self._prev_default_sink and self._prev_default_sink.startswith("ec_"):
            log.warning("Captured sink default %r looks like a stuck AEC sink; "
                        "ignoring (will not restore on exit, will not pin as master).",
                        self._prev_default_sink)
            self._prev_default_sink = None

        self._unload_pre_existing()

        cmd = [
            "pactl", "load-module", "module-echo-cancel",
            "aec_method=webrtc",
            f"source_name={self.source_name}",
            f"sink_name={self.sink_name}",
            "use_master_format=1",
        ]
        # Explicit master pinning — without these, PulseAudio picks the
        # master based on whatever's default at module-load instant,
        # which is racy after the unload-pre-existing step (the previous
        # ec_* may still be teardown-pending, so default temporarily
        # falls back to *anything*, not necessarily the user's speakers).
        # Pinning to the captured "real" defaults makes ec_source/ec_sink
        # route through the user's actual mic and speaker deterministically.
        if self._prev_default_sink:
            cmd.append(f"sink_master={self._prev_default_sink}")
        if self._prev_default_source:
            cmd.append(f"source_master={self._prev_default_source}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"pactl load-module failed (exit {proc.returncode}): "
                f"stderr={proc.stderr.strip()!r}"
            )
        try:
            self.module_id = int(proc.stdout.strip())
        except ValueError:
            raise RuntimeError(f"unexpected pactl output: {proc.stdout!r}")
        log.info(
            "Loaded echo-cancel module %d (source=%s sink=%s aec_method=webrtc, "
            "sink_master=%s, source_master=%s)",
            self.module_id, self.source_name, self.sink_name,
            self._prev_default_sink, self._prev_default_source,
        )
        self._wait_for_devices()
        self._set_as_defaults()
        self._refresh_sounddevice()

    def _capture_previous_defaults(self) -> None:
        """Read `pactl info` to remember what the system defaults were
        before we change them, so we can restore on unload."""
        proc = subprocess.run(["pactl", "info"], capture_output=True, text=True)
        if proc.returncode != 0:
            log.warning("pactl info failed; cannot capture previous defaults: %s",
                        proc.stderr.strip())
            return
        for line in proc.stdout.splitlines():
            if line.startswith("Default Source: "):
                self._prev_default_source = line.split(": ", 1)[1].strip()
            elif line.startswith("Default Sink: "):
                self._prev_default_sink = line.split(": ", 1)[1].strip()
        log.info("Captured previous defaults (source=%s, sink=%s)",
                 self._prev_default_source, self._prev_default_sink)

    def _set_as_defaults(self) -> None:
        """Make ec_source / ec_sink the system default so PortAudio's
        'default' device routes through the AEC."""
        for cmd in (
            ["pactl", "set-default-source", self.source_name],
            ["pactl", "set-default-sink",   self.sink_name],
        ):
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                log.warning("%s failed: %s", " ".join(cmd), proc.stderr.strip())
                continue
        log.info("Set %s/%s as system defaults (sounddevice 'default' will route here)",
                 self.source_name, self.sink_name)

    def _restore_defaults(self) -> None:
        """Restore the system defaults captured before we switched them.
        Called on unload, *before* the module itself is removed."""
        if self._prev_default_source:
            subprocess.run(
                ["pactl", "set-default-source", self._prev_default_source],
                capture_output=True,
            )
        if self._prev_default_sink:
            subprocess.run(
                ["pactl", "set-default-sink", self._prev_default_sink],
                capture_output=True,
            )
        if self._prev_default_source or self._prev_default_sink:
            log.info("Restored previous defaults (source=%s, sink=%s)",
                     self._prev_default_source, self._prev_default_sink)

    @staticmethod
    def _refresh_sounddevice() -> None:
        """Force sounddevice/PortAudio to re-enumerate so any cached
        device list reflects the freshly loaded ec_source/ec_sink.
        PortAudio's PulseAudio host API caches on first use; new
        PulseAudio nodes added after that cache is built aren't seen
        until we re-init.

        Note: PortAudio doesn't expose individual PulseAudio nodes by
        name (only an aggregated 'default' entry that follows whichever
        sink/source PulseAudio considers default). That's fine — we set
        ec_source/ec_sink as the PulseAudio defaults in _set_as_defaults,
        and open streams with device=None ('default') so they route
        through the AEC. We deliberately don't try to resolve
        ec_source/ec_sink by name here — that would always 'fail' and
        produce misleading warnings."""
        try:
            sd._terminate()
            sd._initialize()
        except Exception as e:
            log.warning("sounddevice/PortAudio refresh failed: %s", e)

    def _unload_pre_existing(self) -> None:
        proc = subprocess.run(
            ["pactl", "list", "short", "modules"],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            return
        unloaded = 0
        for line in proc.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2 and parts[1] == "module-echo-cancel":
                module_id = parts[0]
                subprocess.run(["pactl", "unload-module", module_id], capture_output=True)
                unloaded += 1
        if unloaded:
            log.info("Unloaded %d pre-existing module-echo-cancel instance(s)", unloaded)

    def _wait_for_devices(self, timeout: float = 5.0) -> None:
        """Block until both source and sink show up in pactl listings.
        Without this, opening the device immediately after load can race
        and fail to find the new node."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            sources = self._list_short("sources")
            sinks   = self._list_short("sinks")
            if self.source_name in sources and self.sink_name in sinks:
                time.sleep(0.2)
                return
            time.sleep(0.1)
        raise RuntimeError(
            f"Echo-cancel devices did not appear within {timeout:.1f}s "
            f"(looking for source={self.source_name!r}, sink={self.sink_name!r})"
        )

    @staticmethod
    def _list_short(kind: str) -> set[str]:
        proc = subprocess.run(
            ["pactl", "list", "short", kind],
            capture_output=True, text=True,
        )
        names: set[str] = set()
        if proc.returncode != 0:
            return names
        for line in proc.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                names.add(parts[1])
        return names

    def unload(self) -> None:
        if self.module_id is None:
            return
        # Restore defaults BEFORE unloading the module — otherwise the
        # ec_source/ec_sink references disappear mid-way and the system
        # is briefly stuck with broken defaults until PulseAudio fixes
        # itself.
        self._restore_defaults()
        proc = subprocess.run(
            ["pactl", "unload-module", str(self.module_id)],
            capture_output=True, text=True,
        )
        if proc.returncode == 0:
            log.info("Unloaded echo-cancel module %d", self.module_id)
        else:
            log.warning(
                "pactl unload-module %d failed: %s",
                self.module_id, proc.stderr.strip(),
            )
        self.module_id = None


# ── Snapshot normalization ─────────────────────────────────────

_PUNCT_RE = re.compile(r"[" + re.escape(string.punctuation) + r"]+$")
_WS_RE    = re.compile(r"\s+")


def _normalize_for_match(s: str) -> str:
    """Lower / strip / collapse whitespace / drop trailing punctuation.
    Used for speculation snapshot vs final-transcript comparison."""
    s = _WS_RE.sub(" ", s.strip().lower())
    s = _PUNCT_RE.sub("", s).strip()
    return s


# ── ASR stream ─────────────────────────────────────────────────

class ASRStream:
    """Continuous ASR. peek_text() takes a snapshot without clearing
    so speculation can fire on intermediate transcripts."""

    def __init__(self):
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._current_text: str = ""
        self._lock = asyncio.Lock()
        self._gate_until: float = 0.0
        self.events: asyncio.Queue = asyncio.Queue()
        # Diagnostic: timestamp set from play_reply right when barge-in
        # detection begins; the next non-gated delta reports its latency
        # and clears it.
        self._tts_started_at: float | None = None

    def mark_tts_started(self) -> None:
        self._tts_started_at = time.monotonic()

    def unmark_tts_started(self) -> None:
        self._tts_started_at = None

    async def connect(self):
        log.info("Connecting to ASR …")
        self._ws = await websockets.connect(ASR_URL, max_size=10 * 1024 * 1024, ping_interval=30)
        resp = json.loads(await self._ws.recv())
        if resp.get("type") != "session.created":
            log.warning("Unexpected first ASR message: %s", resp.get("type"))
        await self._ws.send(json.dumps({"type": "session.update", "model": ASR_MODEL}))
        await self._ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        log.info("ASR ready.")

    async def send_audio(self, pcm_bytes: bytes):
        if self._ws:
            await self._ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm_bytes).decode("utf-8"),
            }))

    def gate_for(self, seconds: float) -> None:
        self._gate_until = time.monotonic() + seconds

    def _gated(self) -> bool:
        return time.monotonic() < self._gate_until

    async def receive_loop(self):
        async for message in self._ws:
            try:
                data = json.loads(message)
            except (json.JSONDecodeError, TypeError):
                continue
            t = data.get("type", "")
            if t == "transcription.delta":
                if self._gated():
                    continue
                delta = data.get("delta", "")
                if delta:
                    if self._tts_started_at is not None:
                        latency = time.monotonic() - self._tts_started_at
                        log.info("First ASR delta during TTS: %.2fs (text=%r)", latency, delta)
                        self._tts_started_at = None
                    async with self._lock:
                        self._current_text += delta
                    await self.events.put(("delta", delta))
            elif t == "transcription.done":
                if self._gated():
                    continue
                final = data.get("text", "")
                log.info("ASR DONE event (text=%r)", final)
                if final:
                    async with self._lock:
                        self._current_text = final
                await self.events.put(("done", final))
            elif t == "error":
                log.error("ASR error: %s", data.get("error", data))

    async def current_text_len(self) -> int:
        async with self._lock:
            return len(self._current_text.strip())

    async def peek_text(self) -> str:
        async with self._lock:
            return self._current_text.strip()

    async def take_transcript(self) -> str:
        async with self._lock:
            txt = self._current_text.strip()
            self._current_text = ""
            return txt

    async def drain_events(self) -> None:
        while not self.events.empty():
            try:
                self.events.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def close(self):
        if self._ws:
            try:
                await self._ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
            except Exception:
                pass
            await self._ws.close()


# ── Mic capture ────────────────────────────────────────────────

class MicCapture:
    """Continuous mic capture. Device may be an int (sounddevice index)
    or a string (name substring matched by sounddevice)."""

    def __init__(self, device: int | str | None):
        self.device = device
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._stream: sd.InputStream | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            log.warning("mic: %s", status)
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, indata.copy().tobytes())

    async def start(self):
        self._loop = asyncio.get_running_loop()
        self._stream = sd.InputStream(
            device=self.device,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=BLOCKSIZE,
            callback=self._callback,
        )
        self._stream.start()
        log.info("Mic started (device=%r)", self.device)

    async def next_chunk(self) -> bytes:
        return await self._queue.get()

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()


# ── LLM (streaming) ────────────────────────────────────────────

class LLMClient:
    def __init__(self, system_prompt: str):
        self.history: list[dict] = [{"role": "system", "content": system_prompt}]
        self._client = httpx.AsyncClient(timeout=60.0)

    def add_user(self, text: str):
        self.history.append({"role": "user", "content": text})
        self._trim()

    def add_assistant(self, text: str):
        self.history.append({"role": "assistant", "content": text})
        self._trim()

    def _trim(self):
        if len(self.history) > MAX_HISTORY + 1:
            self.history = [self.history[0]] + self.history[-MAX_HISTORY:]

    async def generate_stream(
        self,
        messages_override: list[dict] | None = None,
        tag: str = "",
    ) -> AsyncIterator[str]:
        """When `messages_override` is provided, use that exact message
        list (no mutation of self.history). Speculation uses this to
        try out a snapshot without committing it to history."""
        messages = messages_override if messages_override is not None else self.history
        payload = {
            "model": LLM_MODEL,
            "messages": messages,
            "max_tokens": LLM_MAX_TOKENS,
            "temperature": LLM_TEMPERATURE,
            "stop": LLM_STOP,
            "stream": True,
        }
        t0 = time.monotonic()
        ttft_logged = False
        async with self._client.stream("POST", LLM_URL, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                choices = data.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {}).get("content", "")
                if delta:
                    if not ttft_logged:
                        log.info("LLM TTFT%s %.2fs", f" ({tag})" if tag else "", time.monotonic() - t0)
                        ttft_logged = True
                    yield delta

    async def aclose(self):
        await self._client.aclose()


# ── Speaker output (long-lived buffer, per-reply stream) ───────

class SpeakerOutput:
    """Persistent audio buffer + lock that every TTS session writes
    into; the underlying sounddevice stream is opened per-reply (in
    play_reply) and closed when the reply ends.

    Per-reply lifecycle is required for AEC routing: a continuously-
    running output client on PipeWire's ec_sink stalls the AEC graph
    and starves the mic on ec_source. Opening the stream only while
    we're actually playing keeps the AEC chain "engaged" briefly. The
    buffer itself stays alive across the gap so speculation's hold→
    release flow still works."""

    def __init__(self, device: int | str | None):
        self.device = device
        self.audio_buf = bytearray()
        self.buf_lock = threading.Lock()
        self._stream: sd.RawOutputStream | None = None

    def _callback(self, outdata, frames, time_info, status):
        nbytes = frames * 2
        with self.buf_lock:
            have = len(self.audio_buf)
            if have >= nbytes:
                outdata[:] = bytes(self.audio_buf[:nbytes])
                del self.audio_buf[:nbytes]
            elif have > 0:
                outdata[:have] = bytes(self.audio_buf[:have])
                outdata[have:] = b"\x00" * (nbytes - have)
                self.audio_buf.clear()
            else:
                outdata[:] = b"\x00" * nbytes

    def start(self):
        if self._stream is not None:
            return
        # No latency="low": we open a fresh stream per reply through
        # PipeWire's AEC routing chain, and a tiny OS buffer underruns
        # repeatedly during the first 1-2s while PipeWire spins up its
        # end of the graph (audible as distortion at the start of each
        # TTS reply). Default (~200ms) gives the chain headroom; cutoff
        # on interrupt is still tight because the fade-out is spliced
        # into the audio buffer before the callback drains it.
        self._stream = sd.RawOutputStream(
            samplerate=TTS_DEVICE_SR, channels=1, dtype="int16",
            device=self.device, callback=self._callback,
        )
        self._stream.start()
        log.info("Speaker stream opened (device=%r)", self.device)

    def stop(self):
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def sink(self) -> tuple[bytearray, threading.Lock]:
        return (self.audio_buf, self.buf_lock)

    def is_empty(self) -> bool:
        with self.buf_lock:
            return len(self.audio_buf) == 0

    def clear(self):
        with self.buf_lock:
            self.audio_buf.clear()

    def clear_with_fadeout(self, fadeout_ms: float = 10.0) -> None:
        """Replace the buffered audio with a short linearly-faded-out
        version of its head, then silence. Eliminates the click / "tss"
        artifact you'd hear from terminating non-zero-crossing audio
        with a hard buffer clear (especially mid-sibilant)."""
        n_samples = max(1, int(TTS_DEVICE_SR * fadeout_ms / 1000))
        n_bytes   = n_samples * 2  # int16 = 2 bytes/sample
        with self.buf_lock:
            if not self.audio_buf:
                return
            head = bytes(self.audio_buf[:n_bytes])
            if len(head) < n_bytes:
                head = head + b"\x00" * (n_bytes - len(head))
            samples = np.frombuffer(head, dtype=np.int16).astype(np.float32)
            ramp    = np.linspace(1.0, 0.0, len(samples), dtype=np.float32)
            faded   = (samples * ramp).astype(np.int16).tobytes()
            self.audio_buf.clear()
            self.audio_buf.extend(faded)

    async def drain(self, poll: float = 0.02):
        while not self.is_empty():
            await asyncio.sleep(poll)


# ── TTS session (one ws lifecycle, swappable sink) ─────────────

class TTSSession:
    """One TTS websocket session. Sender pipes a text stream to the
    server; receiver writes resulting PCM (after 24→48 k resample)
    into the currently registered sink. The sink can be swapped
    atomically — used by SpeculationManager.release() to flip a
    held-buffer session over to the speaker mid-stream."""

    def __init__(self, voice: str):
        self.voice = voice
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._sender_task: asyncio.Task | None = None
        self._recv_task: asyncio.Task | None = None
        self._sink: tuple[bytearray, threading.Lock] | None = None
        self._sink_lock = threading.Lock()
        self._cancelled = False
        self.full_text_parts: list[str] = []
        self.done = asyncio.Event()
        self.t_open = 0.0
        self.t_first: float | None = None

    def set_sink(self, sink: tuple[bytearray, threading.Lock]) -> None:
        with self._sink_lock:
            self._sink = sink

    def cancel_writes(self) -> None:
        """Synchronously stop any further sink writes. Call this just
        before mutating the speaker buffer directly (e.g. for fade-out
        on interrupt) to prevent the receiver task from racing in a
        fresh chunk that would defeat the fade."""
        self._cancelled = True

    def _write_to_sink(self, data: bytes) -> None:
        if self._cancelled:
            return
        with self._sink_lock:
            sink = self._sink
        if sink is None:
            return
        buf, lock = sink
        with lock:
            buf.extend(data)

    async def start(self, text_stream: AsyncIterator[str]) -> None:
        self.t_open = time.monotonic()
        self._ws = await websockets.connect(TTS_WS_URL, max_size=50 * 1024 * 1024)
        await self._ws.send(json.dumps({
            "type": "session.config",
            "model": TTS_MODEL,
            "voice": self.voice,
            "response_format": "pcm",
            "stream_audio": True,
        }))
        self._sender_task = asyncio.create_task(self._sender(text_stream))
        self._recv_task   = asyncio.create_task(self._receiver())

    async def _sender(self, text_stream: AsyncIterator[str]) -> None:
        try:
            async for delta in text_stream:
                if self._cancelled:
                    break
                self.full_text_parts.append(delta)
                try:
                    await self._ws.send(json.dumps({"type": "input.text", "text": delta}))
                except Exception:
                    break
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("LLM stream error: %s", e)
        finally:
            joined = "".join(self.full_text_parts).rstrip()
            if joined and joined[-1] not in ".!?":
                try:
                    await self._ws.send(json.dumps({"type": "input.text", "text": "."}))
                except Exception:
                    pass
            try:
                await self._ws.send(json.dumps({"type": "input.done"}))
            except Exception:
                pass

    async def _receiver(self) -> None:
        resampler = soxr.ResampleStream(TTS_STREAM_SR, TTS_DEVICE_SR, 1, dtype="int16")
        try:
            while not self._cancelled:
                try:
                    msg = await asyncio.wait_for(self._ws.recv(), timeout=0.05)
                except asyncio.TimeoutError:
                    continue
                except websockets.ConnectionClosed:
                    break

                if isinstance(msg, (bytes, bytearray)):
                    if self.t_first is None:
                        self.t_first = time.monotonic()
                        log.info("TTS first audio in %.2fs", self.t_first - self.t_open)
                    arr24 = np.frombuffer(msg, dtype=np.int16)
                    arr48 = resampler.resample_chunk(arr24)
                    if arr48.size:
                        self._write_to_sink(arr48.astype(np.int16).tobytes())
                    continue

                try:
                    data = json.loads(msg)
                except (ValueError, TypeError):
                    continue
                if data.get("type") == "session.done":
                    tail = resampler.resample_chunk(np.zeros(0, dtype=np.int16), last=True)
                    if tail.size:
                        self._write_to_sink(tail.astype(np.int16).tobytes())
                    break
                elif data.get("type") == "error":
                    log.error("TTS stream error: %s", data.get("message"))
                    break
        except asyncio.CancelledError:
            raise
        finally:
            self.done.set()

    async def close(self, cancelled: bool = False) -> None:
        self._cancelled = self._cancelled or cancelled
        for task in (self._sender_task, self._recv_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        self.done.set()

    @property
    def full_text(self) -> str:
        s = "".join(self.full_text_parts).strip()
        if s and s[-1] not in ".!?":
            s += "."
        return s


# ── Speculation manager ────────────────────────────────────────

class SpeculationManager:
    """At most one in-flight speculation. Audio accumulates in a
    private hold buffer. release() flushes the hold buffer into the
    speaker sink and redirects the live session there. discard()
    cancels and throws away the hold buffer."""

    def __init__(self, voice: str, llm: LLMClient):
        self.voice = voice
        self.llm = llm
        self._snapshot: str | None = None
        self._session: TTSSession | None = None
        self._hold_buf: bytearray | None = None
        self._hold_lock: threading.Lock | None = None
        self._start_task: asyncio.Task | None = None

    def is_idle(self) -> bool:
        return self._session is None

    def snapshot(self) -> str | None:
        return self._snapshot

    def hold_size(self) -> int:
        if self._hold_buf is None or self._hold_lock is None:
            return 0
        with self._hold_lock:
            return len(self._hold_buf)

    def fire(self, snapshot: str) -> None:
        """Cancel any in-flight speculation and start a fresh one."""
        if self._session is not None:
            self._cancel_in_flight()

        log.info("🔮 spec fire on %r (%d chars)", snapshot, len(snapshot))
        self._snapshot   = snapshot
        self._hold_buf   = bytearray()
        self._hold_lock  = threading.Lock()
        self._session    = TTSSession(self.voice)
        self._session.set_sink((self._hold_buf, self._hold_lock))

        messages_override = list(self.llm.history) + [
            {"role": "user", "content": snapshot}
        ]
        text_stream = self.llm.generate_stream(
            messages_override=messages_override, tag="spec",
        )
        self._start_task = asyncio.create_task(self._session.start(text_stream))

    def _cancel_in_flight(self) -> None:
        """Fire-and-forget cleanup of the previous speculation."""
        if self._session is None:
            return
        old = self._session
        # Detach state first so a re-fire that follows can install fresh state.
        self._snapshot  = None
        self._session   = None
        self._hold_buf  = None
        self._hold_lock = None
        self._start_task = None
        asyncio.create_task(old.close(cancelled=True))

    def discard(self) -> None:
        if self._session is None:
            return
        log.info("💨 spec discarded")
        self._cancel_in_flight()

    async def release_to_speaker(self, speaker: SpeakerOutput) -> TTSSession:
        """Flush the hold buffer into the speaker, swap the live sink,
        and hand the session back to the caller. After this call,
        the manager is idle. Caller is responsible for awaiting the
        session's completion and closing it."""
        assert self._session is not None
        log.info("✨ spec hit (releasing %d bytes held)", self.hold_size())

        # Drain hold buffer → speaker buffer.
        spk_buf, spk_lock = speaker.sink()
        with self._hold_lock:
            data = bytes(self._hold_buf)
            self._hold_buf.clear()
        with spk_lock:
            spk_buf.extend(data)

        # Live session now writes directly to the speaker.
        self._session.set_sink((spk_buf, spk_lock))

        session = self._session
        self._snapshot   = None
        self._session    = None
        self._hold_buf   = None
        self._hold_lock  = None
        self._start_task = None
        return session

    def check_safety_cap(self) -> None:
        """Bail out of a runaway speculation that's holding too much
        audio (LLM error path, server stuck, etc.)."""
        if self._session is None:
            return
        if self.hold_size() > SPECULATION_MAX_HOLD_BYTES:
            log.warning("speculation hold buffer exceeded %d bytes — discarding",
                        SPECULATION_MAX_HOLD_BYTES)
            self._cancel_in_flight()


# ── Turn-event helpers ─────────────────────────────────────────

async def wait_speech_onset(asr: ASRStream, min_chars: int) -> None:
    if await asr.current_text_len() >= min_chars:
        return
    while True:
        await asr.events.get()
        if await asr.current_text_len() >= min_chars:
            return


async def collect_turn_with_speculation(
    asr: ASRStream,
    spec_mgr: SpeculationManager | None,
    stability: float,
    turn_min_chars: int,
    spec_min_chars: int,
    spec_min_growth: int,
) -> str:
    """Collect deltas until done / stability. While collecting, fire
    speculations on transcript snapshots whenever the text has grown
    by enough since the last fire. Returns the final transcript."""
    last_spec_len = 0
    while True:
        try:
            evt = await asyncio.wait_for(asr.events.get(), timeout=stability)
        except asyncio.TimeoutError:
            if await asr.current_text_len() >= turn_min_chars:
                return await asr.take_transcript()
            continue

        kind, _ = evt
        if kind == "done":
            txt = await asr.take_transcript()
            if len(txt) >= turn_min_chars:
                return txt
            continue

        # delta — maybe fire speculation
        if spec_mgr is None:
            continue
        spec_mgr.check_safety_cap()
        cur_len = await asr.current_text_len()
        if cur_len < spec_min_chars:
            continue
        if cur_len - last_spec_len < spec_min_growth:
            continue
        snapshot = await asr.peek_text()
        spec_mgr.fire(snapshot)
        last_spec_len = cur_len


# ── Reply playback (race session-done vs onset) ────────────────

async def play_reply(
    session: TTSSession,
    asr: ASRStream,
    speaker: SpeakerOutput,
) -> tuple[bool, str]:
    """Wait for actual playback to finish OR the user to barge in.
    Returns (completed, full_text). Cleans up the session either way.

    Opens the speaker stream on entry and closes it before returning
    (per-reply lifecycle required for AEC routing). Arms an AEC warmup
    gate at the same moment, so residual echo from the speaker spinning
    up doesn't trigger a false barge-in while AEC3 converges.

    NB: session.done fires when the TTS *server* is done generating —
    often well before the speaker has played the buffered audio. We
    need barge-in detection alive for the entire playback span, not
    just the streaming span. So race onset against the full pipeline:
    server-done THEN buffer-drain."""
    speaker.start()
    asr.gate_for(AEC_WARMUP)
    try:
        asr.mark_tts_started()
        onset_task = asyncio.create_task(wait_speech_onset(asr, BARGEIN_MIN_CHARS))

        async def _full_playback():
            await session.done.wait()
            await speaker.drain()

        playback_task = asyncio.create_task(_full_playback())
        try:
            finished, _ = await asyncio.wait(
                [onset_task, playback_task], return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            if not playback_task.done():
                playback_task.cancel()
                try:
                    await playback_task
                except (asyncio.CancelledError, Exception):
                    pass

        if onset_task in finished:
            log.info("⚡ interrupted")
            # 1. Stop the receiver from writing more chunks (sync flag)
            #    BEFORE we touch the speaker buffer — otherwise a chunk in
            #    flight would land after the fade and defeat it.
            # 2. Replace the speaker buffer head with a short linear
            #    fade-out instead of a hard clear, so cutoff doesn't click
            #    or hiss on mid-word sibilants.
            # 3. Then async-cleanup the session.
            session.cancel_writes()
            speaker.clear_with_fadeout()
            log.info("🔇 audio cut")
            await session.close(cancelled=True)
            asr.unmark_tts_started()
            return (False, session.full_text)

        # Playback completed naturally (server done + buffer drained).
        onset_task.cancel()
        try:
            await onset_task
        except asyncio.CancelledError:
            pass
        await session.close(cancelled=False)
        asr.gate_for(GRACE_AFTER_TTS)
        await asr.drain_events()
        asr.unmark_tts_started()
        return (True, session.full_text)
    finally:
        speaker.stop()


# ── Main loop ──────────────────────────────────────────────────

async def _mic_pump(mic: MicCapture, asr: ASRStream):
    try:
        while True:
            chunk = await mic.next_chunk()
            await asr.send_audio(chunk)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        log.error("mic pump crashed: %s", e)
        raise


async def run(
    mic_device: int | str | None,
    speaker_device: int | str | None,
    voice: str,
    speculation: bool,
    system_prompt: str,
):
    mic     = MicCapture(mic_device)
    asr     = ASRStream()
    llm     = LLMClient(system_prompt)
    speaker = SpeakerOutput(speaker_device)
    spec_mgr = SpeculationManager(voice, llm) if speculation else None

    await asr.connect()
    await mic.start()
    # Speaker stream is opened per-reply inside play_reply (AEC routing
    # requires it). The persistent buffer + lock are already alive.
    asr_task  = asyncio.create_task(asr.receive_loop())
    pump_task = asyncio.create_task(_mic_pump(mic, asr))

    log.info("━" * 56)
    log.info("  mistral-edge-voice — full-duplex voice agent")
    log.info("  Mic         : %s", _device_label(mic_device, "input"))
    log.info("  Speaker     : %s", _device_label(speaker_device, "output"))
    log.info("  Voice       : %s", voice)
    log.info("  Speculation : %s", "ON" if speculation else "OFF")
    log.info("  AEC3 is active; open speakers OK.")
    log.info("  Ctrl+C to quit.")
    log.info("━" * 56)

    asr.gate_for(GRACE_AFTER_TURN)
    await asr.drain_events()

    try:
        while True:
            # 1. Wait for the user to start speaking.
            await wait_speech_onset(asr, BARGEIN_MIN_CHARS)

            # 2. Collect the turn while opportunistically speculating.
            transcript = await collect_turn_with_speculation(
                asr, spec_mgr,
                END_OF_TURN_STABILITY,
                TURN_MIN_CHARS,
                SPECULATION_MIN_CHARS,
                SPECULATION_MIN_GROWTH,
            )
            asr.gate_for(GRACE_AFTER_TURN)
            await asr.drain_events()

            log.info("🧑 you   : %s", transcript)
            llm.add_user(transcript)

            # 3. Decide: speculation hit, miss, or no speculation at all.
            session: TTSSession | None = None
            if spec_mgr is not None and not spec_mgr.is_idle():
                snap = spec_mgr.snapshot() or ""
                if _normalize_for_match(snap) == _normalize_for_match(transcript):
                    session = await spec_mgr.release_to_speaker(speaker)
                else:
                    log.info("spec miss: snap=%r final=%r", snap, transcript)
                    spec_mgr.discard()

            if session is None:
                # Fresh reply — open a new TTS session piped straight to speaker.
                session = TTSSession(voice)
                session.set_sink(speaker.sink())
                text_stream = llm.generate_stream()
                await session.start(text_stream)

            # 4. Play / race against barge-in.
            completed, full_text = await play_reply(session, asr, speaker)

            log.info("🤖 agent : %s", full_text)
            # Persist whatever the LLM generated, even on interrupt. Without
            # this, every interrupted reply is wiped from history and the
            # LLM has no memory of what it tried to say — leading to the
            # exact same answer being regenerated next turn. The user heard
            # less than full_text (audio was cut), but giving the LLM the
            # full generation is the lesser of two evils vs. repetition.
            if full_text:
                llm.add_assistant(full_text)

    except KeyboardInterrupt:
        log.info("Shutting down …")
    finally:
        mic.stop()
        speaker.stop()
        if spec_mgr is not None:
            spec_mgr.discard()
        await asr.close()
        await llm.aclose()
        pump_task.cancel()
        asr_task.cancel()


# ── CLI ────────────────────────────────────────────────────────

def _device_arg(s: str) -> int | str:
    """argparse type: integer if it parses, otherwise the raw string."""
    try:
        return int(s)
    except ValueError:
        return s


def _load_prompt(path_str: str) -> str:
    """Load the system prompt from a text file. Relative paths resolve
    against the directory containing this script, so `python voice_agent.py`
    works from any cwd."""
    path = Path(path_str)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path.read_text(encoding="utf-8").strip()


def main():
    p = argparse.ArgumentParser(description="mistral-edge-voice — full-duplex voice agent.")
    p.add_argument(
        "--mic", type=_device_arg, default=None,
        help=("Mic device (int index or name substring). Default: system "
              "default, which the AEC module installer points at "
              f"{EC_SOURCE_NAME!r} for the script's lifetime, so audio "
              "routes through WebRTC AEC3 automatically."),
    )
    p.add_argument(
        "--speaker", type=_device_arg, default=None,
        help=("Speaker device (int index or name substring). Default: "
              "system default — pointed at "
              f"{EC_SINK_NAME!r} by the AEC module installer."),
    )
    p.add_argument(
        "--voice", default="neutral_female",
        help=("Voxtral TTS voice — also picks the spoken language. "
              "Examples: neutral_female (English), de_female/de_male (German), "
              "fr_female, es_male, it_female, nl_male, pt_female, hi_male, ar_male, "
              "casual_male, cheerful_female. Full list: "
              "`curl http://localhost:8003/v1/audio/voices`."),
    )
    p.add_argument(
        "--no-speculation", action="store_true",
        help="Disable speculation. Use as an A/B baseline against the default speculative flow.",
    )
    p.add_argument(
        "--prompt-file", default=DEFAULT_PROMPT_FILE,
        help=("Path to a text file containing the system prompt. Relative "
              f"paths resolve to the directory of this script. Default: {DEFAULT_PROMPT_FILE}."),
    )
    args = p.parse_args()

    try:
        system_prompt = _load_prompt(args.prompt_file)
    except OSError as e:
        log.error("Could not load prompt file %r: %s", args.prompt_file, e)
        return
    if not system_prompt:
        log.error("Prompt file %r is empty.", args.prompt_file)
        return

    aec = EchoCancelModule()
    try:
        aec.load()
    except RuntimeError as e:
        log.error("Failed to load echo-cancel module: %s", e)
        return

    try:
        asyncio.run(run(
            args.mic, args.speaker, args.voice,
            speculation=not args.no_speculation,
            system_prompt=system_prompt,
        ))
    except KeyboardInterrupt:
        pass
    finally:
        aec.unload()


if __name__ == "__main__":
    main()
