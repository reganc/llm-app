"""Speech to text for Telegram voice notes: local faster-whisper, CPU only.

The shared GPU stays with Ollama. The model loads on the first voice note and
stays loaded; it is downloaded once into DATA_DIR/whisper (the app_data
volume). One transcription at a time, off the event loop. Transcripts are the
user's words: never logged, like any message text.

Model choice (2026-11-26, i5-14600K, 8 threads, int8, an 8.9 s Ogg/Opus clip
with names and tickers): base.en 0.6 s but "Hause my morning", "men to";
small.en 1.5 s, "Ment at"; large-v3-turbo 5.1 s, every word right. Turbo is
the default (about half real time); WHISPER_MODEL=small.en trades accuracy
for speed.
"""
from __future__ import annotations

import asyncio
import io
import logging
import threading

from config import CFG, DATA_DIR

log = logging.getLogger("llm-api.voice")

MAX_SECONDS = 300           # longer notes are refused before download
_model = None
_load_lock = threading.Lock()
_one_at_a_time = asyncio.Semaphore(1)


class VoiceError(Exception):
    """A voice note that can't be transcribed; the text says why."""


def _load():
    global _model
    if _model is None:
        with _load_lock:
            if _model is None:
                from faster_whisper import WhisperModel   # heavy import, deferred
                _model = WhisperModel(CFG.whisper_model, device="cpu", compute_type="int8",
                                      cpu_threads=CFG.whisper_threads,
                                      download_root=str(DATA_DIR / "whisper"))
                log.info("whisper loaded: %s", CFG.whisper_model)
    return _model


def _transcribe_sync(audio: bytes) -> str:
    segments, _info = _load().transcribe(io.BytesIO(audio), beam_size=1, vad_filter=True)
    return " ".join(s.text.strip() for s in segments).strip()


async def transcribe(audio: bytes) -> str:
    """The words in a voice note. VoiceError when it can't be read or has no speech."""
    if not CFG.voice_enabled:
        raise VoiceError("Voice notes are switched off (VOICE_ENABLED=false).")
    async with _one_at_a_time:
        try:
            text = await asyncio.to_thread(_transcribe_sync, audio)
        except Exception as e:          # never the audio or its text in a log line
            log.warning("transcription failed: %s", type(e).__name__)
            raise VoiceError("I couldn't make out that voice note.") from None
    if not text:
        raise VoiceError("I didn't hear any speech in that voice note.")
    return text
