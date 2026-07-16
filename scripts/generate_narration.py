import argparse
import os
import re
import shutil
import subprocess
import asyncio
from pathlib import Path

import requests

from utils import get_logger, retry

logger = get_logger(__name__)

DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # George - Warm, Captivating Storyteller
TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

TONE_VOICE_MAP = {
    "expository": "en-US-AriaNeural",
    "dialogue": "en-US-JennyNeural",
    "tension": "en-US-AriaNeural",
    "action": "en-US-AriaNeural",
    "wonder": "en-GB-SoniaNeural",
    "mystery": "en-US-MichelleNeural",
    "sadness": "en-US-JennyNeural",
    "joy": "en-US-AnaNeural",
}

TONE_VOICE_DESCRIPTION = {
    "expository": "Warm, clear — world-building",
    "dialogue": "Natural, expressive — conversation",
    "tension": "Focused, urgent — danger",
    "action": "Energetic — high motion",
    "wonder": "Soft, elegant — awe, discovery",
    "mystery": "Crisp — suspense",
    "sadness": "Gentle, emotional — melancholy",
    "joy": "Bright, youthful — happy",
}


def voice_for_tone(tone: str) -> str:
    return TONE_VOICE_MAP.get(tone, "en-US-AriaNeural")


# ------------------------------------------------------------------- helpers
def _words_of(text):
    return re.findall(r"[^\s]+", text.strip())


def _estimate_timings(text, duration):
    words = _words_of(text)
    if not words:
        return []
    total = sum(len(w) + 2 for w in words)
    out, t = [], 0.0
    for w in words:
        d = duration * (len(w) + 2) / total
        out.append((w, t, t + d))
        t += d
    return out


def _probe(path):
    import json as _json
    raw = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", path])
    return float(_json.loads(raw)["format"]["duration"])


# ---------------------------------------------------------------- ElevenLabs
@retry(attempts=3, base_delay=5.0)
def _call_elevenlabs(text: str, voice_id: str, api_key: str) -> bytes:
    response = requests.post(
        TTS_URL.format(voice_id=voice_id),
        headers={"xi-api-key": api_key, "Content-Type": "application/json"},
        json={
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        },
        timeout=120,
    )
    if response.status_code != 200:
        raise RuntimeError(f"ElevenLabs HTTP {response.status_code}: {response.text[:200]}")
    return response.content


def _elevenlabs(text, voice_id, output_path):
    audio_bytes = _call_elevenlabs(text, voice_id, os.environ["ELEVENLABS_API_KEY"])
    with open(output_path, "wb") as f:
        f.write(audio_bytes)
    logger.info(f"Wrote narration audio (ElevenLabs): {output_path} ({len(audio_bytes) / 1024:.0f} KB)")
    return output_path, _estimate_timings(text, _probe(output_path))


# ------------------------------------------------------------------- Kokoro
_KOKORO_VOICE_MAP = {
    "en-US-AriaNeural": "af_heart", "en-US-JennyNeural": "af_bella",
    "en-US-GuyNeural": "am_adam", "en-GB-SoniaNeural": "af_sarah",
    "en-GB-RyanNeural": "am_michael", "en-US-AnaNeural": "af_nova",
    "en-US-MichelleNeural": "af_sky", "en-US-SteffanNeural": "am_liam",
}


def _kokoro(text, voice, output_path):
    import numpy as _np
    import soundfile as _sf
    from local_models import ensure_kokoro

    mapped = _KOKORO_VOICE_MAP.get(voice, "af_heart")
    pipeline = ensure_kokoro()
    gen = pipeline(text, voice=mapped, speed=1, split_pattern=r"\n+")
    chunks = []
    for _gs, _ps, audio in gen:
        chunks.append(audio)
    if not chunks:
        raise RuntimeError("kokoro produced no audio")
    full = _np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    wav = output_path + ".wav"
    _sf.write(wav, full, 24000)
    subprocess.run(["ffmpeg", "-y", "-i", wav, "-b:a", "192k", output_path],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.remove(wav)
    logger.info(f"Wrote narration audio (Kokoro): {output_path}")
    return output_path, _estimate_timings(text, _probe(output_path))


# ---------------------------------------------------------------- edge-tts
async def _edge_async(text, voice, out_path):
    import edge_tts
    words = []
    try:
        communicate = edge_tts.Communicate(text, voice, boundary="WordBoundary")
    except TypeError:
        communicate = edge_tts.Communicate(text, voice)
    with open(out_path, "wb") as f:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                start = chunk["offset"] / 1e7
                end = (chunk["offset"] + chunk["duration"]) / 1e7
                words.append((chunk["text"], start, end))
    if not words or os.path.getsize(out_path) < 1024:
        raise RuntimeError("edge-tts produced no audio")
    return words


def _edge_tts(text, voice, output_path):
    words = asyncio.run(_edge_async(text, voice, output_path))
    logger.info(f"Wrote narration audio (edge-tts): {output_path}")
    return output_path, words


# ------------------------------------------------------------------ espeak-ng
def _espeak(text, voice, output_path):
    exe = shutil.which("espeak-ng") or shutil.which("espeak") or ""
    if not exe:
        raise RuntimeError("espeak-ng not installed")
    wav = output_path + ".wav"
    subprocess.run([exe, "-v", "en-us", "-s", "165", "-w", wav, text], check=True)
    subprocess.run(["ffmpeg", "-y", "-i", wav, "-b:a", "192k", output_path],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.remove(wav)
    logger.info(f"Wrote narration audio (espeak-ng): {output_path}")
    return output_path, _estimate_timings(text, _probe(output_path))


# ---------------------------------------------------------------------- gTTS
def _gtts(text, voice, output_path):
    from gtts import gTTS
    tts = gTTS(text=text)
    tts.save(output_path)
    logger.info(f"Wrote narration audio (gTTS): {output_path}")
    return output_path, _estimate_timings(text, _probe(output_path))


# ------------------------------------------------------------------ fallback chain
def generate_narration(text: str, output_path: str, voice_id: str = DEFAULT_VOICE_ID) -> str:
    """Generate narration audio. Returns path to audio file.

    Fallback chain: ElevenLabs → Kokoro → edge-tts → gTTS → espeak-ng.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    chain = []

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if api_key:
        chain.append(("elevenlabs", lambda: _elevenlabs(text, voice_id, output_path)))

    chain += [
        ("kokoro", lambda: _kokoro(text, voice_id, output_path)),
        ("edge-tts", lambda: _edge_tts(text, voice_id, output_path)),
        ("gtts", lambda: _gtts(text, voice_id, output_path)),
        ("espeak-ng", lambda: _espeak(text, voice_id, output_path)),
    ]

    last_err = None
    for name, fn in chain:
        try:
            path, _words = fn()
            return path
        except Exception as e:
            logger.warning(f"TTS provider {name} failed: {e}")
            last_err = e

    raise RuntimeError(f"All TTS providers failed: {last_err}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate narration audio from text (ElevenLabs → Kokoro → edge-tts → gTTS → espeak-ng)")
    parser.add_argument("--text", required=True, help="Narration text to speak")
    parser.add_argument("--output", required=True, help="Path to write the output audio file")
    parser.add_argument("--voice-id", default=DEFAULT_VOICE_ID, help="ElevenLabs voice id")
    args = parser.parse_args()

    generate_narration(args.text, args.output, args.voice_id)


if __name__ == "__main__":
    main()
