import argparse
import os
import re
import shutil
import subprocess
import asyncio
from pathlib import Path

from utils import get_logger

logger = get_logger(__name__)

DEFAULT_VOICE_ID = "en-US-AriaNeural"

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
    import shutil as _shutil
    exe = _shutil.which("ffprobe") or _shutil.which("ffmpeg") or ""
    if not exe:
        from moviepy import AudioFileClip
        with AudioFileClip(path) as clip:
            return clip.duration
    if "ffprobe" in exe:
        raw = subprocess.check_output(
            [exe, "-v", "error", "-show_entries", "format=duration",
             "-of", "json", path])
        return float(_json.loads(raw)["format"]["duration"])
    result = subprocess.run([exe, "-i", path, "-f", "null", "-"],
                            capture_output=True, text=True)
    for line in result.stderr.split("\n"):
        if "Duration" in line:
            dur = line.split("Duration: ")[1].split(",")[0].strip()
            h, m, s = dur.split(":")
            return float(h) * 3600 + float(m) * 60 + float(s)
    raise RuntimeError(f"Could not probe duration for {path}")


# ------------------------------------------------------------------- Kokoro
_KOKORO_VOICE_MAP = {
    "en-US-AriaNeural": "af_heart", "en-US-JennyNeural": "af_bella",
    "en-US-GuyNeural": "am_adam", "en-GB-SoniaNeural": "af_sarah",
    "en-GB-RyanNeural": "am_michael", "en-US-AnaNeural": "af_nova",
    "en-US-MichelleNeural": "af_sky", "en-US-SteffanNeural": "am_liam",
    "en-US-ChristopherNeural": "am_michael",
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


# A voice id that turns out to be wrong/deprecated/unavailable (like the
# previous NARRATOR_VOICE typo) shouldn't drop straight to gTTS and lose
# neural quality + word-timing data for every line using it — retry once
# against a known-good voice before giving up on edge-tts entirely.
_EDGE_TTS_SAFE_DEFAULT = "en-US-AriaNeural"


def _edge_tts(text, voice, output_path):
    try:
        words = asyncio.run(_edge_async(text, voice, output_path))
    except Exception as exc:
        if voice == _EDGE_TTS_SAFE_DEFAULT:
            raise
        logger.warning(
            f"edge-tts voice {voice!r} failed ({exc}); retrying with {_EDGE_TTS_SAFE_DEFAULT!r}"
        )
        words = asyncio.run(_edge_async(text, _EDGE_TTS_SAFE_DEFAULT, output_path))
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

    Fallback chain: Kokoro → edge-tts → gTTS → espeak-ng. voice_id is an
    edge-tts voice name (see generate_dialogue_audio.VOICE_POOL); Kokoro maps
    it through _KOKORO_VOICE_MAP (falling back to a default if it isn't one
    of Kokoro's own voices) and is only actually attempted if it's been
    installed locally — see requirements.txt.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    chain = [
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
    parser = argparse.ArgumentParser(description="Generate narration audio from text (Kokoro → edge-tts → gTTS → espeak-ng)")
    parser.add_argument("--text", required=True, help="Narration text to speak")
    parser.add_argument("--output", required=True, help="Path to write the output audio file")
    parser.add_argument("--voice-id", default=DEFAULT_VOICE_ID, help="edge-tts voice name (e.g. en-US-AriaNeural)")
    args = parser.parse_args()

    generate_narration(args.text, args.output, args.voice_id)


if __name__ == "__main__":
    main()
