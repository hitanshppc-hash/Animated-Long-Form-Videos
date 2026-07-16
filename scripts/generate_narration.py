import argparse
import os
from pathlib import Path

import requests

from utils import get_logger, retry

logger = get_logger(__name__)

DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # George - Warm, Captivating Storyteller
TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"


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


def _fallback_gtts(text: str, output_path: str) -> None:
    from gtts import gTTS

    logger.info("Falling back to local gTTS narration")
    tts = gTTS(text=text)
    tts.save(output_path)


def generate_narration(text: str, output_path: str, voice_id: str = DEFAULT_VOICE_ID) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get("ELEVENLABS_API_KEY")

    if api_key:
        try:
            audio_bytes = _call_elevenlabs(text, voice_id, api_key)
            with open(output_path, "wb") as f:
                f.write(audio_bytes)
            logger.info(f"Wrote narration audio (ElevenLabs): {output_path} ({len(audio_bytes) / 1024:.0f} KB)")
            return
        except Exception as exc:
            logger.warning(f"ElevenLabs narration failed: {exc}")
    else:
        logger.warning("ELEVENLABS_API_KEY not set, using local fallback")

    _fallback_gtts(text, output_path)
    logger.info(f"Wrote narration audio (gTTS fallback): {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate narration audio from text (ElevenLabs, falls back to local gTTS)")
    parser.add_argument("--text", required=True, help="Narration text to speak")
    parser.add_argument("--output", required=True, help="Path to write the output audio file")
    parser.add_argument("--voice-id", default=DEFAULT_VOICE_ID, help="ElevenLabs voice id")
    args = parser.parse_args()

    generate_narration(args.text, args.output, args.voice_id)


if __name__ == "__main__":
    main()
