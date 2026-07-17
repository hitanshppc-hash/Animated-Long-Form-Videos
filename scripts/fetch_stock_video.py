import argparse
import os
from pathlib import Path

import requests

from utils import get_logger

logger = get_logger(__name__)

PEXELS_VIDEO_SEARCH_URL = "https://api.pexels.com/videos/search"

# Canonical output resolution/fps for every clip. Pexels returns landscape
# results at all kinds of widths/heights/framerates depending on the source
# video, and merging clips of mismatched resolution is what was causing the
# xfade crossfade filter to fail on every run (falling back to hard cuts) and
# made caption sizing/position unreliable. Normalizing every clip to the same
# 1080p 16:9 frame here fixes both, and matches standard YouTube upload specs.
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
TARGET_FPS = 24


def _pick_file(video: dict, target_width: int = TARGET_WIDTH) -> dict:
    files = video.get("video_files", [])
    mp4_files = [f for f in files if f.get("file_type") == "video/mp4" and f.get("width")]
    if not mp4_files:
        raise RuntimeError("No usable mp4 files in Pexels video result")
    return min(mp4_files, key=lambda f: abs(f["width"] - target_width))


def fetch_stock_video(query: str, output_path: str, max_duration: float = 12.0) -> None:
    api_key = os.environ.get("PEXELS_API_KEY")
    if not api_key:
        raise RuntimeError("PEXELS_API_KEY environment variable is not set")

    response = requests.get(
        PEXELS_VIDEO_SEARCH_URL,
        headers={"Authorization": api_key},
        params={"query": query, "per_page": 5, "orientation": "landscape"},
        timeout=30,
    )
    response.raise_for_status()
    videos = response.json().get("videos") or []
    if not videos:
        raise RuntimeError(f"No Pexels stock video results for query: {query!r}")

    file_info = _pick_file(videos[0])
    raw_path = str(Path(output_path).with_suffix(".raw.mp4"))

    video_resp = requests.get(file_info["link"], timeout=60)
    video_resp.raise_for_status()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(raw_path).write_bytes(video_resp.content)

    from moviepy import VideoFileClip

    with VideoFileClip(raw_path) as clip:
        trimmed = clip.subclipped(0, min(max_duration, clip.duration))

        # Scale to cover the canonical frame, then center-crop, so every clip
        # ends up at the exact same WxH regardless of the source aspect ratio.
        scale = max(TARGET_WIDTH / trimmed.w, TARGET_HEIGHT / trimmed.h)
        resized = trimmed.resized(scale)
        normalized = resized.cropped(
            width=TARGET_WIDTH,
            height=TARGET_HEIGHT,
            x_center=resized.w / 2,
            y_center=resized.h / 2,
        )
        normalized.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=TARGET_FPS,
            logger=None,
        )

    Path(raw_path).unlink(missing_ok=True)
    logger.info(f"Wrote stock video fallback ({query!r}) to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a real stock video clip from Pexels as a video-generation fallback")
    parser.add_argument("--query", required=True, help="Search query describing the desired clip")
    parser.add_argument("--output", required=True, help="Path to write the output .mp4")
    parser.add_argument("--max-duration", type=float, default=12.0, help="Trim the clip to at most this many seconds")
    args = parser.parse_args()

    fetch_stock_video(args.query, args.output, args.max_duration)


if __name__ == "__main__":
    main()
