import argparse
from pathlib import Path

from moviepy import VideoFileClip

from utils import get_logger

logger = get_logger(__name__)


def extract_last_frame(video_path: str, output_path: str) -> None:
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with VideoFileClip(video_path) as clip:
        t = max(0.0, clip.duration - 1.0 / clip.fps)
        clip.save_frame(output_path, t=t)

    logger.info(f"Wrote frame: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract the last frame of a video as an image")
    parser.add_argument("--video", required=True, help="Path to the input video")
    parser.add_argument("--output", required=True, help="Path to write the extracted frame (.png)")
    args = parser.parse_args()

    extract_last_frame(args.video, args.output)


if __name__ == "__main__":
    main()
