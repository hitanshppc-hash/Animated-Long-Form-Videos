import argparse
import os
from pathlib import Path

from extract_keywords import extract_keywords
from utils import get_logger

logger = get_logger(__name__)


def generate_clip(
    image_path: str,
    prompt: str,
    output_path: str,
    dry_run: bool = False,
    clip_duration: float = 12.0,
    stock_query: str = "",
) -> None:
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Seed image not found: {image_path}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        logger.info(f"[dry-run] Would fetch stock video for prompt={prompt!r} -> {output_path}")
        return

    api_key = os.environ.get("PEXELS_API_KEY")
    if not api_key:
        raise RuntimeError("PEXELS_API_KEY environment variable is not set — video generation requires a stock video API key")

    from fetch_stock_video import fetch_stock_video

    # Prefer the LLM-authored stock_query (written for real-world stock
    # footage) over stripping keywords out of `prompt`, which is written as
    # an image/video-generation prompt (invented characters, fantastical
    # imagery) that a stock video library can't actually match.
    query = stock_query.strip() if stock_query and stock_query.strip() else extract_keywords(prompt, max_words=5)
    logger.info(f"Fetching stock video for: {query!r}")
    fetch_stock_video(query, output_path, max_duration=clip_duration)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one video clip from a seed image and prompt")
    parser.add_argument("--image", required=True, help="Path to the seed image")
    parser.add_argument("--prompt", required=True, help="Text prompt describing the motion/scene")
    parser.add_argument("--output", required=True, help="Path to write the output .mp4")
    parser.add_argument("--clip-duration", type=float, default=12.0, help="Seconds per clip")
    parser.add_argument("--stock-query", default="", help="Stock-footage search query (falls back to keywords extracted from --prompt)")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without calling the API")
    args = parser.parse_args()

    generate_clip(
        args.image,
        args.prompt,
        args.output,
        dry_run=args.dry_run,
        clip_duration=args.clip_duration,
        stock_query=args.stock_query,
    )


if __name__ == "__main__":
    main()
