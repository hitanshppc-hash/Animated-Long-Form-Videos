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
    model: str | None = None,
    negative_prompt: str | None = None,
    num_frames: int | None = None,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
    seed: int | None = None,
    dry_run: bool = False,
    clip_duration: float = 12.0,
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

    short_query = extract_keywords(prompt, max_words=5)
    logger.info(f"Fetching stock video for: {short_query!r} (from {len(prompt)} chars)")
    fetch_stock_video(short_query, output_path, max_duration=clip_duration)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one video clip from a seed image and prompt")
    parser.add_argument("--image", required=True, help="Path to the seed image")
    parser.add_argument("--prompt", required=True, help="Text prompt describing the motion/scene")
    parser.add_argument("--output", required=True, help="Path to write the output .mp4")
    parser.add_argument("--model", help="Preferred Hugging Face model id to try first")
    parser.add_argument("--negative-prompt")
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--num-inference-steps", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--clip-duration", type=float, default=12.0, help="Seconds per clip")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without calling the API")
    args = parser.parse_args()

    generate_clip(
        args.image,
        args.prompt,
        args.output,
        args.model,
        negative_prompt=args.negative_prompt,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        dry_run=args.dry_run,
        clip_duration=args.clip_duration,
    )


if __name__ == "__main__":
    main()
