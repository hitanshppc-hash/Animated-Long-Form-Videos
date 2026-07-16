import argparse
import os
from pathlib import Path

from huggingface_hub import InferenceClient

from utils import get_logger, retry

logger = get_logger(__name__)

DEFAULT_MODEL = "Wan-AI/Wan2.1-I2V-14B-480P"

# (model, provider) pairs tried in order until one succeeds. No genuinely
# small (<5B) image-to-video model is currently live on Inference Providers —
# stable-video-diffusion-img2vid-xt and the original LTX-Video have no live
# provider mapping, and every current-generation I2V model clusters around
# 13-19B for usable motion quality. The 480P Wan2.1 variant is the cheapest/
# fastest available (same params, lower output resolution = less compute per
# clip), so it's tried first; the rest are progressively heavier fallbacks.
VIDEO_FALLBACKS = [
    ("Wan-AI/Wan2.1-I2V-14B-480P", "wavespeed"),
    ("Wan-AI/Wan2.2-I2V-A14B", "fal-ai"),
    ("Wan-AI/Wan2.2-I2V-A14B", "together"),
    ("Wan-AI/Wan2.2-I2V-A14B", "wavespeed"),
    ("Wan-AI/Wan2.1-I2V-14B-720P", "fal-ai"),
    ("Wan-AI/Wan2.1-I2V-14B-720P", "wavespeed"),
    ("tencent/HunyuanVideo-I2V", "fal-ai"),
    ("Lightricks/LTX-2", "fal-ai"),
    ("Lightricks/LTX-2", "wavespeed"),
]


@retry(attempts=2, base_delay=5.0)
def _call_image_to_video(token: str, image_path: str, model: str, provider: str, prompt: str, **kwargs) -> bytes:
    client = InferenceClient(provider=provider, api_key=token)
    return client.image_to_video(image_path, model=model, prompt=prompt, **kwargs)


def _fallback_order(preferred_model: str | None) -> list:
    if not preferred_model:
        return VIDEO_FALLBACKS
    preferred = [pair for pair in VIDEO_FALLBACKS if pair[0] == preferred_model]
    if not preferred:
        preferred = [(preferred_model, "fal-ai")]
    rest = [pair for pair in VIDEO_FALLBACKS if pair[0] != preferred_model]
    return preferred + rest


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
) -> None:
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Seed image not found: {image_path}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        logger.info(f"[dry-run] Would generate clip: model={model or DEFAULT_MODEL} prompt={prompt!r} -> {output_path}")
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN environment variable is not set")

    extra = {
        "negative_prompt": negative_prompt,
        "num_frames": num_frames,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
    }
    extra = {k: v for k, v in extra.items() if v is not None}

    errors = []
    for candidate_model, provider in _fallback_order(model):
        try:
            logger.info(f"Requesting clip from {candidate_model} via {provider}: {prompt[:70]!r}")
            video_bytes = _call_image_to_video(token, image_path, candidate_model, provider, prompt, **extra)
            with open(output_path, "wb") as f:
                f.write(video_bytes)
            logger.info(
                f"Wrote clip via {candidate_model}/{provider}: {output_path} ({len(video_bytes) / 1024:.0f} KB)"
            )
            return
        except Exception as exc:
            logger.warning(f"{candidate_model}/{provider} failed: {exc}")
            errors.append(f"{candidate_model}/{provider}: {exc}")

    # Every AI video model/provider option is routed through the same HF
    # Inference Providers account, so a single billing/quota outage there
    # takes all of them out at once. Fall back to a real, independent Pexels
    # stock video clip (separate account) so the pipeline degrades instead
    # of hard-failing.
    if os.environ.get("PEXELS_API_KEY"):
        try:
            from fetch_stock_video import fetch_stock_video

            logger.warning("All AI video generation options failed, falling back to a Pexels stock clip")
            fetch_stock_video(prompt, output_path)
            return
        except Exception as exc:
            errors.append(f"pexels-stock-fallback: {exc}")

    raise RuntimeError("All video-generation options (AI models and stock fallback) failed:\n" + "\n".join(errors))


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
    )


if __name__ == "__main__":
    main()
