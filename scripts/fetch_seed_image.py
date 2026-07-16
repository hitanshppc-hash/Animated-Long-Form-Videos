import argparse
import os
from pathlib import Path

import requests

from utils import get_logger

logger = get_logger(__name__)

PEXELS_SEARCH_URL = "https://api.pexels.com/v1/search"


def _search(api_key: str, query: str, orientation: str | None) -> list:
    params = {"query": query, "per_page": 10}
    if orientation:
        params["orientation"] = orientation

    response = requests.get(
        PEXELS_SEARCH_URL,
        headers={"Authorization": api_key},
        params=params,
        timeout=30,
    )
    response.raise_for_status()
    return response.json().get("photos") or []


def _placeholder_image(query: str, output_path: str) -> None:
    from PIL import Image, ImageDraw

    logger.warning(f"Falling back to a generated placeholder image for {query!r}")
    img = Image.new("RGB", (1024, 576), color=(30, 34, 42))
    draw = ImageDraw.Draw(img)
    text = query[:60]
    draw.text((40, 260), text, fill=(220, 220, 220))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def fetch_seed_image(query: str, output_path: str, orientation: str = "landscape") -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get("PEXELS_API_KEY")

    if api_key:
        try:
            # per_page=1 combined with orientation can return an empty page even
            # when total_results is nonzero, so ask for more candidates and fall
            # back to an unfiltered search if the orientation-filtered one is empty.
            photos = _search(api_key, query, orientation)
            if not photos:
                logger.warning(f"No {orientation} results for {query!r}, retrying without orientation filter")
                photos = _search(api_key, query, None)

            if photos:
                image_url = photos[0]["src"]["large"]
                image_resp = requests.get(image_url, timeout=30)
                image_resp.raise_for_status()
                Path(output_path).write_bytes(image_resp.content)
                logger.info(f"Wrote seed image ({query!r}) to {output_path}")
                return
            logger.warning(f"No Pexels results at all for {query!r}")
        except Exception as exc:
            logger.warning(f"Pexels fetch failed: {exc}")
    else:
        logger.warning("PEXELS_API_KEY not set, using placeholder image")

    _placeholder_image(query, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a seed image from Pexels for a text query")
    parser.add_argument("--query", required=True, help="Search query describing the desired image")
    parser.add_argument("--output", required=True, help="Path to write the downloaded image")
    parser.add_argument("--orientation", default="landscape", choices=["landscape", "portrait", "square"])
    args = parser.parse_args()

    fetch_seed_image(args.query, args.output, args.orientation)


if __name__ == "__main__":
    main()
