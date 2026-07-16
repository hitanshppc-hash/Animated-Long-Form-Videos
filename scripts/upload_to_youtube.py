import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from youtube_upload import upload


def main():
    title = os.environ.get("YT_VIDEO_TITLE", "").strip()
    if not title:
        meta_path = "output/final_video.metadata.json"
        if os.path.exists(meta_path):
            meta = json.load(open(meta_path))
            title = meta.get("title", "Untitled Video")
        else:
            title = "Untitled Video"

    desc = os.environ.get("YT_VIDEO_DESC", "").strip()
    if not desc:
        desc = "Generated with the Animated Long-Form Video Pipeline"

    tags_input = os.environ.get("YT_VIDEO_TAGS", "").strip()
    tags = [t.strip() for t in tags_input.split(",") if t.strip()] if tags_input else ["animation", "long-form", "ai-generated"]

    privacy = os.environ.get("YT_PRIVACY", "public").strip().lower()

    url = upload(
        "output/final_video.mp4",
        title=title,
        description=desc,
        tags=tags,
        privacy=privacy,
    )
    print(f"Uploaded: {url}")


if __name__ == "__main__":
    main()
