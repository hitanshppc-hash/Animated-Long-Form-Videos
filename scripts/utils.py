import logging
import time
from functools import wraps
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def retry(attempts: int = 4, base_delay: float = 5.0, exceptions: tuple = (Exception,)):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            last_exc = None
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == attempts:
                        break
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt}/{attempts}): {exc}. Retrying in {delay:.0f}s..."
                    )
                    time.sleep(delay)
            raise last_exc

        return wrapper

    return decorator


_THUMBNAIL_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def make_thumbnail(image_path: str, title: str, output_path: str, size: tuple = (1280, 720)) -> None:
    """Overlay the video title on the seed frame: a dark bottom band with
    bold wrapped text, instead of shipping the bare frame as-is."""
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    img = ImageOps.fit(Image.open(image_path).convert("RGB"), size, Image.LANCZOS)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    band_height = int(img.height * 0.32)
    draw.rectangle([(0, img.height - band_height), (img.width, img.height)], fill=(0, 0, 0, 165))

    font = None
    for candidate in _THUMBNAIL_FONT_CANDIDATES:
        if Path(candidate).exists():
            font = ImageFont.truetype(candidate, 64)
            break
    if font is None:
        font = ImageFont.load_default(size=64)

    text = (title or "").strip().upper()
    max_width = img.width - 100
    words = text.split()
    lines = []
    cur = ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if not cur or draw.textlength(trial, font=font) <= max_width:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    lines = lines[:3]

    line_height = font.size + 14
    y = img.height - 20 - line_height * len(lines)
    for line in lines:
        x = (img.width - draw.textlength(line, font=font)) / 2
        draw.text((x, y), line, font=font, fill=(255, 255, 255, 255))
        y += line_height

    Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB").save(output_path, quality=92)


def video_duration(path: str) -> float:
    import json as _json
    import shutil as _shutil
    import subprocess as _sub
    probe = _shutil.which("ffprobe") or _shutil.which("ffmpeg")
    if not probe:
        from moviepy import VideoFileClip
        with VideoFileClip(path) as clip:
            return clip.duration
    if "ffprobe" in probe:
        raw = _sub.check_output(
            [probe, "-v", "error", "-show_entries", "format=duration",
             "-of", "json", path]
        )
        return float(_json.loads(raw)["format"]["duration"])
    result = _sub.run([probe, "-i", path, "-f", "null", "-"],
                      capture_output=True, text=True)
    for line in result.stderr.split("\n"):
        if "Duration" in line:
            dur = line.split("Duration: ")[1].split(",")[0].strip()
            h, m, s = dur.split(":")
            return float(h) * 3600 + float(m) * 60 + float(s)
    raise RuntimeError(f"Could not probe duration for {path}")
