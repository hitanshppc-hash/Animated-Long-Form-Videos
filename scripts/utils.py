import logging
import time
from functools import wraps


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
