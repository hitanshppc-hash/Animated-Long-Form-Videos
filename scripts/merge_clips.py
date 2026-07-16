import argparse
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List

from utils import get_logger

logger = get_logger(__name__)


def _probe_duration(path: str) -> float:
    import json as _json
    import shutil as _shutil
    probe = _shutil.which("ffprobe") or _shutil.which("ffmpeg")
    if not probe:
        from moviepy import VideoFileClip
        with VideoFileClip(path) as clip:
            return clip.duration
    if "ffprobe" in probe:
        raw = subprocess.check_output(
            [probe, "-v", "error", "-show_entries", "format=duration",
             "-of", "json", path]
        )
        return float(_json.loads(raw)["format"]["duration"])
    result = subprocess.run([probe, "-i", path, "-f", "null", "-"],
                            capture_output=True, text=True)
    for line in result.stderr.split("\n"):
        if "Duration" in line:
            dur = line.split("Duration: ")[1].split(",")[0].strip()
            h, m, s = dur.split(":")
            return float(h) * 3600 + float(m) * 60 + float(s)
    raise RuntimeError(f"Could not probe duration for {path}")


def _merge_concat_demuxer(clip_paths: List[str], output_path: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for p in clip_paths:
            f.write(f"file '{Path(p).resolve()}'\n")
        list_path = f.name

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", list_path, "-c", "copy", output_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    finally:
        os.unlink(list_path)

    logger.info(f"Merged {len(clip_paths)} clips (hard cuts) -> {output_path}")


def _merge_with_xfade(clip_paths: List[str], output_path: str, crossfade: float) -> None:
    durations = [_probe_duration(p) for p in clip_paths]
    n = len(clip_paths)

    input_args = []
    for p in clip_paths:
        input_args += ["-i", p]

    parts = []
    for i in range(1, n):
        v_in = f"[0:v]" if i == 1 else f"[v{i-1}]"
        v_lbl = f"vout" if i == n - 1 else f"v{i}"
        offset = sum(durations[:i]) - i * crossfade
        parts.append(f"{v_in}[{i}:v]xfade=transition=fade:duration={crossfade}:offset={offset}[{v_lbl}]")

    filter_complex = "; ".join(parts)

    result = subprocess.run(
        ["ffmpeg", "-y"] + input_args +
        ["-filter_complex", filter_complex,
         "-map", "[vout]", "-an",
         "-c:v", "libx264", output_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        err = result.stderr[:500]
        raise RuntimeError(f"ffmpeg xfade failed (rc={result.returncode}): {err}")

    total = sum(durations) - crossfade * (n - 1)
    logger.info(f"Merged {n} clips ({crossfade}s crossfade, no audio) -> {output_path} ({total:.1f}s)")


def merge_clips(clip_paths: List[str], output_path: str, crossfade_duration: float = 0.0) -> None:
    for clip in clip_paths:
        if not Path(clip).exists():
            raise FileNotFoundError(f"Clip not found: {clip}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    crossfade = min(crossfade_duration, 2.0)
    if crossfade > 0 and len(clip_paths) > 1:
        try:
            _merge_with_xfade(clip_paths, output_path, crossfade)
            return
        except Exception as exc:
            logger.warning(f"xfade merge failed ({exc}), falling back to hard cuts")
    _merge_concat_demuxer(clip_paths, output_path)


def attach_narration(video_path: str, audio_path: str, output_path: str) -> None:
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    video_dur = _probe_duration(video_path)
    audio_dur = _probe_duration(audio_path)

    if audio_dur > video_dur:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-i", audio_path,
             "-map", "0:v", "-map", "1:a",
             "-ss", "0", "-t", str(video_dur),
             "-c:v", "copy", "-c:a", "aac", output_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    else:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-i", audio_path,
             "-map", "0:v", "-map", "1:a",
             "-c:v", "copy", "-c:a", "aac", output_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    logger.info(f"Wrote narrated video: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple video clips into one long-form video")
    parser.add_argument("--clips", nargs="+", required=True, help="Ordered list of clip paths to merge")
    parser.add_argument("--output", required=True, help="Path to write the merged .mp4")
    parser.add_argument("--crossfade", type=float, default=0.0, help="Crossfade duration in seconds (0 = hard cuts)")
    args = parser.parse_args()

    merge_clips(args.clips, args.output, args.crossfade)


if __name__ == "__main__":
    main()
