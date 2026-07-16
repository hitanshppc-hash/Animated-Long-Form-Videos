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
    raw = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", path]
    )
    return float(_json.loads(raw)["format"]["duration"])


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

    filter_parts = []
    stream_spec = f"[0:v][0:a][1:v][1:a]"
    offset = durations[0] - crossfade
    prev_v = "xf0_v"
    prev_a = "xf0_a"
    filter_parts.append(f"{stream_spec}xfade=transition=fade:duration={crossfade}:offset={offset}[{prev_v}][{prev_a}]")

    for i in range(2, len(clip_paths)):
        offset = sum(durations[:i]) - crossfade
        prev_v_name = prev_v
        prev_a_name = prev_a
        cur_v_name = f"xf{i-1}_v"
        cur_a_name = f"xf{i-1}_a"
        filter_parts.append(
            f"[{prev_v_name}][{prev_a_name}][{i}:v][{i}:a]"
            f"xfade=transition=fade:duration={crossfade}:offset={offset}[{cur_v_name}][{cur_a_name}]"
        )
        prev_v, prev_a = cur_v_name, cur_a_name

    filter_complex = "; ".join(filter_parts)
    input_args = []
    for p in clip_paths:
        input_args += ["-i", p]

    subprocess.run(
        ["ffmpeg", "-y"] + input_args +
        ["-filter_complex", filter_complex,
         "-map", f"[{prev_v}]", "-map", f"[{prev_a}]",
         "-c:v", "libx264", "-c:a", "aac", output_path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    total = sum(durations)
    logger.info(f"Merged {len(clip_paths)} clips ({crossfade}s crossfade) -> {output_path} ({total:.1f}s)")


def merge_clips(clip_paths: List[str], output_path: str, crossfade_duration: float = 0.0) -> None:
    for clip in clip_paths:
        if not Path(clip).exists():
            raise FileNotFoundError(f"Clip not found: {clip}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if crossfade_duration > 0 and len(clip_paths) > 1:
        _merge_with_xfade(clip_paths, output_path, crossfade_duration)
    else:
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
