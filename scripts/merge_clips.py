import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List

from utils import get_logger, video_duration as probe_duration

logger = get_logger(__name__)


def _run_ff(cmd: list, desc: str = "ffmpeg") -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        err = result.stderr[:600]
        raise RuntimeError(f"{desc} failed (rc={result.returncode}): {err}")


# -------------------------------------------------------------------- concat
def _try_concat_demuxer(clip_paths: List[str], output_path: str) -> bool:
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for p in clip_paths:
                f.write(f"file '{Path(p).resolve()}'\n")
            list_path = f.name
        try:
            _run_ff(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                 "-i", list_path, "-c", "copy", output_path],
                "concat demuxer",
            )
            logger.info(f"Merged {len(clip_paths)} clips (hard cuts) -> {output_path}")
            return True
        finally:
            os.unlink(list_path)
    except Exception as e:
        logger.warning(f"concat demuxer failed: {e}")
        return False


def _try_concat_reencode(clip_paths: List[str], output_path: str) -> bool:
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for p in clip_paths:
                f.write(f"file '{Path(p).resolve()}'\n")
            list_path = f.name
        try:
            _run_ff(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                 "-i", list_path, "-c:v", "libx264", "-preset", "veryfast", "-an", output_path],
                "concat re-encode",
            )
            logger.info(f"Re-encoded merge of {len(clip_paths)} clips -> {output_path}")
            return True
        finally:
            os.unlink(list_path)
    except Exception as e:
        logger.warning(f"concat re-encode failed: {e}")
        return False


def _try_moviepy_concat(clip_paths: List[str], output_path: str) -> bool:
    try:
        from moviepy import VideoFileClip, concatenate_videoclips
        clips = [VideoFileClip(p) for p in clip_paths]
        try:
            final = concatenate_videoclips(clips, method="compose")
            final.write_videofile(output_path, codec="libx264", audio_codec="aac", preset="veryfast", logger=None)
            final.close()
        finally:
            for c in clips:
                c.close()
        logger.info(f"MoviePy merge of {len(clip_paths)} clips -> {output_path}")
        return True
    except Exception as e:
        logger.warning(f"moviepy concat failed: {e}")
        return False


def _merge_concat(clip_paths: List[str], output_path: str) -> None:
    for fn in [_try_concat_demuxer, _try_concat_reencode, _try_moviepy_concat]:
        if fn(clip_paths, output_path):
            return
    raise RuntimeError(f"All concat methods failed for {len(clip_paths)} clips")


# ------------------------------------------------------------- xfade (batched)
_BATCH_SIZE = 10
# A single ffmpeg xfade filter_complex with too many simultaneous -i inputs
# risks resource exhaustion (open fds, filter graph size). Verified working
# directly at 40 clips; the cron rotation also runs 75/100-scene videos, so
# route those straight to the batched path instead of gambling on a direct
# attempt first.
_MAX_DIRECT_XFADE_CLIPS = 50


def _xfade_batch(clip_paths: List[str], output_path: str, crossfade: float) -> None:
    durations = [probe_duration(p) for p in clip_paths]
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
    _run_ff(
        ["ffmpeg", "-y"] + input_args +
        ["-filter_complex", filter_complex,
         "-map", "[vout]", "-an",
         # YouTube re-transcodes on ingest, so trading a slower/denser
         # encode for a faster one here costs nothing perceptible while
         # cutting real time off the merge step.
         "-c:v", "libx264", "-preset", "veryfast", output_path],
        "xfade",
    )
    total = sum(durations) - crossfade * (n - 1)
    logger.info(f"xfade batch of {n} clips ({crossfade}s) -> {output_path} ({total:.1f}s)")


def _try_xfade_sequential(clip_paths: List[str], output_path: str, crossfade: float) -> bool:
    try:
        _xfade_batch(clip_paths, output_path, crossfade)
        return True
    except Exception as e:
        logger.warning(f"xfade all-at-once failed ({e}); trying batched xfade")
        return False


def _try_xfade_batched(clip_paths: List[str], output_path: str, crossfade: float) -> bool:
    try:
        work_dir = Path(output_path).parent / ".__xfade_batches"
        work_dir.mkdir(parents=True, exist_ok=True)
        batch_files = []
        for i in range(0, len(clip_paths), _BATCH_SIZE):
            batch = clip_paths[i:i + _BATCH_SIZE]
            batch_out = str(work_dir / f"batch_{i:04d}.mp4")
            try:
                _xfade_batch(batch, batch_out, crossfade)
            except Exception:
                _merge_concat(batch, batch_out)
            batch_files.append(batch_out)
        _merge_concat(batch_files, output_path)
        for f in batch_files:
            try:
                os.unlink(f)
            except OSError:
                pass
        try:
            work_dir.rmdir()
        except OSError:
            pass
        logger.info(f"Batched xfade merge of {len(clip_paths)} clips -> {output_path}")
        return True
    except Exception as e:
        logger.warning(f"batched xfade failed ({e})")
        return False


# ------------------------------------------------------------------- public
def merge_clips(clip_paths: List[str], output_path: str, crossfade_duration: float = 0.0) -> None:
    for clip in clip_paths:
        if not Path(clip).exists():
            raise FileNotFoundError(f"Clip not found: {clip}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    crossfade = min(crossfade_duration, 2.0)

    if crossfade > 0 and len(clip_paths) > 1:
        if len(clip_paths) <= _MAX_DIRECT_XFADE_CLIPS:
            if _try_xfade_sequential(clip_paths, output_path, crossfade):
                return
        else:
            logger.info(
                f"{len(clip_paths)} clips exceeds direct-xfade threshold "
                f"({_MAX_DIRECT_XFADE_CLIPS}); going straight to batched xfade"
            )
        if _try_xfade_batched(clip_paths, output_path, crossfade):
            return
        logger.warning("All xfade methods failed, falling back to hard cuts")

    _merge_concat(clip_paths, output_path)


def attach_narration(video_path: str, audio_path: str, output_path: str) -> None:
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    duration = probe_duration(video_path)
    try:
        _run_ff(
            ["ffmpeg", "-y", "-i", video_path, "-i", audio_path,
             "-map", "0:v", "-map", "1:a",
             "-af", f"apad=whole_dur={duration}",
             "-ss", "0", "-t", str(duration),
             "-c:v", "copy", "-c:a", "aac", "-shortest", output_path],
            "attach_narration",
        )
        logger.info(f"Wrote narrated video: {output_path}")
        return
    except Exception as e:
        logger.warning(f"ffmpeg narration attach failed ({e}), trying moviepy")

    try:
        from moviepy import AudioClip, AudioFileClip, CompositeAudioClip, VideoFileClip
        video = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path)
        try:
            if audio.duration > video.duration:
                audio = audio.subclipped(0, video.duration)
            elif audio.duration < video.duration:
                # Pad with trailing silence so narration/video end together
                # instead of leaving the rest of the video silent.
                silence = AudioClip(lambda t: 0, duration=video.duration - audio.duration, fps=audio.fps)
                audio = CompositeAudioClip([audio, silence.with_start(audio.duration)])
            final = video.with_audio(audio)
            final.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)
            final.close()
        finally:
            video.close()
            audio.close()
        logger.info(f"Wrote narrated video (moviepy fallback): {output_path}")
    except Exception as e2:
        raise RuntimeError(f"All narration attach methods failed: {e2}")


def burn_subtitles(video_path: str, subtitle_path: str, output_path: str) -> None:
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not Path(subtitle_path).exists():
        raise FileNotFoundError(f"Subtitle file not found: {subtitle_path}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(Path(output_path).with_suffix(".tmp.burned.mp4"))

    escaped = str(Path(subtitle_path).resolve()).replace(":", "\\:").replace("'", "\\'")
    try:
        _run_ff(
            ["ffmpeg", "-y", "-i", video_path,
             "-vf", f"subtitles='{escaped}'",
             "-c:a", "copy", "-c:v", "libx264", "-crf", "22", "-preset", "fast",
             tmp],
            "burn_subtitles",
        )
        os.replace(tmp, output_path)
        logger.info(f"Wrote captioned video: {output_path}")
    except Exception as e:
        if Path(tmp).exists():
            os.remove(tmp)
        logger.warning(f"Subtitle burn-in failed ({e}), using original video")
        if video_path != output_path:
            shutil.copy2(video_path, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple video clips into one long-form video")
    parser.add_argument("--clips", nargs="+", required=True, help="Ordered list of clip paths to merge")
    parser.add_argument("--output", required=True, help="Path to write the merged .mp4")
    parser.add_argument("--crossfade", type=float, default=0.0, help="Crossfade duration in seconds (0 = hard cuts)")
    args = parser.parse_args()

    merge_clips(args.clips, args.output, args.crossfade)


if __name__ == "__main__":
    main()
