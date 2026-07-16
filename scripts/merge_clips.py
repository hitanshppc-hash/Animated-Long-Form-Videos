import argparse
from pathlib import Path
from typing import List

from moviepy import AudioFileClip, VideoFileClip, concatenate_videoclips, vfx

from utils import get_logger

logger = get_logger(__name__)


def merge_clips(clip_paths: List[str], output_path: str, crossfade_duration: float = 0.0) -> None:
    for clip in clip_paths:
        if not Path(clip).exists():
            raise FileNotFoundError(f"Clip not found: {clip}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    clips = [VideoFileClip(p) for p in clip_paths]
    try:
        if crossfade_duration > 0 and len(clips) > 1:
            logger.info(f"Merging {len(clips)} clips with {crossfade_duration}s crossfade")
            faded = [clips[0]] + [c.with_effects([vfx.CrossFadeIn(crossfade_duration)]) for c in clips[1:]]
            final = concatenate_videoclips(faded, method="compose", padding=-crossfade_duration)
        else:
            logger.info(f"Merging {len(clips)} clips (hard cuts)")
            final = concatenate_videoclips(clips, method="compose")

        try:
            final.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)
        finally:
            final.close()

        logger.info(f"Wrote merged video: {output_path} ({final.duration:.1f}s)")
    finally:
        for c in clips:
            c.close()


def attach_narration(video_path: str, audio_path: str, output_path: str) -> None:
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)
    try:
        if audio.duration > video.duration:
            audio = audio.subclipped(0, video.duration)
        final = video.with_audio(audio)
        try:
            final.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)
        finally:
            final.close()
        logger.info(f"Wrote narrated video: {output_path}")
    finally:
        video.close()
        audio.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple video clips into one long-form video")
    parser.add_argument("--clips", nargs="+", required=True, help="Ordered list of clip paths to merge")
    parser.add_argument("--output", required=True, help="Path to write the merged .mp4")
    parser.add_argument("--crossfade", type=float, default=0.0, help="Crossfade duration in seconds (0 = hard cuts)")
    args = parser.parse_args()

    merge_clips(args.clips, args.output, args.crossfade)


if __name__ == "__main__":
    main()
