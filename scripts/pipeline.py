import argparse
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from generate_clip import generate_clip
from generate_storyboard import generate_storyboard
from generate_dialogue_audio import generate_dialogue_track, write_srt, write_ass
from extract_last_frame import extract_last_frame
from fetch_seed_image import fetch_seed_image
from merge_clips import merge_clips, attach_narration, burn_subtitles
from history import load_titles, append_title
from utils import get_logger, video_duration

logger = get_logger(__name__)


def _load_storyboard(storyboard_path: str) -> dict:
    data = json.loads(Path(storyboard_path).read_text())
    scenes = data.get("scenes")
    if not scenes or not isinstance(scenes, list):
        raise ValueError(f"Storyboard {storyboard_path} has no 'scenes' list")
    for i, scene in enumerate(scenes):
        if not scene.get("prompt"):
            raise ValueError(f"Scene {i} is missing a 'prompt'")
    return data


def _character_context(characters: list) -> str:
    parts = [f"{c['name']} ({c['description']})" for c in characters if c.get("name") and c.get("description")]
    return ("Characters — " + "; ".join(parts) + ". ") if parts else ""


def run_pipeline(
    storyboard_path: str,
    init_image: str | None,
    work_dir: str,
    output_path: str,
    idea: str = "",
    num_scenes: int = 40,
    crossfade: float = 0.0,
    narrate: bool = False,
    dry_run: bool = False,
    resume: bool = True,
    history_path: str = "history.json",
    clip_duration: float = 12.0,
    parallel_workers: int = 3,
) -> None:
    start_time = time.monotonic()
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    if not Path(storyboard_path).exists():
        logger.info("No storyboard found, generating one dynamically")
        avoid_titles = load_titles(history_path)
        generate_storyboard(idea, num_scenes, storyboard_path, avoid_titles=avoid_titles)

    storyboard = _load_storyboard(storyboard_path)
    scenes = storyboard["scenes"]
    character_ctx = _character_context(storyboard.get("characters", []))
    world_ctx = storyboard.get("world_description", "")

    dialogue_line_count = sum(len(s.get("dialogue", [])) for s in scenes)
    logger.info(
        f"Plan: {len(scenes)} scenes, {len(storyboard.get('characters', []))} characters, "
        f"{dialogue_line_count} dialogue lines, narration={'on' if narrate else 'off'}, "
        f"crossfade={crossfade}s -- estimated API calls: {len(scenes)} video"
        + (f" + {dialogue_line_count} narration lines" if narrate else "")
    )

    if not init_image:
        init_image = str(work / "seed.jpg")
        if not Path(init_image).exists():
            query = storyboard.get("title") or (scenes[0]["prompt"][:80] if scenes else "abstract art")
            logger.info(f"No seed image given, fetching one for: {query!r}")
            fetch_seed_image(query, init_image)

    manifest_path = work / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if resume and manifest_path.exists() else {}

    clip_paths = [None] * len(scenes)
    pending = []

    for i, scene in enumerate(scenes):
        clip_path = str(work / f"clip_{i:03d}.mp4")
        frame_path = str(work / f"frame_{i:03d}.png")
        key = str(i)

        if resume and manifest.get(key) == "done" and Path(clip_path).exists() and Path(frame_path).exists():
            logger.info(f"[{i + 1}/{len(scenes)}] Skipping (already generated): {scene['prompt'][:60]}")
            clip_paths[i] = clip_path
            continue

        pending.append((i, scene, clip_path, frame_path))

    if pending and not dry_run and parallel_workers > 1:
        logger.info(f"Generating {len(pending)} clips in parallel ({parallel_workers} workers)...")

        def _gen(item):
            idx, scene, clip_path, frame_path = item
            generate_clip(
                init_image,
                scene["prompt"],
                clip_path,
                clip_duration=clip_duration,
            )
            extract_last_frame(clip_path, frame_path)
            return idx, clip_path

        with ThreadPoolExecutor(max_workers=parallel_workers) as pool:
            fut_map = {pool.submit(_gen, item): item[0] for item in pending}
            for future in as_completed(fut_map):
                idx, clip_path = future.result()
                clip_paths[idx] = clip_path
                logger.info(f"[{idx + 1}/{len(scenes)}] Generated clip: {scenes[idx]['prompt'][:60]}")

        for idx, _, _, _ in pending:
            manifest[str(idx)] = "done"
        manifest_path.write_text(json.dumps(manifest, indent=2))

    elif pending:
        current_image = init_image
        for idx, scene, clip_path, frame_path in pending:
            logger.info(f"[{idx + 1}/{len(scenes)}] Generating clip: {scene['prompt'][:60]}")
            generate_clip(
                current_image,
                scene["prompt"],
                clip_path,
                clip_duration=clip_duration,
            )
            clip_paths[idx] = clip_path

            if not dry_run:
                extract_last_frame(clip_path, frame_path)
                current_image = frame_path
                manifest[str(idx)] = "done"
                manifest_path.write_text(json.dumps(manifest, indent=2))

    clip_paths = [p for p in clip_paths if p is not None]

    if dry_run:
        logger.info(f"[dry-run] Would merge {len(clip_paths)} clips into {output_path}")
        return

    logger.info(f"Merging {len(clip_paths)} clips into {output_path}")
    merged_path = str(work / "merged_no_narration.mp4") if narrate else output_path
    merge_clips(clip_paths, merged_path, crossfade)

    if narrate:
        cues = generate_dialogue_track(scenes, work_dir, str(work / "narration.mp3"))
        if cues:
            attach_narration(merged_path, str(work / "narration.mp3"), output_path)
            srt_path = str(Path(output_path).with_suffix(".srt"))
            write_srt(cues, srt_path)
            ass_path = str(Path(output_path).with_suffix(".ass"))
            write_ass(cues, ass_path)
            burn_subtitles(output_path, ass_path, output_path)
        else:
            logger.warning("No dialogue lines found in storyboard, skipping narration")
            shutil.copy(merged_path, output_path)

    thumbnail_path = Path(output_path).parent / "thumbnail.jpg"
    try:
        shutil.copy(init_image, thumbnail_path)
    except Exception as exc:
        logger.warning(f"Could not write thumbnail: {exc}")

    metadata = {
        "title": storyboard.get("title", ""),
        "world_description": world_ctx,
        "description": storyboard.get("synopsis", ""),
        "characters": [c.get("name") for c in storyboard.get("characters", [])],
        "scene_count": len(scenes),
        "duration_seconds": round(video_duration(output_path), 1),
        "narrated": narrate,
        "elapsed_seconds": round(time.monotonic() - start_time, 1),
    }
    Path(output_path).with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2))

    if storyboard.get("title"):
        append_title(storyboard["title"], history_path)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write("## Long-form video generated\n\n")
            f.write(f"- Title: {metadata['title']}\n")
            f.write(f"- Scenes: {metadata['scene_count']}\n")
            f.write(f"- Characters: {', '.join(metadata['characters'])}\n")
            f.write(f"- Duration: {metadata['duration_seconds']}s\n")
            f.write(f"- Narration: {'yes' if narrate else 'no'}\n")
            f.write(f"- Elapsed: {metadata['elapsed_seconds']}s\n")
            f.write(f"- Output: `{output_path}`\n")

    logger.info(f"Done in {metadata['elapsed_seconds']}s.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full idea -> long-form animated video pipeline")
    parser.add_argument("--idea", default="", help="Story idea/premise/theme; leave blank to let the LLM invent one")
    parser.add_argument("--scenes", type=int, default=40, help="Number of scenes to generate if no storyboard exists yet")
    parser.add_argument("--storyboard", default="storyboard.json", help="Path to storyboard JSON (generated if missing)")
    parser.add_argument("--init-image", default=None, help="Path to the first seed image (auto-fetched if omitted)")
    parser.add_argument("--work-dir", default="work", help="Directory to store intermediate clips/frames")
    parser.add_argument("--output", default="output/final_video.mp4", help="Path to write the final merged video")

    parser.add_argument("--crossfade", type=float, default=0.0, help="Crossfade duration in seconds between clips")
    parser.add_argument("--clip-duration", type=float, default=12.0, help="Seconds per clip (8-12 recommended)")
    parser.add_argument("--parallel-workers", type=int, default=3, help="Number of parallel clip download workers")
    parser.add_argument("--narrate", action="store_true", help="Generate dialogue narration and burn in captions (.srt)")
    parser.add_argument("--dry-run", action="store_true", help="Validate storyboard/inputs without calling any API")
    parser.add_argument("--no-resume", action="store_true", help="Ignore any existing manifest and regenerate all clips")
    parser.add_argument("--history", default="history.json", help="Path to the past-titles history file")
    args = parser.parse_args()

    run_pipeline(
        args.storyboard,
        args.init_image,
        args.work_dir,
        args.output,
        idea=args.idea,
        num_scenes=args.scenes,
        crossfade=args.crossfade,
        narrate=args.narrate,
        dry_run=args.dry_run,
        resume=not args.no_resume,
        history_path=args.history,
        clip_duration=args.clip_duration,
        parallel_workers=args.parallel_workers,
    )


if __name__ == "__main__":
    main()
