from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from generate_narration import generate_narration
from utils import get_logger

logger = get_logger(__name__)

# A small pool of distinct edge-tts (free, no API key) voices; characters are
# assigned one deterministically so the same character always sounds the
# same across a run. Names must be real edge-tts voice ids (also used
# directly by generate_narration._KOKORO_VOICE_MAP if Kokoro is installed).
VOICE_POOL = [
    "en-US-AriaNeural",   # warm, versatile female (default/narrator)
    "en-US-GuyNeural",    # confident, energetic male
    "en-GB-RyanNeural",   # laid-back, resonant British male
    "en-US-JennyNeural",  # natural, expressive female
    "en-US-AnaNeural",    # bright, youthful female
]


def assign_voices(character_names: list) -> dict:
    return {name: VOICE_POOL[i % len(VOICE_POOL)] for i, name in enumerate(sorted(set(character_names)))}


def generate_dialogue_track(
    scenes: list,
    work_dir: str,
    output_path: str,
    scene_offsets: list = None,
    scene_durations: list = None,
    max_workers: int = 4,
) -> list:
    from moviepy import AudioFileClip, CompositeAudioClip

    lines = [
        (i, j, scene.get("dialogue", [])[j])
        for i, scene in enumerate(scenes)
        for j in range(len(scene.get("dialogue", [])))
        if scene.get("dialogue", [])[j].get("line")
    ]
    if not lines:
        return []

    voice_map = assign_voices([entry["character"] for _, _, entry in lines if entry.get("character")])
    work = Path(work_dir) / "dialogue"
    work.mkdir(parents=True, exist_ok=True)

    def _render(item):
        i, j, entry = item
        text = entry["line"]
        character = entry.get("character", "Narrator")
        seg_path = work / f"line_{i:03d}_{j:02d}.mp3"
        voice_id = voice_map.get(character, VOICE_POOL[0])
        generate_narration(text, str(seg_path), voice_id=voice_id)
        return {"scene": i, "line": j, "character": character, "text": text, "path": str(seg_path)}

    logger.info(f"Generating {len(lines)} dialogue lines across {len(voice_map)} voices ({max_workers} in parallel)")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        rendered = list(pool.map(_render, lines))

    # Preserve scene/line order even though rendering ran concurrently.
    rendered.sort(key=lambda r: (r["scene"], r["line"]))

    # Place each scene's dialogue at that scene's actual position in the merged
    # video timeline (scene_offsets/scene_durations), instead of packing every
    # line back-to-back regardless of scene length. Without this, total
    # narration length (sum of TTS line durations) ends up far shorter than
    # total video length (scenes * clip_duration), so the audio finishes long
    # before the video does.
    LEAD_IN = 0.3
    GAP = 0.3
    scene_cursor = {}
    fallback_t = 0.0

    clips = []
    cues = []
    for r in rendered:
        clip = AudioFileClip(r["path"])
        scene_i = r["scene"]

        if scene_offsets is not None:
            scene_start = scene_offsets[scene_i]
            start_t = scene_cursor.get(scene_i, scene_start + LEAD_IN)
            if scene_durations is not None:
                scene_end = scene_start + scene_durations[scene_i]
                start_t = min(start_t, max(scene_start, scene_end - clip.duration))
        else:
            start_t = fallback_t

        end_t = start_t + clip.duration
        cues.append({"start": start_t, "end": end_t, "character": r["character"], "text": r["text"]})
        clips.append(clip.with_start(start_t))

        if scene_offsets is not None:
            scene_cursor[scene_i] = end_t + GAP
        else:
            fallback_t = end_t + 0.4

    final = CompositeAudioClip(clips)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    final.write_audiofile(output_path, logger=None)
    total = final.duration
    final.close()
    for c in clips:
        c.close()

    logger.info(f"Wrote dialogue track ({len(cues)} lines, {total:.1f}s) to {output_path}")
    return cues


def write_srt(cues: list, srt_path: str) -> None:
    def fmt(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")

    lines = []
    for i, cue in enumerate(cues, 1):
        lines.append(str(i))
        lines.append(f"{fmt(cue['start'])} --> {fmt(cue['end'])}")
        lines.append(f"{cue['character']}: {cue['text']}")
        lines.append("")

    Path(srt_path).parent.mkdir(parents=True, exist_ok=True)
    Path(srt_path).write_text("\n".join(lines))
    logger.info(f"Wrote captions ({len(cues)} cues) to {srt_path}")


_ASS_COLORS = [
    "&H00FF6600&", "&H000066FF&", "&H0000FF66&",
    "&H0066FFFF&", "&H00FF00FF&", "&H00FF3366&",
]


def write_ass(cues: list, ass_path: str) -> str:
    def fmt(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h:01d}:{m:02d}:{s:05.2f}"

    lines = [
        "[Script Info]",
        "Title: Long-Form Video Captions",
        "ScriptType: v4.00+",
        # Must match the canonical output resolution (fetch_stock_video.TARGET_WIDTH/HEIGHT).
        # Without an explicit PlayRes, libass falls back to guessing based on
        # whatever resolution the current frame happens to report, which made
        # captions render at wildly inconsistent sizes/positions.
        "PlayResX: 1920",
        "PlayResY: 1080",
        "WrapStyle: 0",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        'Style: W,DejaVu Sans,48,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,3,2,2,80,80,60,1',
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    for i, cue in enumerate(cues):
        color = _ASS_COLORS[i % len(_ASS_COLORS)]
        text = f"{cue['character']}: {cue['text']}"
        lines.append(
            f"Dialogue: 0,{fmt(cue['start'])},{fmt(cue['end'])},W,,0,0,0,,"
            f"{{\\c{color}}}{text}"
        )

    Path(ass_path).parent.mkdir(parents=True, exist_ok=True)
    Path(ass_path).write_text("\n".join(lines))
    logger.info(f"Wrote captions ({len(cues)} cues) to {ass_path}")
    return ass_path
