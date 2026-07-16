from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from generate_narration import generate_narration
from utils import get_logger

logger = get_logger(__name__)

# A small pool of distinct ElevenLabs voices; characters are assigned one
# deterministically so the same character always sounds the same across a run.
VOICE_POOL = [
    "JBFqnCBsd6RMkjVDRZzb",  # George - warm, captivating storyteller (default/narrator)
    "EXAVITQu4vr4xnSDxMaL",  # Sarah - mature, reassuring, confident
    "CwhRBWXzGAHq8TQ4Fs17",  # Roger - laid-back, casual, resonant
    "IKne3meq5aSn9XLyUdCD",  # Charlie - deep, confident, energetic
    "FGY2WhTYpPnrIDTdsKH5",  # Laura - enthusiastic, quirky attitude
]


def assign_voices(character_names: list) -> dict:
    return {name: VOICE_POOL[i % len(VOICE_POOL)] for i, name in enumerate(sorted(set(character_names)))}


def generate_dialogue_track(scenes: list, work_dir: str, output_path: str, max_workers: int = 4) -> list:
    from moviepy import AudioClip, AudioFileClip, concatenate_audioclips

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

    silence = AudioClip(lambda t: 0, duration=0.4, fps=44100)
    clips = []
    cues = []
    t = 0.0
    for r in rendered:
        clip = AudioFileClip(r["path"])
        cues.append({"start": t, "end": t + clip.duration, "character": r["character"], "text": r["text"]})
        clips.append(clip)
        clips.append(silence)
        t += clip.duration + silence.duration

    final = concatenate_audioclips(clips)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    final.write_audiofile(output_path, logger=None)
    final.close()
    for c in clips:
        if c is not silence:
            c.close()

    logger.info(f"Wrote dialogue track ({len(cues)} lines, {t:.1f}s) to {output_path}")
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
        "WrapStyle: 0",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        'Style: W,DejaVu Sans,58,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,3,2,2,20,20,50,1',
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
