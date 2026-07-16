import argparse
import json
import re
from pathlib import Path

from llm_client import chat_completion
from utils import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = (Path(__file__).parent / "prompts" / "storyboard_system.txt").read_text()

DEFAULT_BATCH_SIZE = 10


def _extract_json(text: str) -> dict:
    cleaned = re.sub(r"(?s)^\s*(?:```(?:json)?\s*)?(.*?)(?:\s*```)?\s*$", r"\1", text.strip())

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in model output: {text[:200]!r}")

    raw = cleaned[start : end + 1]

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    raw = re.sub(r",\s*}", "}", raw)
    raw = re.sub(r",\s*]", "]", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    raw = re.sub(r'(?<=\w)"(?=\w)', r'\"', raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    raw = re.sub(r"(?<!\\)'", '"', raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in model output (near char {exc.pos}): {raw[exc.pos-100:exc.pos+100]!r}") from exc


def _generate_batch(
    idea: str,
    summary: str,
    title: str,
    world_description: str,
    characters: list,
    batch_size: int,
    is_first: bool,
    avoid_titles: list = None,
) -> dict:
    idea = idea or "Invent an original, interesting animated short story premise."
    if is_first:
        user_prompt = (
            f"Story idea: {idea}\n"
            f"Write the first {batch_size} scenes, invent a title, define world_description, "
            f"and define the character roster."
        )
        if avoid_titles:
            user_prompt += (
                f"\nAlready-used titles/premises, do NOT repeat or closely resemble any of these — invent "
                f"something clearly different: {json.dumps(avoid_titles)}"
            )
    else:
        user_prompt = (
            f"Story idea: {idea}\n"
            f"Title (keep identical): {title}\n"
            f"World description (keep identical): {world_description}\n"
            f"Existing character roster (keep identical, may add new ones): {json.dumps(characters)}\n"
            f"Story so far: {summary}\n"
            f"Continue with the next {batch_size} scenes."
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(3):
        content = chat_completion(messages=messages, max_tokens=8192)
        try:
            data = _extract_json(content)
            break
        except ValueError as e:
            if attempt == 2:
                raise
            logger.warning(f"JSON parse failed (attempt {attempt+1}/3): {e}")
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"The JSON was invalid: {e}\n\nFix the JSON syntax errors. Output ONLY valid JSON — no extra text."})

    scenes = data.get("scenes")
    if not scenes or not isinstance(scenes, list):
        raise ValueError(f"Model output missing 'scenes' list: {content[:300]!r}")
    for i, scene in enumerate(scenes):
        if not scene.get("prompt"):
            raise ValueError(f"Scene {i} missing 'prompt': {scene}")
        scene.setdefault("action", "")
        scene.setdefault("dialogue", [])
        scene.setdefault("tone", "expository")

    return data


def _merge_characters(existing: list, new: list) -> list:
    by_name = {c["name"]: c for c in existing if c.get("name")}
    for c in new or []:
        if c.get("name") and c["name"] not in by_name:
            by_name[c["name"]] = c
    return list(by_name.values())


def generate_storyboard(
    idea: str,
    num_scenes: int,
    output_path: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    avoid_titles: list = None,
) -> dict:
    all_scenes = []
    summary = ""
    title = ""
    world_description = ""
    characters = []
    remaining = num_scenes
    batch_num = 0

    while remaining > 0:
        this_batch = min(batch_size, remaining)
        batch_num += 1
        logger.info(f"Generating scenes batch {batch_num} ({this_batch} scenes, {len(all_scenes)}/{num_scenes} so far)")

        data = _generate_batch(
            idea,
            summary,
            title,
            world_description,
            characters,
            this_batch,
            is_first=(batch_num == 1),
            avoid_titles=avoid_titles,
        )
        all_scenes.extend(data["scenes"])
        summary = data.get("summary", summary)
        characters = _merge_characters(characters, data.get("characters", []))
        if batch_num == 1:
            title = data.get("title") or (idea[:60] if idea else "Untitled Story")
            world_description = data.get("world_description", "")
        remaining -= this_batch

    storyboard = {
        "title": title,
        "world_description": world_description,
        "characters": characters,
        "scenes": all_scenes,
        "synopsis": summary,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(storyboard, indent=2))
    logger.info(f"Wrote storyboard ({len(all_scenes)} scenes, {len(characters)} characters) to {output_path}")
    return storyboard


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a storyboard JSON from a story idea via an LLM fallback chain")
    parser.add_argument("--idea", default="", help="Story idea/premise/theme; leave blank to let the LLM invent one")
    parser.add_argument("--scenes", type=int, default=6, help="Total number of scenes to generate")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Scenes generated per LLM call")
    parser.add_argument("--output", default="storyboard.json", help="Path to write the storyboard JSON")
    args = parser.parse_args()

    generate_storyboard(args.idea, args.scenes, args.output, args.batch_size)


if __name__ == "__main__":
    main()
