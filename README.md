# Animated Long-Form Video Pipeline

Generates a long-form animated video by chaining several short image-to-video
clips end to end. Each clip's last frame becomes the seed image for the next
clip, so the story flows continuously instead of jumping between unrelated
shots. All generation happens on Hugging Face's hosted Inference Providers
(via the `fal-ai` provider) — nothing downloads or runs locally, so this
works on a plain GitHub Actions runner with no GPU and minimal disk. Clip
merging and frame extraction use [MoviePy](https://github.com/Zulko/moviepy)
(a proper Python video library, not raw ffmpeg CLI calls) — its `ffmpeg`
binary dependency installs automatically via `pip install -r
requirements.txt`, no separate system package step needed.

## How it works

1. `storyboard.json` lists an ordered sequence of scene prompts (optionally
   with per-scene overrides — see below).
2. `scripts/pipeline.py` runs through the scenes:
   - `scripts/generate_clip.py` calls the Hugging Face `image_to_video` API
     for the current scene, using the current seed image + prompt. API calls
     retry automatically (4 attempts, exponential backoff) on transient
     failures.
   - `scripts/extract_last_frame.py` grabs the last frame of that clip with
     MoviePy — this becomes the seed image for the next scene.
   - Progress is checkpointed to `work/manifest.json` after each clip. If the
     pipeline is re-run over the same `--work-dir`, already-generated clips
     are skipped instead of re-generated (use `--no-resume` to force a clean
     regeneration). This matters because a failed run partway through a long
     storyboard shouldn't throw away clips you already paid API credits for.
3. `scripts/merge_clips.py` stitches all the generated clips into one final
   `.mp4` with MoviePy — either hard cuts (default) or a crossfade transition
   between clips (`--crossfade <seconds>`).

## Setup

1. Add a Hugging Face token as a **repository secret** named `HF_TOKEN`
   (Settings > Secrets and variables > Actions > New repository secret).
   Never commit a token to the repo — the workflow reads it from
   `secrets.HF_TOKEN` only, and a pre-flight step fails the run early with a
   clear error if the secret isn't set.
2. Commit an initial seed image (the first frame of your story) somewhere
   in the repo, e.g. `assets/seed.png`.
3. Edit `storyboard.example.json` (or add your own storyboard JSON) with
   your scene prompts.

## Running via GitHub Actions

Trigger the **Generate Long-Form Video** workflow manually (Actions tab >
Generate Long-Form Video > Run workflow), providing:

- `storyboard_path` — path to your storyboard JSON (defaults to
  `storyboard.example.json`)
- `init_image_path` — path to your committed seed image, e.g.
  `assets/seed.png`
- `model` — optional, defaults to `Wan-AI/Wan2.2-I2V-A14B`
- `crossfade` — optional crossfade duration in seconds between clips
  (defaults to `0`, hard cuts)

The finished video is uploaded as a workflow artifact named `final-video`,
and a run summary (scene count, duration, model) is written to the job
summary. Only one generation runs at a time (`concurrency` guard) so a
double-trigger can't burn API credits twice. If the run fails, the partial
`work/` directory (clips, frames, manifest) is uploaded as a `pipeline-debug`
artifact so you can see exactly where it stopped.

## Running locally

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in HF_TOKEN
export HF_TOKEN=$(grep HF_TOKEN .env | cut -d= -f2)

python scripts/pipeline.py \
  --storyboard storyboard.example.json \
  --init-image assets/seed.png \
  --output output/final_video.mp4
```

`ffmpeg` itself is installed automatically as a MoviePy dependency — no
separate system install needed.

Useful flags:
- `--dry-run` — validates the storyboard and seed image and prints what
  would be generated, without calling the API or spending credits.
- `--crossfade 0.5` — crossfade clips together instead of hard cuts.
- `--no-resume` — ignore any existing `work/manifest.json` and regenerate
  every clip from scratch.

## Storyboard format

```json
{
  "title": "Example Story",
  "scenes": [
    { "prompt": "..." },
    {
      "prompt": "...",
      "negative_prompt": "...",
      "num_frames": 81,
      "num_inference_steps": 30,
      "guidance_scale": 7.5,
      "seed": 42,
      "model": "Wan-AI/Wan2.2-I2V-A14B"
    }
  ]
}
```

Each scene becomes one generated clip, chained onto the previous clip's
final frame. Only `prompt` is required — every other field is an optional
per-scene override of the pipeline defaults, letting you mix models or
tune individual shots without changing the rest of the storyboard.

## Notes

- `image_to_video` requires an image input — there's no pure text-to-video
  step here, so you always need one seed image to start the story.
- Cost/quota is whatever your Hugging Face Inference Providers account (and
  the underlying `fal-ai` provider) charges per clip — check your account
  before running a long storyboard, and consider `--dry-run` first.
- Swap `model` (globally or per scene) to any other Inference-Providers-
  hosted image-to-video model if you want different quality/speed/cost
  tradeoffs.
