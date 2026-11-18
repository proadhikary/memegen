# memegen — local meme generation

## memeloop.py — agentic loop (recommended)

```
brief ─► WRITER (LLM) ─► 10 new concepts ─► uniqueness filter (vs. output/all_memes.csv)
            │
            ▼
       ART DIRECTOR (LLM) ─► spec: format, aspect, layout (single / side_by_side / stacked),
            ▲                 art style, scene per panel, header band, every text block with
            │                 position / region / on_object label, font, size, colours, boxes
            │                       │
            │                       ▼
            │                IMAGE MODEL (GPU 1) ─► panels (fresh random seeds, perceptual-hash dedupe)
            │                       │
            │                       ▼
            │                GROUNDER (VLM) ─► finds objects for "on_object" labels
            │                       │
            │                       ▼
            │                COMPOSER (PIL) ─► final meme (collision-free text, auto contrast)
            │                       │
            └── revise ◄── CRITIC (VLM) scores 1-10; < --accept → revise (max --rounds), best kept
```

```bash
source .venv/bin/activate
# 10 NEW memes every time you run it (never repeats earlier ones for the same brief)
python memeloop.py --brief "Superstition: You should not proceed a road if a cat crosses before you. Give 5 scenarios where it is not harmful and 5 where it is harmful"

python memeloop.py --briefs briefs.txt              # 10 new memes per brief in the file
python memeloop.py --brief "..." --batches 0        # loop forever, 10 new memes per brief per batch
python memeloop.py --brief "..." --model qwen       # Qwen-Image-2.1-Turbo (non-commercial license)
```
Options: `-n 10`, `--rounds 3`, `--accept 8`, `--llm-gpu 0 --image-gpu 1`,
`--director-llm <hf id>` (use a different model for the art director).

### Output
```
output/all_memes.csv                 master CSV of every meme ever made (used for uniqueness)
output/<run_id>/memes.csv            this run's CSV
output/<run_id>/memes/<id>.png       final memes
output/<run_id>/panels/<id>_p0.png   raw generated panels (no text)
output/<run_id>/specs/<id>.json      concept + full art-director spec + critique + seeds
```
CSV columns: meme_id, run_id, created_at, brief_id, brief, label, scenario, joke, captions, visual_idea,
format, layout, aspect, style, header_text, panel_scenes, texts (positions/fonts/colours as JSON),
image_model, seeds, rounds, status (accepted / needs_review), critic_score, critic_problems,
image_path, spec_path, dhash.

Speed (2× A6000, Z-Image): ~8-9 min per 10 memes incl. up to 3 critique rounds; the LLM is the bottleneck.

## Image models
| flag | model | license |
|---|---|---|
| `zimage` (default) | Tongyi-MAI/Z-Image-Turbo | Apache-2.0 (commercial OK) |
| `qwen` | Qwen/Qwen-Image-2.1-Turbo | Qwen Research License (non-commercial), best cartoon look |

## Setup (new machine)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```
Models download automatically from Hugging Face on first run (~19 GB LLM + ~31 GB per image model).
Needs 2 GPUs with >= 24 GB (Z-Image) / >= 40 GB (Qwen-Image), or one 48 GB GPU with Z-Image:
`--llm-gpu 0 --image-gpu 0`.

## Files
| file | role |
|---|---|
| memeloop.py | CLI + orchestration loop, uniqueness, CSV |
| agents.py | writer / art director / grounder / critic prompts and validators |
| llm.py | batched Qwen3.5 vision-language wrapper |
| images.py | image-model backends (Z-Image-Turbo, Qwen-Image-2.1-Turbo) |
| layout.py | composer: panels, header band, text placement, fonts |
| fonts/ | OFL / Apache fonts (Anton, Bangers, Permanent Marker, Oswald, Inter, Comic Neue) |
| briefs.txt | example briefs |
