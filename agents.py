"""The four agents of the meme loop. Each is a prompt + validator on top of llm.LLM."""
import json
import random

from layout import ASPECTS, FONTS, LAYOUTS, POSITIONS
from llm import msg

# ---------------------------------------------------------------- 1. writer
WRITER_SYSTEM = """You are a sharp, witty meme writer for a big social-media meme page.
You get a BRIEF and write {n} NEW meme concepts that fulfil it.
- If the brief asks for scenarios in several categories (e.g. "not harmful" and "harmful"),
  split the {n} memes evenly between the categories, grouped by category.
- Every concept must be clearly about the brief's premise; the joke comes from the premise itself.
- All concepts must be different from each other AND from the "ALREADY USED" list (new situations,
  new characters, new punchlines - not rewordings).
- Captions are short (max ~10 words each), punchy, internet-meme voice. VARY the caption formats across the set:
  "ME:" / "NOBODY:" / "WHEN ..." / "POV: ..." / "MY BRAIN: ..." / "THEM: ... ME: ..." / dialogue / a single
  one-liner / "EXPECTATION vs REALITY" / labelled characters. Use any opener at most twice per set.
  Captions in one meme must not repeat each other.
- Captions must be the joke itself: NEVER write the category label ("harmful", "not harmful") or restate the brief in a caption.
- Keep humour light; never hateful; no real people; don't mock religions or ethnic groups.

Return ONLY JSON: {{"memes": [{{
  "label": "category from the brief, e.g. not harmful / harmful",
  "scenario": "one plain sentence describing the situation",
  "joke": "one sentence: why it's funny / the twist",
  "captions": ["setup text", "punchline text"],
  "visual_idea": "what the picture should show to sell the joke"
}}]}}"""


def writer_conv(brief, n, avoid):
    used = "\n".join(f"- {a}" for a in avoid[-60:]) or "(none)"
    return [msg("system", WRITER_SYSTEM.format(n=n)),
            msg("user", f"BRIEF:\n{brief}\n\nALREADY USED (do not repeat or paraphrase):\n{used}\n\nWrite {n} memes.")]


def validate_writer(obj):
    out = []
    for m in obj["memes"]:
        caps = [c for c in m.get("captions", []) if isinstance(c, str) and c.strip()]
        if all(isinstance(m.get(k), str) and m[k].strip() for k in ("label", "scenario", "visual_idea")) and caps:
            out.append({"label": m["label"].strip().lower(), "scenario": m["scenario"].strip(),
                        "joke": str(m.get("joke", "")).strip(), "captions": caps[:4],
                        "visual_idea": m["visual_idea"].strip()})
    if not out:
        raise ValueError("no memes")
    return out


# ---------------------------------------------------------------- 2. art director
FORMAT_GUIDE = {
    "classic": 'layout "single"; setup text at "top", punchline at "bottom", font "anton", white text, black outline.',
    "tweet": 'layout "single" with a "header" (tweet-style caption above the picture, font "inter", black on white); usually no text on the image itself.',
    "side_by_side": 'layout "side_by_side": two panels compared (expectation vs reality, them vs me, before vs after); short text in each panel.',
    "stacked": 'layout "stacked": two panels top/bottom (setup then reaction/consequence), text inside each panel.',
    "labels": 'layout "single"; characters/objects labelled with short words using position "on_object" + "object" (e.g. "ME", "BLACK CAT", "MY MOM"); optionally one caption at top.',
    "comic": 'layout "single" or "stacked"; font "bangers", text in coloured boxes ("bg") at top-left/bottom-right like comic narration.',
}
STYLE_FAMILIES = [
    "bold flat cartoon, thick black outlines, saturated colours, solid bright background",
    "3D animated movie render, soft lighting, expressive characters",
    "photorealistic candid smartphone photo, natural lighting",
    "retro 1960s comic book, halftone dots, bold inks",
    "anime / manga illustration, dramatic speed lines",
    "claymation stop-motion look, plasticine characters",
    "watercolor storybook illustration, warm palette",
    "cinematic movie still, dramatic lighting, shallow depth of field",
    "simple stick-figure webcomic on white background",
    "pixel-art retro video game scene",
]

DIRECTOR_SYSTEM = """You are the ART DIRECTOR of a meme studio. For one meme concept you design the whole
visual: format, layout, art style, scene to generate with an AI image model, and exact text placement.

Return ONLY JSON with this schema:
{{
 "format": "short name of the meme format you chose",
 "aspect": one of {aspects},
 "layout": one of {layouts}   (single = 1 panel, side_by_side / stacked = exactly 2 panels),
 "style": "concrete art style for the image model (medium, line work, colours, lighting, mood)",
 "background": "#hex canvas colour shown in gaps/bands",
 "header": null or {{"text": "...", "font": "inter", "color": "#hex", "bg": "#hex", "align": "left|center"}},
 "panels": [
   {{"scene": "detailed visual description for the image model: subjects, appearance, expressions, action, setting, camera framing. Include where empty space must be left for text (e.g. 'clear empty sky in the top quarter').",
     "texts": [{{"text": "...", "position": one of {positions},
                 "object": "only for on_object: short noun phrase of the thing to label, as seen in the scene",
                 "region": optional [x1,y1,x2,y2] in percent of the panel to override position,
                 "font": one of {fonts}, "size": "small|medium|large",
                 "color": "#hex", "outline": "#hex or null", "bg": "#hex box behind text or null",
                 "align": "center|left|right"}}]
   }}
 ]
}}
Rules:
- Use the concept's captions (you may shorten or split them to fit the format; keep the joke intact).
- The scene must NEVER contain written text, signs, letters or speech bubbles; all text is added by us.
- Scenes must be self-contained (the image model sees each panel separately): repeat character descriptions in every panel.
- Put text where the scene leaves empty space; never over faces. Ensure strong contrast (white + black outline on busy images, or a "bg" box).
- Fonts: anton=classic meme, bangers=comic, marker=handwritten, oswald=poster/news, inter=tweet/clean, comic=friendly.
- Keep total text short; meme text must be readable on a phone.
- Header/caption text is joke text only: never the brief, the category label, or an explanation.
- Each text in a panel gets its own distinct position (never two texts at the same/overlapping spot).
- In two-panel layouts each panel says something DIFFERENT (setup vs punchline); never repeat a text.
- Use size "medium" or "large" for captions; "small" only for short on_object labels (1-3 words)."""


def director_conv(concept, brief, fmt, style_family):
    # the director only sees the joke, not the brief/category, so it can't paste them onto the meme
    c = {k: concept[k] for k in ("scenario", "joke", "captions", "visual_idea")}
    return [msg("system", DIRECTOR_SYSTEM.format(aspects=list(ASPECTS), layouts=list(LAYOUTS),
                                                positions=list(POSITIONS) + ["on_object"], fonts=list(FONTS))),
            msg("user",
                f"CONCEPT:\n{json.dumps(c, ensure_ascii=False, indent=1)}\n\n"
                f"Suggested format (use it unless it clearly doesn't fit the joke): {fmt} -> {FORMAT_GUIDE[fmt]}\n"
                f"Suggested style family (adapt freely): {style_family}\n\nReturn the JSON spec.")]


def revise_conv(conv, spec, critique):
    return conv + [msg("assistant", json.dumps(spec, ensure_ascii=False)),
                   msg("user", "A critic reviewed the rendered meme:\n"
                       f"{json.dumps(critique, ensure_ascii=False, indent=1)}\n\n"
                       "Return a corrected, complete JSON spec. Only change a panel's scene if the picture "
                       "itself was the problem; fix text wording/position/size/colours for layout problems.")]


def validate_spec(s):
    s = dict(s)
    s["aspect"] = s.get("aspect") if s.get("aspect") in ASPECTS else "16:9"
    s["layout"] = s.get("layout") if s.get("layout") in LAYOUTS else "single"
    panels = [p for p in s["panels"] if isinstance(p, dict) and isinstance(p.get("scene"), str) and p["scene"].strip()]
    need = LAYOUTS[s["layout"]]
    if len(panels) < need:
        if not panels:
            raise ValueError("no panels")
        s["layout"], need = "single", 1
    s["panels"] = panels[:need]
    for p in s["panels"]:
        texts = []
        for t in p.get("texts") or []:
            if not isinstance(t, dict) or not str(t.get("text", "")).strip():
                continue
            t["text"] = str(t["text"]).strip()
            if t.get("position") not in POSITIONS and t.get("position") != "on_object":
                t["position"] = "bottom"
            if t["position"] == "on_object" and not str(t.get("object", "")).strip():
                t["position"] = "top"
            if t.get("font") not in FONTS:
                t["font"] = "anton"
            texts.append(t)
        p["texts"] = texts
    hdr = s.get("header")
    s["header"] = hdr if isinstance(hdr, dict) and str(hdr.get("text", "")).strip() else None
    if not s["header"] and not any(p["texts"] for p in s["panels"]):
        raise ValueError("meme has no text")
    s["style"] = str(s.get("style") or STYLE_FAMILIES[0])
    return s


def pick_formats(n, rng: random.Random):
    """Diverse (format, style) suggestions for a batch: cycle through shuffled lists."""
    fmts, styles = list(FORMAT_GUIDE), list(STYLE_FAMILIES)
    rng.shuffle(fmts)
    rng.shuffle(styles)
    return [(fmts[i % len(fmts)], styles[i % len(styles)]) for i in range(n)]


# ---------------------------------------------------------------- 3. grounder
def ground_conv(image, objects):
    return [msg("user", image,
                f"Locate each of these in the image: {json.dumps(objects)}. "
                'Output JSON list [{"label": ..., "bbox_2d": [x1, y1, x2, y2]}] in the same order, '
                "coordinates normalised to 0-1000.")]


def validate_ground(obj):
    if isinstance(obj, dict):
        obj = [obj]
    return [o["bbox_2d"] for o in obj if isinstance(o, dict) and len(o.get("bbox_2d", [])) == 4]


# ---------------------------------------------------------------- 4. critic
CRITIC_SYSTEM = """You are a strict meme quality critic. You see a finished meme plus what it was supposed to be.
Score 1-10 and check:
1. The picture clearly shows the intended scene/joke (right characters, action, emotion).
2. All overlay text is fully readable, not cut off, and does NOT cover faces or the key action.
3. No garbled/fake lettering or random text inside the illustration itself.
4. The meme is funny and instantly understandable; it fits its category label.
5. Visual quality: no broken anatomy, melted faces, extra limbs.
Return ONLY JSON: {"score": int, "problems": ["..."], "fix_picture": "concrete change to the scene, or empty",
"fix_layout": "concrete change to text/positions/format, or empty"}"""


def critic_conv(image, concept, spec):
    summary = {"label": concept["label"], "scenario": concept["scenario"], "joke": concept["joke"],
               "format": spec.get("format"), "scenes": [p["scene"] for p in spec["panels"]]}
    return [msg("system", CRITIC_SYSTEM),
            msg("user", image, f"Intended meme:\n{json.dumps(summary, ensure_ascii=False, indent=1)}\nReview it.")]


def validate_critique(obj):
    obj["score"] = int(obj["score"])
    obj["problems"] = [str(p) for p in obj.get("problems", [])]
    return obj
