#!/usr/bin/env python
"""Agentic meme loop: writer -> art director -> image model -> (grounding) -> compose -> critic -> revise.

Every invocation produces N (default 10) NEW memes per brief, never repeating earlier ones
(checked against output/all_memes.csv by text similarity and image perceptual hash),
and writes a CSV with every detail.

  python memeloop.py --brief "Superstition: ... Give 5 scenarios where it is not harmful and where it is harmful"
  python memeloop.py --briefs briefs.txt            # 10 new memes for every brief in the file
  python memeloop.py --brief "..." --batches 5      # keep looping: 5 x 10 new memes
"""
import argparse
import csv
import difflib
import hashlib
import json
import random
import re
import secrets
import time
from datetime import datetime
from pathlib import Path

from PIL import Image

import agents as A
from layout import ASPECTS, compose, gen_size, panel_boxes

OUT = Path(__file__).parent / "output"
MASTER = OUT / "all_memes.csv"
FIELDS = ["meme_id", "run_id", "created_at", "brief_id", "brief", "label", "scenario", "joke", "captions",
          "visual_idea", "format", "layout", "aspect", "style", "header_text", "panel_scenes", "texts",
          "image_model", "seeds", "rounds", "status", "critic_score", "critic_problems", "image_path", "spec_path",
          "dhash"]


def log(*a):
    print(f"[{datetime.now():%H:%M:%S}]", *a, flush=True)


def brief_id(b):
    return hashlib.sha1(b.strip().encode()).hexdigest()[:10]


def norm(s):
    return re.sub(r"[^a-z0-9 ]+", "", s.lower()).strip()


def dhash(img, size=8):
    g = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    px = list(g.getdata())
    bits = [px[r * (size + 1) + c] > px[r * (size + 1) + c + 1] for r in range(size) for c in range(size)]
    return sum(1 << i for i, b in enumerate(bits) if b)


def hamming(a, b):
    return bin(a ^ b).count("1")


def load_history():
    if not MASTER.exists():
        return []
    with MASTER.open(newline="") as f:
        return list(csv.DictReader(f))


def concept_key(c):
    return norm(c["scenario"] + " " + " ".join(c["captions"]))


def is_dup(key, keys, thr=0.72):
    return any(difflib.SequenceMatcher(None, key, k).ratio() >= thr for k in keys)


def fallback_spec(concept, style):
    caps = concept["captions"]
    return {"format": "classic", "aspect": "16:9", "layout": "single", "style": style, "background": "#ffffff",
            "header": None,
            "panels": [{"scene": concept["visual_idea"] + " Leave empty space at the top and bottom.",
                        "texts": [{"text": caps[0], "position": "top", "font": "anton", "color": "#ffffff", "outline": "#000000"}]
                        + ([{"text": " ".join(caps[1:]), "position": "bottom", "font": "anton", "color": "#ffffff",
                             "outline": "#000000"}] if len(caps) > 1 else [])}]}


class MemeLoop:
    def __init__(self, args):
        from images import ImageBackend
        from llm import LLM
        self.args = args
        log(f"loading LLM {args.llm} on cuda:{args.llm_gpu}")
        self.llm = LLM(args.llm, f"cuda:{args.llm_gpu}")
        self.director = self.llm if args.director_llm in (None, args.llm) else LLM(args.director_llm, f"cuda:{args.llm_gpu}")
        log(f"loading image model {args.model} on cuda:{args.image_gpu}")
        self.img = ImageBackend(args.model, f"cuda:{args.image_gpu}")
        self.cache = {}

    # ---------------------------------------------------------- writer + uniqueness
    def write_concepts(self, brief, n, history):
        bid = brief_id(brief)
        prev = [r for r in history if r["brief_id"] == bid]
        avoid = [f'{r["scenario"]} | {r["captions"]}' for r in prev]
        keys = [norm(r["scenario"] + " " + r["captions"].replace(" | ", " ")) for r in prev]
        chosen, labels, quota = [], None, None
        for attempt in range(5):
            if quota is not None:
                short = {l: q - sum(c["label"] == l for c in chosen) for l, q in quota.items()}
                short = {l: k for l, k in short.items() if k > 0}
                if not short:
                    break
            elif len(chosen) >= n:
                break
            used = avoid + [f'{c["scenario"]} | {" | ".join(c["captions"])}' for c in chosen]
            if quota is None:
                conv = A.writer_conv(brief, n, used)
            else:  # targeted refill of the categories that are short
                ask = sum(short.values())
                conv = A.writer_conv(brief, ask, used)
                conv[-1]["content"][0]["text"] += ("\nIMPORTANT: write ONLY these: "
                                                  + ", ".join(f'{k} memes with label "{l}"' for l, k in short.items()) + ".")
            got = self.llm.chat_json([conv], A.validate_writer, temperature=0.95, max_new_tokens=4096)[0] or []
            if labels is None and got:
                # categories = distinct labels of the first answer; equal share each
                labels = list(dict.fromkeys(c["label"] for c in got))
                quota = {l: n // len(labels) + (i < n % len(labels)) for i, l in enumerate(labels)}
            for c in got:
                if c["label"] not in quota:  # map near-miss labels ("harmful scenario") onto known ones
                    match = difflib.get_close_matches(c["label"], list(quota), n=1, cutoff=0.6)
                    if not match:
                        continue
                    c["label"] = match[0]
                if sum(x["label"] == c["label"] for x in chosen) >= quota[c["label"]]:
                    continue
                k = concept_key(c)
                if is_dup(k, keys):
                    log(f"  skip duplicate concept: {c['scenario'][:70]}")
                    continue
                keys.append(k)
                chosen.append(c)
        chosen.sort(key=lambda c: list(quota).index(c["label"]) if quota else 0)
        return chosen[:n]

    # ---------------------------------------------------------- rendering
    def render_panel(self, spec, pi, seed, w, h, seen_hashes):
        scene = spec["panels"][pi]["scene"]
        gw, gh = gen_size(w, h)
        for _ in range(3):
            key = hashlib.sha1(f"{scene}|{spec['style']}|{gw}x{gh}|{seed}|{self.args.model}".encode()).hexdigest()
            if key in self.cache:
                return self.cache[key], seed
            im = self.img(scene, gw, gh, seed, style=spec["style"])
            hsh = dhash(im)
            if not any(hamming(hsh, o) <= 5 for o in seen_hashes):
                seen_hashes.append(hsh)
                self.cache[key] = im
                return im, seed
            log("  near-duplicate image, re-seeding")
            seed = secrets.randbits(31)
        self.cache[key] = im
        return im, seed

    def render_all(self, items, seen_hashes):
        for it in items:
            spec = it["spec"]
            W, H = ASPECTS[spec["aspect"]]
            boxes = panel_boxes(spec["layout"], W, H)
            imgs, seeds = [], []
            for pi, (x1, y1, x2, y2) in enumerate(boxes):
                old = it["seeds"][pi] if pi < len(it["seeds"]) else secrets.randbits(31)
                im, s = self.render_panel(spec, pi, old, x2 - x1, y2 - y1, seen_hashes)
                imgs.append(im)
                seeds.append(s)
            it["panel_images"], it["seeds"] = imgs, seeds
        # grounding for on_object labels (batched over all panels that need it)
        jobs = []
        for ii, it in enumerate(items):
            it["grounding"] = {}
            for pi, p in enumerate(it["spec"]["panels"]):
                tis = [ti for ti, t in enumerate(p["texts"]) if t["position"] == "on_object"]
                if tis:
                    jobs.append((ii, pi, tis))
        if jobs:
            convs = [A.ground_conv(items[ii]["panel_images"][pi], [items[ii]["spec"]["panels"][pi]["texts"][t]["object"] for t in tis])
                     for ii, pi, tis in jobs]
            res = self.llm.chat_json(convs, A.validate_ground, temperature=0, max_new_tokens=400, retries=2)
            for (ii, pi, tis), boxes in zip(jobs, res):
                spec = items[ii]["spec"]
                W, H = ASPECTS[spec["aspect"]]
                x1, y1, x2, y2 = panel_boxes(spec["layout"], W, H)[pi]
                pw, ph = x2 - x1, y2 - y1
                for ti, b in zip(tis, boxes or []):
                    items[ii]["grounding"][(pi, ti)] = (b[0] / 1000 * pw, b[1] / 1000 * ph, b[2] / 1000 * pw, b[3] / 1000 * ph)
                for ti in tis[len(boxes or []):]:  # not found -> fall back to a fixed position
                    spec["panels"][pi]["texts"][ti]["position"] = "top" if ti % 2 == 0 else "bottom"
        for it in items:
            it["meme"] = compose(it["spec"], it["panel_images"], it["grounding"])

    # ---------------------------------------------------------- one batch for one brief
    def run_batch(self, brief, run_dir, run_id, history):
        a = self.args
        rng = random.Random(secrets.randbits(64))
        t0 = time.time()
        concepts = self.write_concepts(brief, a.n, history)
        log(f"writer: {len(concepts)} new concepts ({time.time() - t0:.0f}s)")
        for c in concepts:
            log(f"  [{c['label']}] {' / '.join(c['captions'])}")
        if not concepts:
            return []
        picks = A.pick_formats(len(concepts), rng)
        convs = [A.director_conv(c, brief, f, s) for c, (f, s) in zip(concepts, picks)]
        t = time.time()
        specs = self.director.chat_json(convs, A.validate_spec, temperature=0.8, max_new_tokens=2048)
        log(f"art director: {sum(s is not None for s in specs)}/{len(specs)} specs ({time.time() - t:.0f}s)")
        items = [{"concept": c, "conv": cv, "spec": s or fallback_spec(c, st), "seeds": [], "round": 0, "best": None}
                 for c, cv, s, (_, st) in zip(concepts, convs, specs, picks)]
        seen_hashes = [int(r["dhash"], 16) for r in history if r.get("dhash")]
        pending = items
        for rnd in range(1, a.rounds + 1):
            t = time.time()
            self.render_all(pending, seen_hashes)
            log(f"round {rnd}: rendered {len(pending)} memes ({time.time() - t:.0f}s)")
            t = time.time()
            crits = self.llm.chat_json([A.critic_conv(self._small(it["meme"]), it["concept"], it["spec"]) for it in pending],
                                       A.validate_critique, temperature=0, max_new_tokens=600)
            log(f"round {rnd}: critic ({time.time() - t:.0f}s) scores " + str([c["score"] if c else None for c in crits]))
            nxt = []
            for it, cr in zip(pending, crits):
                cr = cr or {"score": 0, "problems": ["critic failed to respond"], "fix_picture": "", "fix_layout": ""}
                it["round"] = rnd
                if it["best"] is None or cr["score"] > it["best"]["critique"]["score"]:
                    it["best"] = {"spec": json.loads(json.dumps(it["spec"])), "meme": it["meme"],
                                  "panels": it["panel_images"], "seeds": it["seeds"], "critique": cr, "round": rnd}
                if cr["score"] < a.accept and rnd < a.rounds:
                    it["critique"] = cr
                    nxt.append(it)
            if not nxt:
                break
            t = time.time()
            convs = [A.revise_conv(it["conv"], it["spec"], it["critique"]) for it in nxt]
            revised = self.director.chat_json(convs, A.validate_spec, temperature=0.7, max_new_tokens=2048)
            for it, s in zip(nxt, revised):
                if s:
                    # new scene -> new seed; unchanged scene -> reuse cached image
                    old = it["spec"]["panels"]
                    it["seeds"] = [it["seeds"][i] if i < len(old) and i < len(s["panels"]) and old[i]["scene"] == s["panels"][i]["scene"]
                                   else secrets.randbits(31) for i in range(len(s["panels"]))]
                    it["spec"] = s
            log(f"round {rnd}: director revised {len(nxt)} memes ({time.time() - t:.0f}s)")
            pending = nxt
        return self.save(items, brief, run_dir, run_id)

    @staticmethod
    def _small(im, side=1024):
        s = side / max(im.size)
        return im.resize((int(im.width * s), int(im.height * s))) if s < 1 else im

    def save(self, items, brief, run_dir, run_id):
        rows = []
        k0 = len(list((run_dir / "memes").glob("*.png")))
        for k, it in enumerate(items):
            b, c, s = it["best"], it["concept"], it["best"]["spec"]
            mid = f"{run_id}_{k0 + k:03d}"
            img_p, spec_p = run_dir / "memes" / f"{mid}.png", run_dir / "specs" / f"{mid}.json"
            b["meme"].save(img_p)
            for pi, p in enumerate(b["panels"]):
                p.save(run_dir / "panels" / f"{mid}_p{pi}.png")
            spec_p.write_text(json.dumps({"concept": c, "spec": s, "critique": b["critique"], "seeds": b["seeds"]},
                                         ensure_ascii=False, indent=1))
            rows.append({
                "meme_id": mid, "run_id": run_id, "created_at": datetime.now().isoformat(timespec="seconds"),
                "brief_id": brief_id(brief), "brief": brief, "label": c["label"], "scenario": c["scenario"],
                "joke": c["joke"], "captions": " | ".join(c["captions"]), "visual_idea": c["visual_idea"],
                "format": s.get("format", ""), "layout": s["layout"], "aspect": s["aspect"], "style": s["style"],
                "header_text": (s.get("header") or {}).get("text", ""),
                "panel_scenes": " || ".join(p["scene"] for p in s["panels"]),
                "texts": json.dumps([[{k2: t.get(k2) for k2 in ("text", "position", "object", "font", "size", "color")}
                                      for t in p["texts"]] for p in s["panels"]], ensure_ascii=False),
                "image_model": self.img.name, "seeds": " ".join(map(str, b["seeds"])), "rounds": it["round"],
                "status": "accepted" if b["critique"]["score"] >= self.args.accept else "needs_review",
                "critic_score": b["critique"]["score"], "critic_problems": " ; ".join(b["critique"]["problems"]),
                "image_path": str(img_p.relative_to(OUT.parent)), "spec_path": str(spec_p.relative_to(OUT.parent)),
                "dhash": f"{dhash(b['panels'][0]):016x}",
            })
        return rows


def write_csv(path, rows, append):
    if append and path.exists():  # migrate files written with an older column set
        with path.open(newline="") as f:
            r = csv.DictReader(f)
            if r.fieldnames != FIELDS:
                old = list(r)
                with path.open("w", newline="") as g:
                    w = csv.DictWriter(g, fieldnames=FIELDS, extrasaction="ignore")
                    w.writeheader()
                    w.writerows(old)
    new = not path.exists() or not append
    with path.open("a" if append else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--brief", action="append", help="brief text (repeatable)")
    ap.add_argument("--briefs", help=".txt file, one brief per line")
    ap.add_argument("-n", type=int, default=10, help="new memes per brief per batch")
    ap.add_argument("--batches", type=int, default=1, help="repeat the whole thing this many times (0 = forever)")
    ap.add_argument("--rounds", type=int, default=3, help="max critic->revise rounds per meme")
    ap.add_argument("--accept", type=int, default=8, help="critic score (1-10) that accepts a meme")
    ap.add_argument("--model", default="zimage", choices=["zimage", "qwen"], help="image model")
    ap.add_argument("--llm", default="Qwen/Qwen3.5-9B", help="writer / critic / grounding model (must be vision-capable)")
    ap.add_argument("--director-llm", default=None, help="separate model for the art director (default: same as --llm)")
    ap.add_argument("--llm-gpu", default="0")
    ap.add_argument("--image-gpu", default="1")
    args = ap.parse_args()

    briefs = list(args.brief or [])
    if args.briefs:
        briefs += [l.strip() for l in Path(args.briefs).read_text().splitlines() if l.strip()]
    if not briefs:
        ap.error("give --brief or --briefs")

    loop = MemeLoop(args)
    b = 0
    while args.batches == 0 or b < args.batches:
        b += 1
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = OUT / run_id
        for sub in ("memes", "panels", "specs"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
        all_rows = []
        for brief in briefs:
            log(f"=== batch {b} | {brief[:90]}")
            rows = loop.run_batch(brief, run_dir, run_id, load_history())
            write_csv(run_dir / "memes.csv", rows, append=True)
            write_csv(MASTER, rows, append=True)
            all_rows += rows
        log(f"batch {b} done: {len(all_rows)} memes -> {run_dir}/memes, CSV {run_dir}/memes.csv (master: {MASTER})")


if __name__ == "__main__":
    main()
