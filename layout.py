"""Compose a meme from an art-director spec: canvas, panels, header band, positioned text."""
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_DIR = Path(__file__).parent / "fonts"
# name -> (file, variable-font weight or None, force uppercase)
FONTS = {
    "anton":   ("Anton-Regular.ttf", None, True),      # classic impact-style meme
    "bangers": ("Bangers-Regular.ttf", None, True),    # comic book
    "marker":  ("PermanentMarker-Regular.ttf", None, False),  # handwritten
    "oswald":  ("Oswald.ttf", b"Bold", True),          # condensed news/poster
    "inter":   ("Inter.ttf", b"SemiBold", False),      # clean tweet / modern UI
    "comic":   ("ComicNeue-Bold.ttf", None, False),    # friendly, comic sans-like
}
ASPECTS = {"1:1": (1024, 1024), "4:5": (928, 1152), "16:9": (1344, 768), "9:16": (768, 1344), "3:2": (1248, 832)}
LAYOUTS = {"single": 1, "side_by_side": 2, "stacked": 2}
POSITIONS = {  # x1, y1, x2, y2 as fractions of the panel; v-align
    "top":          ((0.03, 0.02, 0.97, 0.24), "top"),
    "bottom":       ((0.03, 0.76, 0.97, 0.98), "bottom"),
    "center":       ((0.08, 0.36, 0.92, 0.64), "center"),
    "top-left":     ((0.03, 0.03, 0.55, 0.30), "top"),
    "top-right":    ((0.45, 0.03, 0.97, 0.30), "top"),
    "bottom-left":  ((0.03, 0.70, 0.55, 0.97), "bottom"),
    "bottom-right": ((0.45, 0.70, 0.97, 0.97), "bottom"),
}
SIZES = {"small": 0.055, "medium": 0.08, "large": 0.11}
GAP = 8


def font(name, size):
    f, weight, _ = FONTS.get(name, FONTS["anton"])
    ft = ImageFont.truetype(str(FONT_DIR / f), size)
    if weight:
        ft.set_variation_by_name(weight)
    return ft


def color(c, default):
    if not isinstance(c, str) or not c.strip():
        return default
    c = c.strip()
    try:
        from PIL import ImageColor
        return ImageColor.getrgb(c if c.startswith("#") or not all(ch in "0123456789abcdefABCDEF" for ch in c) else "#" + c)
    except ValueError:
        return default


def _lum(c):
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def readable(fill, bg):
    """Keep fill if it contrasts with bg, else pick black/white."""
    if bg is None or abs(_lum(fill) - _lum(bg)) >= 110:
        return fill
    return (15, 15, 15) if _lum(bg) > 128 else (255, 255, 255)


def r16(x):
    return max(512, int(round(x / 16)) * 16)


def panel_boxes(layout, W, H):
    """Pixel boxes of each panel on the canvas (before any header band)."""
    if layout == "side_by_side":
        w = (W - GAP) // 2
        return [(0, 0, w, H), (w + GAP, 0, W, H)]
    if layout == "stacked":
        h = (H - GAP) // 2
        return [(0, 0, W, h), (0, h + GAP, W, H)]
    return [(0, 0, W, H)]


def gen_size(w, h, megapixels=1.05):
    """Image-model resolution for a panel of w x h (keeps aspect, ~1MP, multiples of 16)."""
    s = math.sqrt(megapixels * 1e6 / (w * h))
    return r16(w * s), r16(h * s)


def _wrap(draw, text, ft, max_w):
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if cur and draw.textlength(trial, font=ft) > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return lines + ([cur] if cur else [])


def draw_text(img, text, box, fname="anton", fill=(255, 255, 255), outline=(0, 0, 0), bg=None,
              size="medium", valign="center", align="center", ref=None, dry=False):
    """Fit `text` into pixel `box` (wrap + shrink) and draw it.
    Returns the rectangle actually covered. dry=True only measures."""
    if not text or not text.strip():
        return None
    if FONTS.get(fname, FONTS["anton"])[2]:
        text = text.upper()
    fill = readable(fill, bg)
    if not outline and not bg:  # guarantee contrast on busy images
        lum = 0.299 * fill[0] + 0.587 * fill[1] + 0.114 * fill[2]
        outline = (0, 0, 0) if lum > 110 else (255, 255, 255)
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = box
    pad = int(min(x2 - x1, y2 - y1) * 0.06) if bg else 0
    bw, bh = (x2 - x1) - 2 * pad, (y2 - y1) - 2 * pad
    ref = ref or min(img.size)
    px = int(SIZES.get(size, SIZES["medium"]) * ref)
    floor = max(14, int(0.042 * ref))  # never shrink below phone-readable size
    while True:
        ft = font(fname, px)
        asc, desc = ft.getmetrics()
        lh = int((asc + desc) * 1.02)
        lines = _wrap(draw, text, ft, bw)
        widest = max(draw.textlength(l, font=ft) for l in lines)
        if (len(lines) * lh <= bh and widest <= bw) or px <= floor:
            break
        px -= 2
    if len(lines) * lh > bh:  # still too tall at the floor size: grow the box inside the image
        H_img, grow = img.size[1], len(lines) * lh - bh
        y1, y2 = (y1, y2 + grow) if valign == "top" else (y1 - grow, y2) if valign == "bottom" else (y1 - grow // 2, y2 + grow - grow // 2)
        shift = max(0, -y1) - max(0, y2 - H_img)
        y1, y2 = y1 + shift, y2 + shift
        bh = y2 - y1 - 2 * pad
    if len(lines) > 1:  # balance: narrowest width that keeps the same number of lines
        w = bw
        while w > bw * 0.35:
            w *= 0.95
            cand = _wrap(draw, text, ft, w)
            if len(cand) > len(lines) or max(draw.textlength(l, font=ft) for l in cand) > w:
                break
            lines = cand
    th = len(lines) * lh
    widest = max(draw.textlength(l, font=ft) for l in lines)
    ty = {"top": y1 + pad, "bottom": y2 - pad - th}.get(valign, (y1 + y2 - th) // 2)
    stroke = max(2, px // 12) if outline else 0
    cx = {"left": x1 + pad + widest / 2, "right": x2 - pad - widest / 2}.get(align, (x1 + x2) / 2)
    p = max(6, px // 4) if bg else stroke
    rect = (cx - widest / 2 - p, ty - p // 2, cx + widest / 2 + p, ty + th + p // 2)
    if dry:
        return rect
    if bg:
        draw.rounded_rectangle(rect, radius=p, fill=bg)
    for ln in lines:
        lw = draw.textlength(ln, font=ft)
        tx = {"left": x1 + pad, "right": x2 - pad - lw}.get(align, (x1 + x2 - lw) / 2)
        draw.text((tx, ty), ln, font=ft, fill=fill, stroke_width=stroke,
                  stroke_fill=outline if outline else None)
        ty += lh
    return rect


def _overlap(a, b, gap=6):
    return not (a[2] + gap <= b[0] or b[2] + gap <= a[0] or a[3] + gap <= b[1] or b[3] + gap <= a[1])


def _candidates(box, occupied, pw, ph):
    """The requested box first, then the same box slid below/above each occupied rect."""
    x1, y1, x2, y2 = box
    h = y2 - y1
    yield box, None
    for o in occupied:
        for ny1 in (o[3] + 10, o[1] - 10 - h):
            ny1 = int(min(max(0, ny1), ph - h))
            yield (x1, ny1, x2, ny1 + h), "center"
        # squeeze into the free space below / above if a full-height box does not fit
        if ph - o[3] - 14 > ph * 0.08:
            yield (x1, int(o[3] + 10), x2, ph - 4), "top"
        if o[1] - 14 > ph * 0.08:
            yield (x1, 4, x2, int(o[1] - 10)), "bottom"


def header_band(text, W, spec):
    """Tweet-style text band (returns an Image) placed above the panels."""
    fname = spec.get("font", "inter")
    fill = color(spec.get("color"), (15, 15, 15))
    bg = color(spec.get("bg"), (255, 255, 255))
    fill = readable(fill, bg)
    px = int(W * 0.042)
    ft = font(fname, px)
    tmp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    if FONTS.get(fname, FONTS["inter"])[2]:
        text = text.upper()
    lines = _wrap(tmp, text, ft, W * 0.92)
    asc, desc = ft.getmetrics()
    lh = int((asc + desc) * 1.15)
    band = Image.new("RGB", (W, int(lh * len(lines) + W * 0.06)), bg)
    d = ImageDraw.Draw(band)
    y = int(W * 0.03)
    align = spec.get("align", "left")
    for ln in lines:
        lw = d.textlength(ln, font=ft)
        x = W * 0.04 if align == "left" else (W - lw) / 2
        d.text((x, y), ln, font=ft, fill=fill)
        y += lh
    return band


def text_box(t, pw, ph, grounded):
    """Pixel box + v-align for one text block inside a panel."""
    if t.get("position") == "on_object" and grounded:
        x1, y1, x2, y2 = grounded
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w = max((x2 - x1) * 1.1, pw * 0.28)
        h = ph * 0.16
        bx1 = min(max(0, cx - w / 2), pw - w)
        by1 = min(max(0, cy - h / 2), ph - h)
        return (int(bx1), int(by1), int(bx1 + w), int(by1 + h)), "center"
    reg = t.get("region")
    if isinstance(reg, list) and len(reg) == 4 and all(isinstance(v, (int, float)) for v in reg):
        x1, y1, x2, y2 = [min(100, max(0, v)) / 100 for v in reg]
        if x2 - x1 > 0.08 and y2 - y1 > 0.05:
            return (int(x1 * pw), int(y1 * ph), int(x2 * pw), int(y2 * ph)), "center"
    pos = t.get("position") if t.get("position") in POSITIONS else "bottom"
    (fx1, fy1, fx2, fy2), va = POSITIONS[pos]
    return (int(fx1 * pw), int(fy1 * ph), int(fx2 * pw), int(fy2 * ph)), va


def compose(spec, panel_images, grounding=None):
    """spec: validated art-director spec. panel_images: PIL images (any size, resized here).
    grounding: {(panel_idx, text_idx): (x1,y1,x2,y2) in panel pixels}."""
    grounding = grounding or {}
    W, H = ASPECTS[spec["aspect"]]
    bg = color(spec.get("background"), (255, 255, 255))
    canvas = Image.new("RGB", (W, H), bg)
    for pi, ((x1, y1, x2, y2), panel, im) in enumerate(zip(panel_boxes(spec["layout"], W, H), spec["panels"], panel_images)):
        pw, ph = x2 - x1, y2 - y1
        tile = im.convert("RGB").resize((pw, ph), Image.LANCZOS)
        occupied = []
        for ti, t in enumerate(panel.get("texts", [])):
            box, va = text_box(t, pw, ph, grounding.get((pi, ti)))
            on_obj = t.get("position") == "on_object"
            kw = dict(fname=t.get("font", "anton"), fill=color(t.get("color"), (255, 255, 255)),
                      outline=color(t.get("outline"), None) if t.get("outline") not in (None, "", "none") else None,
                      bg=color(t.get("bg"), None) if t.get("bg") not in (None, "", "none") else None,
                      size=t.get("size", "small" if on_obj else "medium"),
                      align=t.get("align", "center"), ref=min(W, H))
            chosen = None
            for cand, cva in _candidates(box, occupied, pw, ph):
                rect = draw_text(tile, t["text"], cand, valign=cva or va, dry=True, **kw)
                if rect and not any(_overlap(rect, o) for o in occupied):
                    chosen = (cand, cva or va)
                    break
            cand, cva = chosen or (box, va)
            rect = draw_text(tile, t["text"], cand, valign=cva, **kw)
            if rect:
                occupied.append(rect)
        canvas.paste(tile, (x1, y1))
    hdr = spec.get("header")
    if isinstance(hdr, dict) and hdr.get("text"):
        band = header_band(hdr["text"], W, hdr)
        full = Image.new("RGB", (W, H + band.height), bg)
        full.paste(band, (0, 0))
        full.paste(canvas, (0, band.height))
        canvas = full
    return canvas
