"""
figures/deck_assets/bg.py — DECORATIVE slide backgrounds for the defense deck.

NOT a data figure (lives in deck_assets/, separate from figures/deck/* which are
honesty-bound). Produces full-bleed 16:9 atmosphere PNGs applied as slide
backgrounds in build_deck.py. "Subtle" intensity per BEAUTIFY-PLAN decision:
faint violet radial glow (top-left), a low whisper of sky depth (bottom-right),
fine film grain. The title/divider variant adds a ghosted dogbone + scatter motif
at ~7% opacity and a hairline rule.

Palette mirrors deck_style.py / build_deck.py exactly.
Run: ~/miniconda3/bin/python figures/deck_assets/bg.py
Out: Presentations/current/figures/bg_master.png, bg_title.png
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "Presentations" / "current" / "figures"

# 16:9 at a projector-friendly resolution (slide is 13.333"x7.5").
W, H = 2400, 1350

# ── palette (matches build_deck.py) ──────────────────────────────────────────
BG = (0x0F, 0x0D, 0x15)
VIOLET = (0xA7, 0x8B, 0xFA)
SKY = (0x7D, 0xD3, 0xFC)
TEAL = (0x2D, 0xD4, 0xBF)
FG = (0xF3, 0xF1, 0xF8)
LINE = (0x39, 0x33, 0x4A)


def _radial(cx, cy, radius):
    """Normalized gaussian radial falloff (1 at center → 0 far out)."""
    yy, xx = np.mgrid[0:H, 0:W]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return np.exp(-d2 / (2.0 * radius ** 2))


def _base_canvas(*, violet_amp=0.11, sky_amp=0.04):
    """Dark ground + tight violet corner glow + faint sky depth + grain.

    The ground (#0F0D15) must dominate — the glow is concentrated in the
    top-left corner with a fast falloff so >70% of the frame stays true dark.
    """
    canvas = np.zeros((H, W, 3), float)
    canvas[:] = BG

    # violet corner glow — tight sigma so it reads as a glow, not a wash
    g_v = _radial(W * 0.10, H * 0.06, W * 0.27)[..., None]
    canvas += np.array(VIOLET, float) * g_v * violet_amp

    # whisper of sky depth, bottom-right corner
    g_s = _radial(W * 0.96, H * 1.00, W * 0.30)[..., None]
    canvas += np.array(SKY, float) * g_s * sky_amp

    # fine film grain — very low amplitude so it reads as texture, not noise
    rng = np.random.default_rng(7)
    grain = rng.normal(0.0, 1.0, (H, W, 1))
    canvas += grain * 2.6

    return np.clip(canvas, 0, 255).astype(np.uint8)


def _dogbone_poly(cx, cy, w, h):
    """ASTM D638 tensile-coupon outline (matches fig_specimen_budget_deck)."""
    g, k, t = 0.46 * w, 0.20 * w, 0.5 * h
    pts = [
        (g, t), (g, 0.62 * t), (k, 0.42 * t), (k, -0.42 * t),
        (g, -0.62 * t), (g, -t),
        (-g, -t), (-g, -0.62 * t), (-k, -0.42 * t), (-k, 0.42 * t),
        (-g, 0.62 * t), (-g, t),
    ]
    return [(cx + px, cy - py) for px, py in pts]


def _title_motif(img):
    """Ghosted dogbone + scatter motif on a transparent overlay, ~7% opacity."""
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    # large ghosted dogbone, lower-right, faint violet outline
    a_line = 26  # ~10% alpha
    d.line(_dogbone_poly(W * 0.78, H * 0.60, W * 0.30, H * 0.92) +
           [_dogbone_poly(W * 0.78, H * 0.60, W * 0.30, H * 0.92)[0]],
           fill=VIOLET + (a_line,), width=3, joint="curve")

    # a second, smaller ghost behind it for layered depth
    d.line(_dogbone_poly(W * 0.62, H * 0.72, W * 0.20, H * 0.62) +
           [_dogbone_poly(W * 0.62, H * 0.72, W * 0.20, H * 0.62)[0]],
           fill=SKY + (18,), width=2, joint="curve")

    # scatter motif — a faint scatter of points drifting up-right (data → signal)
    rng = np.random.default_rng(11)
    n = 70
    xs = rng.uniform(W * 0.05, W * 0.55, n)
    ys = H * 0.78 - (xs - W * 0.05) * 0.34 + rng.normal(0, H * 0.045, n)
    for x, y in zip(xs, ys):
        r = rng.uniform(2.5, 5.5)
        a = int(rng.uniform(14, 40))
        col = VIOLET if rng.random() > 0.4 else SKY
        d.ellipse([x - r, y - r, x + r, y + r], fill=col + (a,))

    overlay = overlay.filter(ImageFilter.GaussianBlur(0.6))
    img.paste(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"),
              (0, 0))
    return img


def _hairline(img, *, color=LINE, alpha=140):
    """Thin horizontal hairline low in the frame (title-slide accent)."""
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    y = int(H * 0.80)
    d.line([(int(W * 0.07), y), (int(W * 0.40), y)], fill=color + (alpha,), width=2)
    img.paste(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"),
              (0, 0))
    return img


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # master content background — the quietest variant. JPEG: no alpha needed,
    # and grain makes PNG ~4MB vs ~250KB as JPEG (matters, embedded on every slide).
    master = Image.fromarray(_base_canvas(violet_amp=0.10, sky_amp=0.035))
    master.save(OUT / "bg_master.jpg", quality=90, optimize=True)
    print(f"Wrote: {OUT / 'bg_master.jpg'}")

    # title / divider background — clean glow only (no scatter / motif / hairline),
    # a touch stronger than master so dividers feel distinct.
    title = Image.fromarray(_base_canvas(violet_amp=0.15, sky_amp=0.055))
    title.save(OUT / "bg_title.jpg", quality=90, optimize=True)
    print(f"Wrote: {OUT / 'bg_title.jpg'}")


if __name__ == "__main__":
    main()
