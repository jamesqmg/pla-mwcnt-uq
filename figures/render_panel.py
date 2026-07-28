"""figures/render_panel.py — render HTML/CSS/SVG panels to print-grade PNG + vector PDF.

Part of the figure pipeline for *designed* (non-data-plot) panels: key-findings
summary boxes, method-selection guides, schematic callouts. Data plots stay in
matplotlib (see ``style.py``). HTML panels give full CSS/typography control and
render via headless Chromium (Playwright) at high DPI — reproducible, scriptable,
version-controlled.

Usage:
    /Users/jimbo/miniconda3/bin/python3 figures/render_panel.py figures/panels/key_findings.html
    # writes <name>.png (4x raster) + <name>.pdf (vector) next to the .html

Requires: ``pip install playwright`` + ``playwright install chromium`` (done 2026-06-30).
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

SCALE = 4  # device pixel ratio for the PNG (4x ≈ crisp well past 300 DPI for print)


def render(html_path: str | Path, *, scale: int = SCALE, pdf: bool = True) -> None:
    """Render the ``.panel`` element of an HTML file to a high-DPI PNG and a vector PDF."""
    html_path = Path(html_path).resolve()
    out_base = html_path.with_suffix("")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(device_scale_factor=scale)
        page.goto(html_path.as_uri())
        page.wait_for_load_state("networkidle")
        el = page.query_selector(".panel") or page.query_selector("body")
        box = el.bounding_box()
        el.screenshot(path=str(out_base.with_suffix(".png")))
        if pdf:
            page.pdf(
                path=str(out_base.with_suffix(".pdf")),
                width=f"{box['width']}px",
                height=f"{box['height']}px",
                print_background=True,
                margin={"top": "0", "bottom": "0", "left": "0", "right": "0"},
            )
        browser.close()
    print(f"Wrote {out_base.with_suffix('.png')}")
    if pdf:
        print(f"Wrote {out_base.with_suffix('.pdf')}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        raise SystemExit("usage: render_panel.py <panel.html> [more.html ...]")
    for arg in args:
        render(Path(arg))
