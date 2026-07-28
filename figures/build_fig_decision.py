#!/usr/bin/env python3
"""figures/build_fig_decision.py — Fig 8, the method-selection decision guide.

Renders figures/fig_decision.html (a Swiss-Modernism decision flowchart on the
Allotrope palette — design direction from the ui-ux-pro-max library) to a
high-resolution PNG using the headless-Chromium shell already present in the
Playwright cache. Exact, selectable text; print-quality. Output: fig_decision.png

Superseded the earlier matplotlib two-panel (risk-posture plane + flow); the
plane was dropped when the flowchart became the whole figure. The standalone
plane/flow generators (build_fig_risk_posture.py, build_fig_method_selection.py)
are retained but no longer embedded.
"""
from __future__ import annotations
import glob
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
HTML = HERE / "fig_decision.html"
OUT = HERE / "fig_decision.png"
# Must match #fig width/height in fig_decision.html; SCALE sets the export DPI.
W, H, SCALE = 1000, 600, 3


def _find_chrome() -> str:
    patterns = [
        "~/Library/Caches/ms-playwright/chromium_headless_shell-*/"
        "chrome-headless-shell-*/chrome-headless-shell",
        "~/Library/Caches/ms-playwright/chromium-*/chrome-mac*/*/Chromium",
    ]
    for pat in patterns:
        hits = sorted(glob.glob(os.path.expanduser(pat)))
        if hits:
            return hits[-1]
    raise FileNotFoundError(
        "headless Chromium not found in the Playwright cache; run "
        "`python3 -m playwright install chromium`.")


def main() -> None:
    chrome = _find_chrome()
    subprocess.run(
        [chrome, "--headless", "--disable-gpu", "--hide-scrollbars",
         f"--force-device-scale-factor={SCALE}",
         f"--window-size={W},{H}",
         "--default-background-color=FFFFFFFF",
         f"--screenshot={OUT}",
         HTML.as_uri()],
        check=True, capture_output=True,
    )
    print(f"Wrote: {OUT}  ({W * SCALE}x{H * SCALE})")


if __name__ == "__main__":
    main()
