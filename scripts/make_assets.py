"""Regenerate every image in the README from nanoqalgos.py itself.

    pip install pillow && python scripts/make_assets.py [gif] [svg]

  assets/demo.gif      Grover search, amplitude by amplitude ("inversion about the mean")
  assets/qae.svg       amplitude estimation vs Monte Carlo: error against cost
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import nanoqalgos as q  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(OUT, exist_ok=True)
INK, PAPER, PANEL, MUTED, LINE = "#1B2430", "#EEF1F4", "#FFFFFF", "#5B6675", "#C9D1DB"
BAR, HOT, OK = "#2163B0", "#D9483B", "#2E9E62"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans{}.ttf"


def font(size, bold=False):
    try:
        return ImageFont.truetype(FONT.format("-Bold" if bold else ""), size)
    except OSError:
        return ImageFont.load_default()


# ------------------------------------------------------------------ the GIF ---
S = 2  # supersampling for smooth edges
W, H = 760, 520
L, R, T, B = 70, 30, 96, 120  # plot margins


def frame(amps, marked, title, caption, prob):
    img = Image.new("RGB", (W * S, H * S), PAPER)
    g = ImageDraw.Draw(img)
    Y = lambda v: S * (T + (1 - v) / 2 * (H - T - B))  # noqa: E731  amplitude in [-1, 1]
    for v in (-1, -0.5, 0, 0.5, 1):
        g.line([S * L, Y(v), S * (W - R), Y(v)], fill=LINE if v else MUTED, width=S)
        g.text((S * (L - 10), Y(v)), f"{v:+.1f}" if v else "0", fill=MUTED,
               font=font(13 * S), anchor="rm")
    n, bw = len(amps), (W - L - R) / len(amps)
    for i, a in enumerate(amps):
        x0 = S * (L + i * bw + bw * 0.14)
        x1 = S * (L + (i + 1) * bw - bw * 0.14)
        y0, y1 = sorted([Y(a), Y(0)])
        g.rectangle([x0, y0, x1, max(y1, y0 + S)], fill=HOT if i == marked else BAR)
    mean = float(np.mean(amps))
    for x in range(L, W - R, 14):  # dashed mean line
        g.line([S * x, Y(mean), S * min(x + 7, W - R), Y(mean)], fill=INK, width=2 * S)
    g.text((S * (W - R), Y(mean) - 8 * S), "mean", fill=INK, font=font(13 * S), anchor="rb")
    g.text((S * 30, S * 22), title, fill=INK, font=font(24 * S, True))
    g.text((S * 30, S * 56), f"P(answer) = {prob:.3f}", fill=HOT if prob < .9 else OK,
           font=font(18 * S, True))
    g.text((S * 30, S * (H - 82)), caption, fill=INK, font=font(19 * S))
    g.text((S * 30, S * (H - 44)), f"Grover search · {n} items · 1 marked (red) · nanoqalgos",
           fill=MUTED, font=font(15 * S))
    return img.resize((W, H), Image.LANCZOS)


def make_gif(n=5, marked=13, tween=7):
    _, theta, steps = q.grover(n, [marked], trace=True)
    text = {"start": ("Start: every item equally likely", "All amplitudes are 1/√N. Measuring now succeeds 1 time in 32."),
            "oracle": ("Oracle: flip the answer's sign", "Probabilities are unchanged, but the mean amplitude drops."),
            "diffusion": ("Diffusion: reflect about the mean", "a → 2·mean − a. The flipped bar shoots far above the rest.")}
    frames, durations = [], []
    prev = None
    for name, amps in steps:
        kind = name.split()[0]
        title, caption = text[kind]
        if kind != "start":
            title = f"Round {name.split()[1]} · " + title
        if prev is not None:  # slide smoothly from the previous amplitudes
            for s in np.linspace(0, 1, tween + 1)[1:-1]:
                mix = (1 - s) * prev + s * amps
                frames.append(frame(mix, marked, title, caption, mix[marked] ** 2))
                durations.append(60)
        frames.append(frame(amps, marked, title, caption, amps[marked] ** 2))
        durations.append(1500)
        prev = amps
    k = (len(steps) - 1) // 2
    frames.append(frame(steps[-1][1], marked, f"Measure: found item {marked} after {k} queries",
                        f"Exact: sin²((2k+1)θ) = {np.sin((2 * k + 1) * theta) ** 2:.4f}.  "
                        f"A classical search needs {(2 ** n + 1) / 2:.1f} on average.",
                        steps[-1][1][marked] ** 2))
    durations.append(3000)
    frames[0].save(os.path.join(OUT, "demo.gif"), save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)


# ---------------------------------------------------------------- the chart ---
def make_svg(ts=range(3, 13), confidence=0.8):
    p, f, scale, _ = q.european_call()
    rows = []
    for t in ts:
        est, probs, a = q.amplitude_estimation(p, f, t)
        calls = 2 * (2 ** t - 1) + 1
        rows.append((calls, q.error_at_confidence(est, probs, a, confidence) * scale,
                     q.monte_carlo_error(a, calls, confidence) * scale))
        print(f"t={t:2d} calls={calls:5d}  QAE ±{rows[-1][1]:.4f}  MC ±{rows[-1][2]:.4f}")
    calls, eq, emc = map(np.array, zip(*rows))
    w, h, l, r, t_, b = 720, 440, 74, 24, 30, 56
    x0, x1, y0, y1 = np.log10(10), np.log10(1e4), -2, 1  # log axes: calls, error in $
    X = lambda c: l + (np.log10(c) - x0) / (x1 - x0) * (w - l - r)  # noqa: E731
    Y = lambda e: t_ + (y1 - np.log10(max(e, 10 ** y0))) / (y1 - y0) * (h - t_ - b)  # noqa: E731
    grid = "".join(f'<line x1="{l}" x2="{w - r}" y1="{Y(10.0 ** k):.1f}" y2="{Y(10.0 ** k):.1f}" class="g"/>'
                   f'<text x="{l - 8}" y="{Y(10.0 ** k) + 4:.1f}" class="t" text-anchor="end">${10.0 ** k:g}</text>'
                   for k in range(y0, y1 + 1))
    xt = "".join(f'<text x="{X(c):.1f}" y="{h - b + 22}" class="t" text-anchor="middle">{c:,}</text>'
                 for c in (10, 100, 1000, 10000))
    pts = lambda e: " ".join(f"{X(c):.1f},{Y(v):.1f}" for c, v in zip(calls, e))  # noqa: E731
    dots = "".join(f'<circle cx="{X(c):.1f}" cy="{Y(v):.1f}" r="4" class="dq"/>' for c, v in zip(calls, eq))
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="Atkinson Hyperlegible, DejaVu Sans, sans-serif">
<style>
  .bg{{fill:#FFFFFF}} .g{{stroke:#C9D1DB;stroke-width:1}} .t{{fill:#5B6675;font-size:13px}} .k{{fill:#1B2430;font-size:14px;font-weight:700}}
  .mc{{stroke:#5B6675;fill:none;stroke-width:3}} .q{{stroke:#2E9E62;fill:none;stroke-width:3}} .dq{{fill:#2E9E62}}
  @media (prefers-color-scheme: dark) {{ .bg{{fill:#1E2631}} .g{{stroke:#34404F}} .t{{fill:#9AA6B4}} .k{{fill:#E7ECF2}} .mc{{stroke:#9AA6B4}} .q{{stroke:#4CC487}} .dq{{fill:#4CC487}} }}
</style>
<rect class="bg" width="{w}" height="{h}" rx="12"/>
{grid}{xt}
<text x="{(l + w - r) / 2}" y="{h - 8}" class="t" text-anchor="middle">calls to the pricing circuit (log scale)</text>
<text x="{l + 12}" y="{h - b - 14}" class="k">option price error at 80% confidence (exact)</text>
<polyline points="{pts(emc)}" class="mc"/>
<polyline points="{pts(eq)}" class="q"/>{dots}
<text x="{X(calls[-3]):.1f}" y="{Y(emc[-3]) - 22:.1f}" class="t" text-anchor="middle">Monte Carlo ∝ 1/√M</text>
<text x="{X(calls[-4]):.1f}" y="{Y(eq[-4]) + 26:.1f}" class="t" text-anchor="middle">amplitude estimation ∝ 1/M</text>
<text x="{X(calls[-1]) - 6:.1f}" y="{Y(eq[-1]) + 22:.1f}" class="k" text-anchor="end">{emc[-1] / eq[-1]:.0f}× smaller at {calls[-1]:,} calls</text>
</svg>'''
    open(os.path.join(OUT, "qae.svg"), "w").write(svg)


if __name__ == "__main__":
    what = sys.argv[1:] or ["gif", "svg"]
    if "gif" in what:
        make_gif()
    if "svg" in what:
        make_svg()
