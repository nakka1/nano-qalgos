"""Command line front-end for nanoqalgos.  `nanoqalgos demo | bench | shor N | grover n`.

Everything printed here is computed live by nanoqalgos.py — nothing is hard-coded.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

import nanoqalgos as q

COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
CUBE = (8, [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)])  # 3-regular, triangle-free, max cut 12
NUCLEAR_REPULSION = 0.7199689944489797  # H2 at 0.735 Å, Hartree


def paint(text, code):
    return f"\033[{code}m{text}\033[0m" if COLOR else text


ok = lambda s: paint(s, "32;1")  # noqa: E731
dim = lambda s: paint(s, "90")  # noqa: E731


def bar(x, width=40):
    full = int(round(abs(x) * width))
    return ("█" * full).ljust(width)


# --------------------------------------------------------------- the checks ---
# Each returns a dict; demo prints it as prose, bench as a markdown row.

def check_qft(n=12):
    rng = np.random.default_rng(0)
    v = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    v /= np.linalg.norm(v)
    out = q.qft(v.reshape((2,) * n)).reshape(-1)
    err = np.abs(out - np.sqrt(2 ** n) * np.fft.ifft(v)).max()
    return dict(n=n, gates=n * (n + 1) // 2 + n // 2, matrix=4 ** n, err=err)


def check_grover(n=16):
    marked = [12345 % 2 ** n]
    psi, theta = q.grover(n, marked)
    k = int(np.floor(np.pi / (4 * theta)))
    p = q.probabilities(psi)[marked].sum()
    return dict(n=n, N=2 ** n, k=k, p=p, formula=np.sin((2 * k + 1) * theta) ** 2,
                classical=(2 ** n + 1) / 2)


def check_shor(N, a):
    probs, t = q.order_finding(a, N)
    good = [y for y in range(len(probs)) if q.factors_from_measurement(y, t, a, N)[1]]
    _, fac = q.factors_from_measurement(max(good, key=lambda y: probs[y]), t, a, N)
    r = next(k for k in range(1, N) if pow(a, k, N) == 1)  # true order, for display
    return dict(N=N, a=a, t=t, qubits=t + (N - 1).bit_length(), r=r,
                factors=tuple(sorted(fac)), p=probs[good].sum())


def check_vqe():
    exact = np.linalg.eigvalsh(q.hamiltonian_matrix(q.H2))[0]
    theta, hist = q.vqe(q.H2)
    return dict(E=hist[-1] + NUCLEAR_REPULSION, exact=exact + NUCLEAR_REPULSION,
                err_mha=abs(hist[-1] - exact) * 1000, evals=len(hist) * 2 * len(theta))


def check_qaoa():
    n, edges = CUBE
    g, b, val = q.qaoa_maxcut(n, edges)
    best = q.cut_values(n, edges).max()
    return dict(ratio=val / best, gap=abs(val - q.qaoa_p1_formula(n, edges, g, b)),
                gamma=g, beta=b, best=best)


def check_qae(t=11, confidence=0.8):
    p, f, scale, bs = q.european_call()
    est, probs, a = q.amplitude_estimation(p, f, t)
    calls = 2 * (2 ** t - 1) + 1  # each Q uses A and A†, plus the first A
    e_q = q.error_at_confidence(est, probs, a, confidence)
    e_mc = q.monte_carlo_error(a, calls, confidence)
    z = 1.2815515655446004  # two-sided 80% normal quantile, to size MC for equal error
    mc_needed = a * (1 - a) * (z / e_q) ** 2
    return dict(t=t, calls=calls, a=a, price=scale * a, bs=bs, err_q=e_q * scale,
                err_mc=e_mc * scale, mc_needed=mc_needed, scale=scale)


# --------------------------------------------------------------------- demo ---
def cmd_demo(args):
    print(ok("nano-qalgos"), "· the quantum algorithms industry uses, from scratch\n")
    c = check_qft()
    print(f"QFT     {c['n']} qubits with {c['gates']} gates reproduce np.fft "
          f"(a {c['matrix']:,}-entry matrix):  max difference {c['err']:.1e}")
    phi, t = 0.3141, 8
    P = q.phase_estimation(np.diag([np.exp(2j * np.pi * phi), 1]), np.array([1, 0]), t)
    print(f"QPE     φ = {phi} read with {t} qubits: most likely y/2^{t} = "
          f"{P.argmax() / 2 ** t:.4f}  (P = {P.max():.2f})")
    c = check_grover(10)
    print(f"Grover  1 item among {c['N']:,}: {c['k']} oracle calls, P(found) = {c['p']:.4f} "
          f"(formula sin²((2k+1)θ) = {c['formula']:.4f}); a classical search needs "
          f"{c['classical']:.0f} on average")
    c = check_shor(21, 2)
    print(f"Shor    N = 21, a = 2: order r = {c['r']} → 21 = {c['factors'][0]} × "
          f"{c['factors'][1]}   ({c['qubits']} qubits, P(success per run) = {c['p']:.2f})")
    c = check_vqe()
    print(f"VQE     H₂ ground energy {c['E']:.6f} Ha vs exact {c['exact']:.6f} Ha "
          f"(off by {c['err_mha']:.1e} mHa; chemical accuracy is 1.6 mHa)")
    c = check_qaoa()
    print(f"QAOA    MaxCut on the cube graph, p = 1: {c['ratio']:.4f} of the optimum "
          f"(simulator vs closed form: {c['gap']:.0e})")
    c = check_qae()
    print(f"QAE     European call: price on the grid {c['price']:.4f} (Black-Scholes {c['bs']:.4f}); "
          f"with {c['calls']:,} calls, 80% error ±{c['err_q']:.4f} vs Monte Carlo ±{c['err_mc']:.4f}")
    print(dim("\nnext:  nanoqalgos bench · nanoqalgos shor 35 · nanoqalgos grover 4 · "
              "read nanoqalgos.py"))


# -------------------------------------------------------------------- bench ---
def bench_rows():
    rows = []
    t0 = time.perf_counter()
    c = check_qft()
    rows.append(("QFT", f"{c['n']} qubits", f"{c['matrix']:,} matrix entries (DFT)",
                 f"{c['gates']} gates", f"= np.fft to {c['err']:.0e}"))
    c = check_grover()
    rows.append(("Grover", f"1 of {c['N']:,} items", f"{c['classical']:,.1f} queries on average",
                 f"**{c['k']} queries**, P = {c['p']:.5f}", f"= sin²((2k+1)θ) to {abs(c['p'] - c['formula']):.0e}"))
    for N, a in ((15, 7), (21, 2), (35, 3)):
        c = check_shor(N, a)
        rows.append(("Shor", f"factor {N}", "sub-exponential (GNFS) at scale",
                     f"{c['qubits']} qubits, P(success) = {c['p']:.2f}/run",
                     f"r = {c['r']} → {c['factors'][0]} × {c['factors'][1]}"))
    c = check_vqe()
    rows.append(("VQE", "H₂ ground energy", "exact diagonalization (2ⁿ × 2ⁿ)",
                 f"{c['evals']} circuit runs", f"{c['E']:.6f} Ha, off by {c['err_mha']:.0e} mHa"))
    c = check_qaoa()
    rows.append(("QAOA", "MaxCut, 3-regular cube", "random cut: 0.5 of optimum",
                 f"**{c['ratio']:.4f}** of optimum (p = 1)", f"= closed form to {c['gap']:.0e}"))
    c = check_qae()
    rows.append(("Amplitude est.", "price a European call", f"Monte Carlo: ±{c['err_mc']:.4f} with {c['calls']:,} samples",
                 f"**±{c['err_q']:.4f}** with {c['calls']:,} calls", f"MC needs ≈{c['mc_needed']:,.0f} samples to match"))
    return rows, time.perf_counter() - t0


def cmd_bench(args):
    rows, secs = bench_rows()
    head = ["algorithm", "problem", "classical", "quantum (simulated exactly)", "check"]
    print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
    for r in rows:
        print("| " + " | ".join(r) + " |")
    print(f"\nAll rows computed in {secs:.1f} s on one CPU core, numpy {np.__version__}. "
          "Errors are at 80% confidence, from exact distributions.")


# --------------------------------------------------------------- shor / grover ---
def cmd_shor(args):
    N = args.N
    a = args.a or next(x for x in range(2, N) if math.gcd(x, N) == 1)
    if N.bit_length() > 7:
        sys.exit(f"N = {N} needs {3 * N.bit_length()} qubits: too many for a laptop state vector")
    probs, t = q.order_finding(a, N)
    print(f"Order finding for a = {a} mod {N}: {t} counting qubits + "
          f"{(N - 1).bit_length()} work qubits\n")
    top = [y for y in np.argsort(probs)[::-1][:8] if probs[y] > 1e-9]
    for y in sorted(top):
        r, fac = q.factors_from_measurement(y, t, a, N)
        frac = [(p_, s_) for p_, s_ in q.convergents(y, 2 ** t) if s_ < N][-1]
        if y == 0:
            tail = "s = 0: no information, run again"
        elif r is None:
            tail = f"y/2^{t} ≈ {frac[0]}/{frac[1]}: {frac[1]} only divides r, run again"
        else:
            tail = f"y/2^{t} ≈ {frac[0]}/{frac[1]} → r = {r} → " + (
                ok(f"{N} = {min(fac)} × {max(fac)}") if fac else "r odd or a^(r/2) = -1, run again")
        print(f"  y = {y:5d}  P = {probs[y]:.3f}  {bar(probs[y] / probs.max(), 24)} {tail}")
    fac, a_, r, tries = q.shor(N, a=a, seed=args.seed)
    how = f"order r = {r}" if r else "gcd(a, N) > 1, found classically"
    print(f"\nSampled run: {N} = {min(fac)} × {max(fac)}  (a = {a_}, {how}, attempt {tries})")


def cmd_grover(args):
    marked = [args.marked % 2 ** args.n]
    _, theta, steps = q.grover(args.n, marked, trace=True)
    mean = lambda v: v.mean()  # noqa: E731
    for name, amp in steps:
        print(f"{name:<12} marked amp {amp[marked[0]]:+.3f}   mean {mean(amp):+.3f}   "
              f"P(marked) {amp[marked[0]] ** 2:.3f}  {bar(amp[marked[0]] ** 2, 30)}")
    print(dim(f"\nθ = {theta:.4f} rad; each oracle+diffusion rotates by 2θ toward the answer"))


# ------------------------------------------------------------------- verify ---
TERM_COLORS = ["36;1", "35;1", "33;1", "34;1"]  # cyan, magenta, yellow, blue: one per term


def cmd_verify(args):
    import nanoqalgos_math as m
    if args.markdown:
        sys.stdout.write(m.markdown(args.seed))
        return
    sel = [e for e in m.EQS if args.which in (None, e.id, str(e.part))]
    if not sel:
        sys.exit(f"no equation or part '{args.which}'. Try: nanoqalgos verify   (lists them all)")
    if args.which is None:  # summary of every equation
        failed = 0
        for part, name in m.PARTS.items():
            print(paint(f"Part {part} · {name}", "1"))
            for e in (e for e in sel if e.part == part):
                r = m.run(e, args.seed)
                failed += not r["ok"]
                mark = ok("✓") if r["ok"] else paint("✗", "31;1")
                gap = f"|Δ| = {r['gap']:.0e}" if r["relation"] == "=" else f"margin {r['gap']:+.3f}"
                text = e.text if len(e.text) <= 62 else e.text[:61] + "…"
                print(f"  {mark} eq {e.id:<4} {text:<62} {dim(gap)}")
        print(f"\n{len(sel) - failed}/{len(sel)} equations match the code (seed {args.seed}). "
              f"See one next to its code:  nanoqalgos verify 3.2")
        sys.exit(1 if failed else 0)
    for e in sel:
        show_equation(m, e, args.seed)


def show_equation(m, e, seed):
    src = m.source(e)
    color = {tag: TERM_COLORS[i % len(TERM_COLORS)] for i, tag in enumerate(e.terms)}
    print(paint(f"eq {e.id} · {e.title}", "1"))
    print(f"\n    {e.text}\n")
    for tag, meaning in e.terms.items():
        lines = ", ".join(f"{a}" if a == b else f"{a}-{b}"
                          for a, b in m.line_ranges(n for n, _, tags in src if tag in tags))
        label = tag[-1] if tag[-1].isalpha() else "•"
        word = "lines" if ("," in lines or "-" in lines) else "line"
        print(f"  {paint(label, color[tag])}  {meaning:<52} {dim(word + ' ' + lines)}")
    print(dim(f"\n  nanoqalgos.py, {q.__name__}.{e.func}()"))
    for n, text, tags in src:
        if tags:
            label = "".join(t[-1] if t[-1].isalpha() else "•" for t in tags)
            print(f"  {paint(label.rjust(2), color[tags[0]])} {n:4d}  {paint(text, color[tags[0]])}")
        else:
            print(f"     {dim(f'{n:4d}')}  {text}")
    r = m.run(e, seed)
    print(f"\n  formula  {r['formula']}\n  code     {r['code']}")
    if r["relation"] == "=":
        verdict = f"max |formula − code| = {r['gap']:.1e}  (tolerance {e.tol:g})"
    else:
        verdict = f"code {r['relation']} bound, margin {r['gap']:+.4f}"
    print(f"  {ok('✓') if r['ok'] else paint('✗', '31;1')} {verdict}")
    if e.note:
        print(dim(f"  {e.note}"))
    print()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nanoqalgos", description=__doc__)
    ap.add_argument("--version", action="version", version=q.__version__)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("demo", help="one line per algorithm, each checked (default)")
    sub.add_parser("bench", help="markdown table: classical vs quantum, with exact checks")
    s = sub.add_parser("shor", help="factor a small N and show the measurement peaks")
    s.add_argument("N", type=int)
    s.add_argument("--a", type=int, default=None)
    s.add_argument("--seed", type=int, default=0)
    g = sub.add_parser("grover", help="watch amplitudes step by step")
    g.add_argument("n", type=int, nargs="?", default=5)
    g.add_argument("--marked", type=int, default=13)
    v = sub.add_parser("verify", help="check every equation against the code that implements it")
    v.add_argument("which", nargs="?", default=None, help="an equation (3.2) or a part (3); omit for all")
    v.add_argument("--seed", type=int, default=0, help="random inputs for the checks")
    v.add_argument("--markdown", action="store_true", help="print MATH.md")
    args = ap.parse_args(argv)
    if args.cmd is None:
        args = ap.parse_args(["demo"])
    {"demo": cmd_demo, "bench": cmd_bench, "shor": cmd_shor, "grover": cmd_grover,
     "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    main()
