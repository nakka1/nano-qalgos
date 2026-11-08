"""
nanoqalgos — the quantum algorithms industry talks about, from scratch, in one NumPy file.

Read it top to bottom. Every algorithm ends in a number you can check against
a classical answer, and the tests do exactly that.

  Part 0  Simulator          A state is a tensor; a gate is a tensor contraction.
  Part 1  QFT                Hadamards + controlled phases  ==  np.fft, to 1e-15.
  Part 2  Phase estimation   Read an eigenvalue e^{2πiφ} out as the bits of φ.
  Part 3  Grover             Two reflections = one rotation by 2θ. √N queries.
  Part 4  Shor               Factoring = phase estimation of  y -> a·y mod N.
  Part 5  VQE                Chemistry: H2 ground energy via parameter-shift gradients.
  Part 6  QAOA               Optimization: MaxCut, checked against a closed form.
  Part 7  Amplitude est.     Finance: an option price with error ~1/M, not 1/√M.

Conventions: n qubits are a tensor of shape (2,)*n, qubit 0 is the most
significant bit, so the flat index of |q0 q1 ... q_{n-1}> is the binary number
q0 q1 ... q_{n-1}. Only dependency: numpy (plus the standard library).

Math <-> code: a comment like `# [eq 3.2a]` marks the line that implements term
(a) of equation 3.2. `nanoqalgos verify 3.2` prints that equation next to these
lines and checks numerically that they agree; MATH.md and docs/math.html show
the same thing with rendered formulas. The equations live in nanoqalgos_math.py.
"""
from __future__ import annotations

import math

import numpy as np

__version__ = "0.1.0"

# =============================================================================
# Part 0 · A state-vector simulator
# =============================================================================
# n qubits = 2**n complex amplitudes, stored as a tensor whose axis k is qubit k.
# A k-qubit gate is a 2^k x 2^k unitary; applying it is a contraction over k
# axes. Cost: O(2^n) per gate. We never build a 2^n x 2^n matrix.

I2 = np.eye(2, dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}
CNOT = np.eye(4, dtype=complex)[[0, 1, 3, 2]]  # |c,t> -> |c, t XOR c>
SWAP = np.eye(4, dtype=complex)[[0, 2, 1, 3]]


def phase(phi):
    """|0> -> |0>,  |1> -> e^{iφ}|1>."""
    return np.diag([1, np.exp(1j * phi)])


def ry(theta):
    """exp(-iθY/2): a real rotation, cos(θ/2)|0> + sin(θ/2)|1> from |0>."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)  # [eq 0.3a]
    return np.array([[c, -s], [s, c]], dtype=complex)  # [eq 0.3b]


def controlled(U):
    """|0><0| ⊗ I + |1><1| ⊗ U. The control is the first qubit."""
    d = U.shape[0]
    out = np.eye(2 * d, dtype=complex)  # [eq 0.2a]
    out[d:, d:] = U  # [eq 0.2b]
    return out


def zeros(n):
    psi = np.zeros((2,) * n, dtype=complex)
    psi[(0,) * n] = 1
    return psi


def apply(psi, U, *qubits):
    """Apply a k-qubit gate U to the listed axes of psi."""
    k = len(qubits)
    U = U.reshape((2,) * 2 * k)  # k output legs, then k input legs [eq 0.1a]
    psi = np.tensordot(U, psi, axes=(list(range(k, 2 * k)), list(qubits)))  # [eq 0.1b]
    return np.moveaxis(psi, list(range(k)), list(qubits))  # put legs back [eq 0.1c]


def probabilities(psi):
    """Born rule: P(outcome) = |amplitude|^2, in flat (binary) order."""
    return np.abs(psi.reshape(-1)) ** 2  # [eq 0.4]


# =============================================================================
# Part 1 · Quantum Fourier transform
# =============================================================================
#   QFT|j> = 1/√N Σ_k e^{2πi jk/N} |k>,   N = 2^n.                  (eq 1.1)
# Same matrix as the DFT (numpy's ifft, times √N). The trick: the output is a
# *product state*, qubit by qubit (eq 1.2), so n(n+1)/2 gates suffice where the matrix
# has N² entries. Qubit j gets H, then a phase 2π/2^m for every later qubit,
# and the order comes out reversed, hence the swaps at the end.


def qft(psi, qubits=None, inverse=False):
    if inverse:  # the QFT matrix is symmetric, so QFT† = conj(QFT) [eq 1.3]
        return np.conj(qft(np.conj(psi), qubits))
    qubits = list(range(psi.ndim)) if qubits is None else list(qubits)
    n = len(qubits)
    for j in range(n):
        psi = apply(psi, H, qubits[j])  # [eq 1.1a, 1.2a]
        for k in range(j + 1, n):  # controlled-R_m, R_m = phase(2π / 2^m) [eq 1.1b, 1.2b]
            psi = apply(psi, controlled(phase(2 * np.pi / 2 ** (k - j + 1))),  # [eq 1.1b, 1.2b]
                        qubits[k], qubits[j])  # [eq 1.1b, 1.2b]
    for j in range(n // 2):  # [eq 1.1c, 1.2c]
        psi = apply(psi, SWAP, qubits[j], qubits[n - 1 - j])  # [eq 1.1c, 1.2c]
    return psi


# =============================================================================
# Part 2 · Quantum phase estimation (QPE)
# =============================================================================
# Given U and an eigenvector U|u> = e^{2πiφ}|u>, learn φ to t bits.
#   1. t counting qubits in |+>: (1/√M) Σ_k |k>,  M = 2^t.
#   2. Counting qubit j controls U^(2^(t-1-j)). Those powers multiply to U^k,
#      so the joint state is  (1/√M) Σ_k |k> ⊗ U^k|u> = (1/√M) Σ_k e^{2πiφk}|k>|u>.  (eq 2.1)
#   3. That is exactly QFT|y> when φ = y/M. Inverse QFT -> measure y -> φ ≈ y/M.
# If φ is not a multiple of 1/M, P(y) = |1/M Σ_k e^{2πik(φ - y/M)}|² (Fejér kernel, eq 2.2):
# sharply peaked, ≥ 4/π² ≈ 0.405 on the nearest y (eq 2.3).
# Step 2 is written below as "row k holds U^k ψ", the same state the controlled
# gates produce; psi need not be an eigenvector (Shor relies on this).


def phase_estimation(U, psi, t):
    """Exact distribution of the t-bit outcome y. U: D x D matrix, psi: length D."""
    M, D = 2 ** t, len(psi)
    rows = np.empty((M, D), dtype=complex)
    rows[0] = psi
    for k in range(1, M):  # [eq 2.1a, 2.2a]
        rows[k] = U @ rows[k - 1]  # U^k ψ [eq 2.1a, 2.2a]
    state = (rows / np.sqrt(M)).reshape((2,) * t + (D,))  # counting qubits + work axis [eq 2.1b]
    state = qft(state, range(t), inverse=True)  # [eq 2.1c, 2.2b]
    return np.sum(np.abs(state.reshape(M, D)) ** 2, axis=1)  # trace out the work register [eq 2.1d, 2.3]


# =============================================================================
# Part 3 · Grover search
# =============================================================================
# Find one of m marked items among N. Start in |s> = H^n|0> (uniform).
#   oracle    O = I - 2Π_marked          flip the sign of the answers      (eq 3.1)
#   diffusion D = 2|s><s| - I = H^n (2|0><0| - I) H^n                  (eq 3.2)
#             i.e. every amplitude a_x -> 2·mean(a) - a_x  "inversion about the mean" (eq 3.3)
# Both are reflections; two reflections make a rotation by 2θ, sin θ = √(m/N),
# in the plane spanned by |marked> and |unmarked>. After k rounds:
#   P(success) = sin²((2k+1)θ),  best at k ≈ π/(4θ) ≈ (π/4)√(N/m).   (eq 3.4)
# A classical search needs ~N/(2m) guesses. That is the quadratic speedup.


def oracle(psi, marked):
    flat = psi.reshape(-1).copy()
    flat[list(marked)] *= -1  # [eq 3.1]
    return flat.reshape(psi.shape)


def diffusion(psi):
    n = psi.ndim
    for q in range(n):  # [eq 3.2a]
        psi = apply(psi, H, q)  # [eq 3.2a]
    psi = -psi  # [eq 3.2b]
    psi[(0,) * n] *= -1  # 2|0><0| - I [eq 3.2c, 3.3]
    for q in range(n):  # [eq 3.2a]
        psi = apply(psi, H, q)  # [eq 3.2a]
    return psi


def grover(n, marked, iterations=None, trace=False):
    """Returns (final state, θ, optional list of (step name, amplitudes))."""
    theta = np.arcsin(np.sqrt(len(marked) / 2 ** n))  # [eq 3.4a]
    if iterations is None:
        iterations = int(np.floor(np.pi / (4 * theta)))  # [eq 3.4b]
    psi = zeros(n)
    for q in range(n):
        psi = apply(psi, H, q)
    steps = [("start", psi.reshape(-1).real.copy())]
    for i in range(iterations):
        psi = oracle(psi, marked)  # [eq 3.4c]
        steps.append((f"oracle {i + 1}", psi.reshape(-1).real.copy()))
        psi = diffusion(psi)  # [eq 3.4c]
        steps.append((f"diffusion {i + 1}", psi.reshape(-1).real.copy()))
    return (psi, theta, steps) if trace else (psi, theta)


# =============================================================================
# Part 4 · Shor's factoring algorithm
# =============================================================================
# To factor N, pick a with gcd(a, N) = 1 and find its order r: the smallest
# r > 0 with a^r ≡ 1 (mod N). If r is even and a^(r/2) ≢ -1, then
#   (a^(r/2) - 1)(a^(r/2) + 1) ≡ 0 (mod N)  ->  gcd(a^(r/2) ± 1, N) are factors.  (eq 4.3)
# The quantum part finds r: U|y> = |a·y mod N> is a permutation with
# eigenvalues e^{2πi s/r}, and |1> is an equal mix of those eigenvectors (eq 4.1).
# Phase estimation on |1> returns y ≈ (s/r)·2^t for a random s; continued
# fractions turn y/2^t back into s/r (eq 4.2). Everything else is classical.


def modmul_matrix(a, N, n):
    """Permutation matrix of y -> a·y mod N on n qubits (y >= N left alone)."""
    image = np.arange(2 ** n)
    image[:N] = (a * image[:N]) % N  # [eq 4.1a]
    U = np.zeros((2 ** n, 2 ** n))
    U[image, np.arange(2 ** n)] = 1  # [eq 4.1b]
    return U


def convergents(num, den):
    """Best rational approximations p/q of num/den, from its continued fraction
    [a0; a1, a2, ...] and the recurrence h_k = a_k h_{k-1} + h_{k-2} (same for q)."""
    num, den = int(num), int(den)
    h0, h1, k0, k1 = 0, 1, 1, 0
    while den:
        a, (num, den) = num // den, (den, num % den)  # [eq 4.2a]
        h0, h1, k0, k1 = h1, a * h1 + h0, k1, a * k1 + k0  # [eq 4.2b]
        yield h1, k1


def order_finding(a, N):
    """Exact distribution of the measured y, and the number of counting qubits t."""
    n = (N - 1).bit_length()
    t = 2 * n  # enough precision to separate fractions with denominator < N
    work = np.zeros(2 ** n)
    work[1] = 1  # start the work register in |1>
    return phase_estimation(modmul_matrix(a, N, n), work, t), t


def factors_from_measurement(y, t, a, N):
    """Classical post-processing of one measurement. Returns (r, factors or None)."""
    for _, r in convergents(y, 2 ** t):
        if 0 < r < N and pow(a, r, N) == 1:  # [eq 4.3a]
            half = pow(a, r // 2, N)  # [eq 4.3b]
            if r % 2 == 0 and half != N - 1:  # [eq 4.3b]
                f = math.gcd(half - 1, N)  # [eq 4.3c]
                if 1 < f < N:
                    return r, (f, N // f)
            return r, None
    return None, None


def shor(N, a=None, seed=None):
    """Factor N. Returns (factors, a, r, attempts) using sampled measurements."""
    rng = np.random.default_rng(seed)
    for attempt in range(1, 50):
        a_ = a or int(rng.integers(2, N - 1))
        if math.gcd(a_, N) > 1:  # lucky: a shares a factor with N already
            return (math.gcd(a_, N), N // math.gcd(a_, N)), a_, None, attempt
        probs, t = order_finding(a_, N)
        y = int(rng.choice(len(probs), p=probs / probs.sum()))
        r, fac = factors_from_measurement(y, t, a_, N)
        if fac:
            return fac, a_, r, attempt
    raise RuntimeError("no factor found")


# =============================================================================
# Part 5 · VQE: variational quantum eigensolver (chemistry)
# =============================================================================
# Ground energy of a molecule = smallest eigenvalue of its Hamiltonian, written
# as a weighted sum of Pauli strings. VQE prepares |ψ(θ)> with a short circuit,
# measures E(θ) = <ψ|H|ψ> = Σ c_P <ψ|P|ψ> (eq 5.1), and lets a classical optimizer
# lower it. By the variational principle E(θ) >= E_0 always (eq 5.3).
# Gradients are exact, from two more circuit runs (parameter-shift rule):
#   for a gate exp(-iθG/2) with G² = I:  ∂E/∂θ = [E(θ + π/2) - E(θ - π/2)] / 2.  (eq 5.2)
#
# H2 at 0.735 Å, STO-3G basis, parity mapping reduced to 2 qubits (electronic
# energy in Hartree; add 0.7199689944 Ha nuclear repulsion for the total).
H2 = [(-1.052373245772859, "II"), (0.39793742484318045, "IZ"),
      (-0.39793742484318045, "ZI"), (-0.01128010425623538, "ZZ"),
      (0.18093119978423156, "XX")]


def expectation(psi, hamiltonian):
    """<ψ|H|ψ> by applying each Pauli string, one single-qubit matrix at a time."""
    total = 0.0
    for coef, word in hamiltonian:
        phi = psi
        for q, p in enumerate(word):
            if p != "I":
                phi = apply(phi, PAULI[p], q)  # [eq 5.1a]
        total += coef * np.vdot(psi, phi).real  # [eq 5.1b]
    return total


def hamiltonian_matrix(hamiltonian):
    """The full 2^n x 2^n matrix, only to check VQE against exact diagonalization."""
    out = 0
    for coef, word in hamiltonian:
        m = np.array([[1.0]])
        for p in word:
            m = np.kron(m, PAULI[p])
        out = out + coef * m
    return out


def ansatz(theta, n, layers):
    """Hardware-efficient circuit: RY on every qubit, then a CNOT chain, repeated."""
    theta = np.reshape(theta, (layers + 1, n))
    psi = zeros(n)
    for layer in range(layers + 1):
        for q in range(n):
            psi = apply(psi, ry(theta[layer, q]), q)
        if layer < layers:
            for q in range(n - 1):
                psi = apply(psi, CNOT, q, q + 1)
    return psi


def vqe(hamiltonian, layers=1, steps=150, lr=0.4, seed=0):
    n = len(hamiltonian[0][1])
    energy = lambda th: expectation(ansatz(th, n, layers), hamiltonian)  # noqa: E731 [eq 5.3]
    theta = np.random.default_rng(seed).uniform(-0.1, 0.1, n * (layers + 1))
    history = []
    for _ in range(steps):
        shift = np.eye(len(theta)) * np.pi / 2  # [eq 5.2a]
        grad = np.array([(energy(theta + s) - energy(theta - s)) / 2 for s in shift])  # [eq 5.2b]
        theta = theta - lr * grad
        history.append(energy(theta))
    return theta, np.array(history)


# =============================================================================
# Part 6 · QAOA: quantum approximate optimization (MaxCut)
# =============================================================================
# MaxCut: color graph nodes 0/1 to cut as many edges as possible. The cost
# C(z) = Σ_edges [z_i ≠ z_j] is diagonal in the computational basis (eq 6.1).
#   |γ,β> = e^{-iβB} e^{-iγC} |+>^n,   B = Σ_j X_j   (one layer, p = 1)  (eq 6.2)
# e^{-iγC} just multiplies amplitude z by e^{-iγC(z)}; e^{-iβX} is cos β - i sin β X.
# For p = 1 the expected cut has a closed form (Wang, Hadfield, Jiang & Rieffel,
# 2018) that depends only on degrees and triangles (eq 6.3). The simulator must match it.


def cut_values(n, edges):
    z = (np.arange(2 ** n)[:, None] >> (n - 1 - np.arange(n))) & 1  # bits of z [eq 6.1a]
    return sum((z[:, i] != z[:, j]).astype(int) for i, j in edges)  # [eq 6.1b]


def qaoa_state(n, edges, gammas, betas):
    c = cut_values(n, edges)
    psi = np.full(2 ** n, 2 ** (-n / 2), dtype=complex).reshape((2,) * n)  # [eq 6.2a]
    for g, b in zip(gammas, betas):
        psi = (np.exp(-1j * g * c) * psi.reshape(-1)).reshape(psi.shape)  # [eq 6.2b]
        mixer = np.cos(b) * I2 - 1j * np.sin(b) * X  # [eq 6.2c]
        for q in range(n):  # [eq 6.2c]
            psi = apply(psi, mixer, q)  # [eq 6.2c]
    return psi


def qaoa_expectation(n, edges, gammas, betas):
    return float(probabilities(qaoa_state(n, edges, gammas, betas)) @ cut_values(n, edges))


def qaoa_p1_formula(n, edges, gamma, beta):
    """Closed-form <C> at p = 1, edge by edge (d = other neighbours, f = triangles)."""
    nbr = [set() for _ in range(n)]
    for i, j in edges:
        nbr[i].add(j)
        nbr[j].add(i)
    total = 0.0
    for u, v in edges:
        d, e, f = len(nbr[u]) - 1, len(nbr[v]) - 1, len(nbr[u] & nbr[v])  # [eq 6.3a]
        c = np.cos(gamma)
        total += (0.5 + 0.25 * np.sin(4 * beta) * np.sin(gamma) * (c ** d + c ** e)  # [eq 6.3b]
                  - 0.25 * np.sin(2 * beta) ** 2 * c ** (d + e - 2 * f)  # [eq 6.3c]
                  * (1 - np.cos(2 * gamma) ** f))  # [eq 6.3c]
    return total


def qaoa_maxcut(n, edges, grid=48, rounds=4):
    """p = 1: scan the (periodic) landscape with the closed form, zoom in around
    the best point a few times, then confirm with the simulator. Returns (γ, β, <C>)."""
    g0, b0, wg, wb = np.pi / 2, np.pi / 4, np.pi / 2, np.pi / 4  # centre and half-width
    for _ in range(rounds):
        gs, bs = np.linspace(g0 - wg, g0 + wg, grid), np.linspace(b0 - wb, b0 + wb, grid)
        vals = np.array([[qaoa_p1_formula(n, edges, g, b) for b in bs] for g in gs])
        i, j = np.unravel_index(np.argmax(vals), vals.shape)
        g0, b0, wg, wb = gs[i], bs[j], 4 * wg / grid, 4 * wb / grid
    return g0, b0, qaoa_expectation(n, edges, [g0], [b0])


# =============================================================================
# Part 7 · Amplitude estimation (finance: option pricing)
# =============================================================================
# Many pricing problems are an expectation a = E[f(S)] with 0 <= f <= 1.
# Load it into one amplitude:
#   A|0> = Σ_i √p_i |i> ⊗ (√(1-f_i)|0> + √f_i|1>),   P(last qubit = 1) = a = sin²θ_a.  (eq 7.2)
# Classical Monte Carlo with M samples has error ~ √(a(1-a)/M).
# Grover's operator Q = -A S_0 A† S_χ (S_0 flips |0>, S_χ flips "last qubit 1")
# rotates by 2θ_a, i.e. has eigenvalues e^{±2iθ_a} (eq 7.4). Phase estimation reads θ_a
# with M - 1 uses of Q (≈ 2M calls to A), and Brassard et al. (2002) prove
#   |ã - a| <= 2π√(a(1-a))/M + π²/M²   with probability >= 8/π² ≈ 0.81.  (eq 7.5)
# Error ~1/M instead of ~1/√M: quadratically fewer calls for the same accuracy.
# Read the fine print, though: the guarantee is at fixed confidence. The Fejér
# tails of Part 2 make the *mean squared* error shrink only like 1/M, which is
# why production code uses iterative / maximum-likelihood variants.


def european_call(n=4, S0=100.0, K=105.0, sigma=0.2, r=0.05, T=1.0):
    """Discretize a lognormal price on 2^n points. Returns (p, f, scale, black_scholes)
    so that the discretized price is scale · Σ p_i f_i and f_i ∈ [0, 1]."""
    mu, s = np.log(S0) + (r - sigma ** 2 / 2) * T, sigma * np.sqrt(T)
    S = np.exp(np.linspace(mu - 3 * s, mu + 3 * s, 2 ** n))  # grid ±3σ in log space
    p = np.exp(-((np.log(S) - mu) ** 2) / (2 * s ** 2))
    p /= p.sum()
    payoff = np.maximum(S - K, 0)
    scale = np.exp(-r * T) * payoff.max()
    N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))  # noqa: E731
    d1 = (math.log(S0 / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))  # [eq 7.1a]
    bs = S0 * N(d1) - K * math.exp(-r * T) * N(d1 - sigma * math.sqrt(T))  # [eq 7.1b]
    return p, payoff / payoff.max(), scale, bs


def amplitude_estimation(p, f, t):
    """Canonical QAE. Returns (estimates ã_y, their exact probabilities, true a)."""
    psi = (np.sqrt(p)[:, None] * np.stack([np.sqrt(1 - f), np.sqrt(f)], 1)).reshape(-1)  # [eq 7.2a]
    D = len(psi)
    w = np.eye(D)[0] - psi  # Householder reflection: a real unitary with A|0> = ψ [eq 7.3a]
    A = np.eye(D) - 2 * np.outer(w, w) / (w @ w)  # [eq 7.3b]
    S0 = np.eye(D)  # [eq 7.4a]
    S0[0, 0] = -1  # [eq 7.4a]
    Schi = np.diag(np.where(np.arange(D) % 2 == 1, -1.0, 1.0))  # good = last qubit 1 [eq 7.4b]
    Q = -A @ S0 @ A.T @ Schi  # [eq 7.4c]
    probs = phase_estimation(Q, psi, t)  # [eq 7.4d]
    estimates = np.sin(np.pi * np.arange(2 ** t) / 2 ** t) ** 2  # [eq 7.5a]
    return estimates, probs, float(p @ f)  # [eq 7.2b]


def error_at_confidence(values, probs, target, confidence=0.8):
    """Smallest ε with P(|value - target| <= ε) >= confidence, from an exact distribution."""
    dist = np.abs(np.asarray(values) - target)
    order = np.argsort(dist)
    return float(dist[order][np.searchsorted(np.cumsum(probs[order]), confidence)])


def monte_carlo_error(a, samples, confidence=0.8):
    """The same quantity for classical Monte Carlo: k ~ Binomial(samples, a), â = k/samples.
    Exact binomial pmf (via log-gamma), no normal approximation."""
    k = np.arange(samples + 1)
    log_choose = np.array([math.lgamma(samples + 1) - math.lgamma(i + 1)  # [eq 7.6a]
                           - math.lgamma(samples - i + 1) for i in k])  # [eq 7.6a]
    pmf = np.exp(log_choose + k * np.log(a) + (samples - k) * np.log1p(-a))  # [eq 7.6b]
    return error_at_confidence(k / samples, pmf, a, confidence)
