"""
The math of nanoqalgos.py, equation by equation, checked against the code that implements it.

Every equation below has:
  - the formula, in LaTeX (browser, MATH.md) and in plain Unicode (terminal);
  - its terms (a, b, c, ...), each tied to the lines of nanoqalgos.py tagged `# [eq X.Ya]`;
  - a check(rng) that returns (formula, code):
        formula: the equation evaluated *as written*, with plain NumPy, never calling
                 the function under test;
        code:    what nanoqalgos.py actually computes.
    They must agree (relation "="), or the code value must respect a bound ("≥", "≤").

`nanoqalgos verify` runs them all; `nanoqalgos verify 3.2` shows one next to its code.
"""
from __future__ import annotations

import inspect
import math
import re
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

import nanoqalgos as q

TAG = re.compile(r"\[eq ([^\]]+)\]")


def term(tag, tex):
    """Wrap a piece of LaTeX so the browser can link it to the code lines tagged `tag`."""
    return r"\htmlClass{t" + tag.replace(".", "_") + "}{" + tex + "}"


@dataclass
class Eq:
    id: str
    title: str
    tex: str
    text: str
    func: str
    terms: dict
    check: object
    relation: str = "="
    tol: float = 1e-9
    note: str = ""
    part: int = field(init=False)

    def __post_init__(self):
        self.part = int(self.id.split(".")[0])
        if r"\\" in self.tex:  # multi-line formula: one line per piece, centred
            self.tex = r"\begin{gathered}" + self.tex + r"\end{gathered}"


EQS = []
PARTS = {0: "Simulator", 1: "Quantum Fourier transform", 2: "Phase estimation", 3: "Grover search",
         4: "Shor's algorithm", 5: "VQE", 6: "QAOA", 7: "Amplitude estimation"}


def equation(id, title, tex, text, func, terms=None, relation="=", tol=1e-9, note=""):
    def register(check):
        EQS.append(Eq(id, title, tex, text, func, terms or {}, check, relation, tol, note))
        return check
    return register


# ---------------------------------------------------------------- helpers ---
def kron_all(*ms):
    out = np.array([[1.0 + 0j]])
    for m in ms:
        out = np.kron(out, m)
    return out


def expm_hermitian(Hm, t):
    """exp(-i t H) for Hermitian H, by diagonalization: the definition, no shortcuts."""
    w, V = np.linalg.eigh(Hm)
    return V @ np.diag(np.exp(-1j * t * w)) @ V.conj().T


def random_state(rng, n):
    v = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    return v / np.linalg.norm(v)


def random_unitary(rng, d):
    Qm, Rm = np.linalg.qr(rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)))
    return Qm * (np.diag(Rm) / np.abs(np.diag(Rm)))


def dft(N):
    j, k = np.meshgrid(np.arange(N), np.arange(N))
    return np.exp(2j * np.pi * j * k / N) / np.sqrt(N)  # F[k, j]


def fejer(x, M):
    return np.abs(np.exp(2j * np.pi * np.outer(x, np.arange(M))).sum(1) / M) ** 2


# =============================================================== Part 0 ====
@equation("0.1", "A gate is a contraction on its qubit's axis",
          r"(U_{q}\psi)_{i_1\dots " + term("0.1c", "i_q") + r"\dots i_n} = " + term("0.1b", r"\sum_{j}")
          + r"\;" + term("0.1a", r"U_{i_q j}") + r"\;\psi_{i_1\dots j \dots i_n}",
          "(U_q ψ)[… i_q …] = Σ_j U[i_q, j] · ψ[… j …]", "apply",
          {"0.1a": "U reshaped into output and input legs", "0.1b": "sum over the input leg (np.tensordot)",
           "0.1c": "output leg moved back to the qubit's axis"},
          note="Checked on a 2-qubit gate acting on qubits (2, 0) of 3, out of order on purpose.")
def _(rng):
    psi, U = random_state(rng, 3).reshape(2, 2, 2), random_unitary(rng, 4)
    # out[b, e, a] = Σ_{c,d} U[a, b, c, d] psi[d, e, c]: qubit 2 ↔ (a, c), qubit 0 ↔ (b, d)
    formula = np.einsum("abcd,dec->bea", U.reshape(2, 2, 2, 2), psi)
    return formula, q.apply(psi, U, 2, 0)


@equation("0.2", "Controlled gate",
          r"C(U) = " + term("0.2a", r"|0\rangle\langle 0|\otimes I") + " + " + term("0.2b", r"|1\rangle\langle 1|\otimes U"),
          "C(U) = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ U", "controlled",
          {"0.2a": "start from the identity (the |0⟩ block)", "0.2b": "overwrite the |1⟩ block with U"})
def _(rng):
    U = random_unitary(rng, 4)
    P0, P1 = np.diag([1, 0]), np.diag([0, 1])
    return np.kron(P0, np.eye(4)) + np.kron(P1, U), q.controlled(U)


@equation("0.3", "Y rotation",
          r"R_Y(\theta) = e^{-i\theta Y/2} = " + term("0.3a", r"\cos\tfrac{\theta}{2}\,I - i\sin\tfrac{\theta}{2}\,Y")
          + r" \\ = " + term("0.3b", r"\begin{pmatrix}\cos\frac{\theta}{2} & -\sin\frac{\theta}{2}\\ \sin\frac{\theta}{2} & \cos\frac{\theta}{2}\end{pmatrix}"),
          "R_Y(θ) = exp(−iθY/2) = cos(θ/2) I − i sin(θ/2) Y", "ry",
          {"0.3a": "cos and sin of the half angle", "0.3b": "the resulting real 2×2 matrix"},
          note="The formula side computes the matrix exponential by diagonalizing Y.")
def _(rng):
    theta = rng.uniform(-np.pi, np.pi)
    return expm_hermitian(q.Y, theta / 2), q.ry(theta)


@equation("0.4", "Born rule",
          r"P(x) = " + term("0.4", r"\big|\langle x|\psi\rangle\big|^{2}"), "P(x) = |⟨x|ψ⟩|²", "probabilities",
          {"0.4": "squared magnitude of every amplitude"})
def _(rng):
    psi = random_state(rng, 4)
    basis = np.eye(16)
    return np.array([abs(np.vdot(basis[x], psi)) ** 2 for x in range(16)]), q.probabilities(psi.reshape((2,) * 4))


# =============================================================== Part 1 ====
@equation("1.1", "QFT circuit = DFT matrix",
          term("1.1c", r"\mathrm{REV}") + r"\;\prod_{j}\Big[\prod_{k>j}" + term("1.1b", r"CR^{\,k\to j}_{k-j+1}") + r"\Big]\,"
          + term("1.1a", r"H_j") + r"\;=\;F \\ F_{kj} = \tfrac{1}{\sqrt N}\,e^{2\pi i\,jk/N},\quad R_m = \mathrm{diag}(1, e^{2\pi i/2^m})",
          "REV · Π_j [Π_{k>j} CR_{k−j+1}] H_j = F,  F[k, j] = e^{2πi jk/N}/√N", "qft",
          {"1.1a": "Hadamard on qubit j", "1.1b": "controlled phase R_m from every later qubit",
           "1.1c": "swaps reverse the qubit order"},
          note="The code side builds the full matrix by running the circuit on every basis state (n = 5).")
def _(rng):
    n, N = 5, 32
    cols = [q.qft(np.eye(N)[j].astype(complex).reshape((2,) * n)).reshape(-1) for j in range(N)]
    return dft(N), np.array(cols).T


@equation("1.2", "QFT output is a product state",
          r"\mathrm{QFT}\,|j\rangle = " + term("1.2c", r"\bigotimes_{\ell=1}^{n}") + r"\tfrac{1}{\sqrt2}\Big(|0\rangle + "
          + term("1.2a", r"e^{i\pi j_{\ell-1}}") + term("1.2b", r"\prod_{m<\ell-1} e^{2\pi i\, j_m 2^{m-\ell}}")
          + r"\,|1\rangle\Big) \\ j = \textstyle\sum_m j_m 2^m",
          "QFT|j⟩ = ⊗_ℓ (|0⟩ + e^{2πi j/2^ℓ}|1⟩)/√2", "qft",
          {"1.2a": "Hadamard: the sign from bit j_{ℓ−1}", "1.2b": "controlled phases: the finer bits",
           "1.2c": "factors come out reversed; swaps put them in order"})
def _(rng):
    n = 6
    j = int(rng.integers(0, 2 ** n))
    factors = [np.array([1, np.exp(2j * np.pi * j / 2 ** ell)]) / np.sqrt(2) for ell in range(1, n + 1)]
    formula = kron_all(*[f[None, :] for f in factors]).reshape(-1)
    return formula, q.qft(np.eye(2 ** n)[j].astype(complex).reshape((2,) * n)).reshape(-1)


@equation("1.3", "Inverse QFT",
          term("1.3", r"\mathrm{QFT}^{\dagger} = \overline{\mathrm{QFT}}") + r"\quad\text{because}\quad F^{T} = F",
          "QFT† = conj(QFT), because F is symmetric", "qft", {"1.3": "conjugate, transform, conjugate back"})
def _(rng):
    psi = random_state(rng, 5)
    return dft(32).conj().T @ psi, q.qft(psi.reshape((2,) * 5), inverse=True).reshape(-1)


# =============================================================== Part 2 ====
@equation("2.1", "Phase estimation, any input state",
          r"P(y) = " + term("2.1d", r"\Big\|") + term("2.1c", r"\tfrac{1}{\sqrt M}\sum_{k=0}^{M-1} e^{-2\pi i\,ky/M}")
          + r"\," + term("2.1b", r"\tfrac{1}{\sqrt M}") + r"\," + term("2.1a", r"U^{k}|\psi\rangle") + term("2.1d", r"\Big\|^{2}"),
          "P(y) = ‖ (1/√M) Σ_k e^{−2πi ky/M} · (1/√M) U^k|ψ⟩ ‖²", "phase_estimation",
          {"2.1a": "row k of the register holds U^k ψ", "2.1b": "uniform superposition amplitude 1/√M",
           "2.1c": "inverse QFT on the counting register", "2.1d": "probability of y: sum over the work register"},
          note="Random 4×4 unitary and random ψ (not an eigenvector), t = 4.")
def _(rng):
    U, psi, t = random_unitary(rng, 4), random_state(rng, 2), 4
    M = 2 ** t
    powers = [np.linalg.matrix_power(U, k) @ psi for k in range(M)]
    formula = [np.linalg.norm(sum(np.exp(-2j * np.pi * k * y / M) * powers[k] for k in range(M)) / M) ** 2
               for y in range(M)]
    return np.array(formula), q.phase_estimation(U, psi, t)


@equation("2.2", "Eigenvector input: the Fejér kernel",
          r"U|u\rangle = e^{2\pi i\varphi}|u\rangle \;\Rightarrow\; P(y) = \Big|\tfrac1M\sum_{k}"
          + term("2.2a", r"e^{2\pi i\varphi k}") + term("2.2b", r"e^{-2\pi i\,ky/M}") + r"\Big|^{2}",
          "P(y) = | (1/M) Σ_k e^{2πiφk} e^{−2πi ky/M} |²", "phase_estimation",
          {"2.2a": "U^k on an eigenvector is just a phase", "2.2b": "inverse QFT"})
def _(rng):
    phi, t = rng.uniform(), 6
    M = 2 ** t
    U = np.diag([np.exp(2j * np.pi * phi), np.exp(2j * np.pi * rng.uniform())])
    return fejer(phi - np.arange(M) / M, M), q.phase_estimation(U, np.array([1, 0]), t)


@equation("2.3", "Nearest outcome has probability ≥ 4/π²",
          term("2.3", r"\max_y P(y)") + r" \;\ge\; \frac{4}{\pi^{2}} \approx 0.405",
          "max_y P(y) ≥ 4/π² ≈ 0.405", "phase_estimation", {"2.3": "the output distribution"},
          relation="≥", note="Code value: the worst case over 50 random phases.")
def _(rng):
    worst = min(q.phase_estimation(np.diag([np.exp(2j * np.pi * phi), 1]), np.array([1, 0]), 5).max()
                for phi in rng.uniform(size=50))
    return 4 / np.pi ** 2, worst


# =============================================================== Part 3 ====
@equation("3.1", "Oracle: flip the answers",
          r"O = I - 2\!\!\sum_{x\in\text{marked}}\!\!" + term("3.1", r"|x\rangle\langle x|"),
          "O = I − 2 Σ_{x marked} |x⟩⟨x|", "oracle", {"3.1": "negate the marked amplitudes"})
def _(rng):
    psi, marked = random_state(rng, 4), rng.choice(16, 3, replace=False)
    O = np.eye(16)
    O[marked, marked] = -1
    return O @ psi, q.oracle(psi.reshape((2,) * 4), marked).reshape(-1)


@equation("3.2", "Diffusion is a reflection about |s⟩",
          term("3.2a", r"H^{\otimes n}") + r"\big(" + term("3.2c", r"2|0\rangle\langle 0|") + term("3.2b", r"\,-\,I")
          + r"\big)" + term("3.2a", r"H^{\otimes n}") + r" = 2|s\rangle\langle s| - I,\qquad |s\rangle = H^{\otimes n}|0\rangle",
          "H⊗ⁿ (2|0⟩⟨0| − I) H⊗ⁿ = 2|s⟩⟨s| − I", "diffusion",
          {"3.2a": "Hadamard on every qubit", "3.2b": "− I: negate everything", "3.2c": "2|0⟩⟨0|: restore |0⟩'s sign"})
def _(rng):
    psi = random_state(rng, 5)
    s = np.full(32, 1 / np.sqrt(32))
    return (2 * np.outer(s, s) - np.eye(32)) @ psi, q.diffusion(psi.reshape((2,) * 5)).reshape(-1)


@equation("3.3", "Inversion about the mean",
          r"a_x \;\mapsto\; 2\," + term("3.3", r"\bar a") + r" - a_x,\qquad \bar a = \tfrac1N\textstyle\sum_x a_x",
          "a_x → 2·mean(a) − a_x", "diffusion", {"3.3": "this sign flip is what produces the mean"})
def _(rng):
    psi = random_state(rng, 5)
    return 2 * psi.mean() - psi, q.diffusion(psi.reshape((2,) * 5)).reshape(-1)


@equation("3.4", "Grover success probability",
          r"P_k = \sin^{2}\!\big(" + term("3.4c", r"(2k+1)") + term("3.4a", r"\theta") + r"\big) \\ "
          + term("3.4a", r"\theta = \arcsin\sqrt{m/N}") + r",\quad " + term("3.4b", r"k^{\star} = \lfloor \pi/(4\theta)\rfloor"),
          "P_k = sin²((2k+1)θ),  θ = arcsin √(m/N),  k* = ⌊π/(4θ)⌋", "grover",
          {"3.4a": "the rotation angle", "3.4b": "the best number of rounds", "3.4c": "each round: oracle + diffusion"},
          note="N = 128, 2 marked items, k = 0 … 10.")
def _(rng):
    n, marked = 7, list(rng.choice(128, 2, replace=False))
    theta = np.arcsin(np.sqrt(2 / 128))
    code = [q.probabilities(q.grover(n, marked, iterations=k)[0])[marked].sum() for k in range(11)]
    return np.sin((2 * np.arange(11) + 1) * theta) ** 2, np.array(code)


# =============================================================== Part 4 ====
@equation("4.1", "Eigenvectors of modular multiplication",
          term("4.1b", "U") + r"|y\rangle = " + term("4.1a", r"|a y \bmod N\rangle") + r" \\ "
          r"U|u_s\rangle = e^{2\pi i s/r}|u_s\rangle \\ |u_s\rangle = \tfrac{1}{\sqrt r}\sum_{k=0}^{r-1} e^{-2\pi i sk/r}|a^{k} \bmod N\rangle",
          "U|y⟩ = |a·y mod N⟩;  U|u_s⟩ = e^{2πi s/r}|u_s⟩", "modmul_matrix",
          {"4.1a": "the permutation y → a·y mod N", "4.1b": "written as a matrix, one 1 per column"},
          note="N = 21, a = 2 (order r = 6): every |u_s⟩ is an eigenvector of the code's matrix.")
def _(rng):
    N, a, n = 21, 2, 5
    r = next(k for k in range(1, N) if pow(a, k, N) == 1)
    U = q.modmul_matrix(a, N, n)
    formula, code = [], []
    for s in range(r):
        u = np.zeros(2 ** n, complex)
        for k in range(r):
            u[pow(a, k, N)] += np.exp(-2j * np.pi * s * k / r) / np.sqrt(r)
        formula.append(np.exp(2j * np.pi * s / r) * u)
        code.append(U @ u)
    return np.array(formula), np.array(code)


@equation("4.2", "Continued-fraction convergents",
          r"\frac{y}{2^{t}} = [a_0; a_1, a_2, \dots],\quad " + term("4.2a", r"a_k = \lfloor x_k \rfloor") + r" \\ "
          + term("4.2b", r"h_k = a_k h_{k-1} + h_{k-2},\;\; q_k = a_k q_{k-1} + q_{k-2}"),
          "a_k = ⌊x_k⌋;  h_k = a_k h_{k−1} + h_{k−2},  q_k = a_k q_{k−1} + q_{k−2}", "convergents",
          {"4.2a": "next partial quotient (Euclid's algorithm)", "4.2b": "the recurrence for numerator and denominator"},
          note="Formula side: evaluate each truncated fraction [a0; …, ak] directly with exact rationals.")
def _(rng):
    y, M = int(rng.integers(1, 4096)), 4096
    x, quotients = Fraction(y, M), []
    while True:  # partial quotients by definition
        quotients.append(math.floor(x))
        if x == math.floor(x):
            break
        x = 1 / (x - math.floor(x))
    formula = []
    for k in range(len(quotients)):
        v = Fraction(quotients[k])
        for aq in reversed(quotients[:k]):
            v = aq + 1 / v
        formula.append((v.numerator, v.denominator))
    return np.array(formula, float), np.array(list(q.convergents(y, M)), float)


@equation("4.3", "From the order to the factors",
          term("4.3a", r"a^{r} \equiv 1 \pmod N") + r",\;\; " + term("4.3b", r"r\text{ even},\; a^{r/2} \not\equiv -1")
          + r" \\ \Rightarrow\; (a^{r/2}-1)(a^{r/2}+1) \equiv 0 \\ \Rightarrow\; " + term("4.3c", r"\gcd(a^{r/2}-1,\,N)") + r"\text{ divides } N",
          "a^r ≡ 1, r even, a^{r/2} ≢ −1  ⇒  gcd(a^{r/2} − 1, N) is a factor", "factors_from_measurement",
          {"4.3a": "r is a period of a mod N", "4.3b": "the two conditions that make it useful",
           "4.3c": "the factor, by Euclid's gcd"},
          note="Eight values of N; the code is fed the ideal measurement y = round(2^t / r).")
def _(rng):
    formula, code = [], []
    for N in (15, 21, 33, 35, 39, 51, 55, 57):
        for a in range(2, N):
            r = next((k for k in range(1, N) if pow(a, k, N) == 1), None) if math.gcd(a, N) == 1 else None
            if r and r % 2 == 0 and pow(a, r // 2, N) != N - 1:
                break
        t = 2 * (N - 1).bit_length()
        formula.append(math.gcd(pow(a, r // 2) - 1, N))
        code.append(q.factors_from_measurement(round(2 ** t / r), t, a, N)[1][0])
    return np.array(formula), np.array(code)


# =============================================================== Part 5 ====
@equation("5.1", "Energy of a Pauli-sum Hamiltonian",
          r"E = \langle\psi|H|\psi\rangle = \sum_{P} c_P\," + term("5.1b", r"\langle\psi|") + term("5.1a", r"P|\psi\rangle"),
          "E = ⟨ψ|H|ψ⟩ = Σ_P c_P ⟨ψ|P|ψ⟩", "expectation",
          {"5.1a": "apply the Pauli string one qubit at a time", "5.1b": "overlap with ⟨ψ|, weighted by c_P"},
          note="Formula side: build H as a 4×4 matrix with Kronecker products, then ψ†Hψ.")
def _(rng):
    psi = random_state(rng, 2)
    Hm = sum(c * kron_all(*[q.PAULI[p] for p in w]) for c, w in q.H2)
    return np.vdot(psi, Hm @ psi).real, q.expectation(psi.reshape(2, 2), q.H2)


@equation("5.2", "Parameter-shift gradient",
          r"\frac{\partial E}{\partial\theta_i} = " + term("5.2b", r"\tfrac12\Big[E\big(\theta + " + term("5.2a", r"\tfrac{\pi}{2}e_i")
                                                                + r"\big) - E\big(\theta - \tfrac{\pi}{2}e_i\big)\Big]"),
          "∂E/∂θ_i = ½ [E(θ + π/2·e_i) − E(θ − π/2·e_i)]", "vqe",
          {"5.2a": "shift one angle by ±π/2", "5.2b": "half the difference: an exact derivative"},
          tol=1e-8, note="One step of vqe() must equal θ − lr·∇E, with ∇E from a plain finite difference.")
def _(rng):
    lr, h = 0.3, 1e-5
    theta0 = np.random.default_rng(7).uniform(-0.1, 0.1, 4)  # vqe's own initialization for seed=7
    E = lambda th: np.vdot(q.ansatz(th, 2, 1), sum(c * kron_all(*[q.PAULI[p] for p in w]) for c, w in q.H2)  # noqa
                           @ q.ansatz(th, 2, 1).reshape(-1)).real
    grad = np.array([(E(theta0 + h * e) - E(theta0 - h * e)) / (2 * h) for e in np.eye(4)])
    return theta0 - lr * grad, q.vqe(q.H2, steps=1, lr=lr, seed=7)[0]


@equation("5.3", "Variational principle",
          term("5.3", r"E(\theta)") + r" = \langle\psi(\theta)|H|\psi(\theta)\rangle \;\ge\; E_0 = \lambda_{\min}(H)",
          "E(θ) ≥ E₀ for every θ", "vqe", {"5.3": "the energy VQE minimizes"},
          relation="≥", note="Code value: the lowest energy over 300 random circuits; bound: exact eigenvalue.")
def _(rng):
    Hm = sum(c * kron_all(*[q.PAULI[p] for p in w]) for c, w in q.H2)
    lowest = min(q.expectation(q.ansatz(th, 2, 1), q.H2) for th in rng.uniform(-np.pi, np.pi, (300, 4)))
    return np.linalg.eigvalsh(Hm)[0], lowest


# =============================================================== Part 6 ====
@equation("6.1", "MaxCut cost",
          r"C(z) = \sum_{(i,j)\in E}" + term("6.1b", r"[\,z_i \neq z_j\,]") + r",\qquad " + term("6.1a", r"z_i = \text{bit } i \text{ of } z"),
          "C(z) = Σ_{(i,j)∈E} [z_i ≠ z_j]", "cut_values",
          {"6.1a": "read the bits of every z at once", "6.1b": "count the edges whose ends differ"})
def _(rng):
    n, edges = 6, [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3)]
    formula = [sum(format(z, "06b")[i] != format(z, "06b")[j] for i, j in edges) for z in range(2 ** n)]
    return np.array(formula), q.cut_values(n, edges)


@equation("6.2", "QAOA state",
          r"|\gamma,\beta\rangle = \prod_{\ell=1}^{p}\Big[" + term("6.2c", r"e^{-i\beta_\ell \sum_j X_j}") + r"\,"
          + term("6.2b", r"e^{-i\gamma_\ell C}") + r"\Big]\," + term("6.2a", r"|+\rangle^{\otimes n}"),
          "|γ,β⟩ = Π_ℓ [e^{−iβ_ℓ Σ X_j} e^{−iγ_ℓ C}] |+⟩⊗ⁿ", "qaoa_state",
          {"6.2a": "uniform start, amplitude 2^{−n/2}", "6.2b": "cost layer: a phase e^{−iγC(z)} per basis state",
           "6.2c": "mixer: e^{−iβX} = cos β I − i sin β X on every qubit"},
          note="Formula side exponentiates the full 2ⁿ×2ⁿ matrices; random graph on 5 nodes, p = 2.")
def _(rng):
    n, edges = 5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)]
    gammas, betas = rng.uniform(0, np.pi, 2), rng.uniform(0, np.pi, 2)
    Cm = np.diag([sum(format(z, "05b")[i] != format(z, "05b")[j] for i, j in edges) for z in range(32)]).astype(float)
    B = sum(kron_all(*[q.X if j == k else q.I2 for k in range(n)]) for j in range(n))
    psi = np.full(32, 1 / np.sqrt(32), complex)
    for g, b in zip(gammas, betas):
        psi = expm_hermitian(B, b) @ expm_hermitian(Cm, g) @ psi
    return psi, q.qaoa_state(n, edges, gammas, betas).reshape(-1)


@equation("6.3", "QAOA p = 1 closed form (Wang et al. 2018)",
          r"\langle C_{uv}\rangle = \tfrac12 + " + term("6.3b", r"\tfrac14\sin 4\beta\,\sin\gamma\,(\cos^{d}\gamma + \cos^{e}\gamma)")
          + r" \\ - " + term("6.3c", r"\tfrac14\sin^{2}2\beta\,\cos^{d+e-2f}\gamma\,(1-\cos^{f}2\gamma)")
          + r" \\ " + term("6.3a", r"d, e, f") + r":\ \text{other neighbours of } u, v;\ \text{triangles on } uv",
          "⟨C_uv⟩ = ½ + ¼ sin4β sinγ (cos^d γ + cos^e γ) − ¼ sin²2β cos^{d+e−2f} γ (1 − cos^f 2γ)", "qaoa_p1_formula",
          {"6.3a": "other neighbours of u and v, and triangles on the edge", "6.3b": "first correction",
           "6.3c": "second correction (vanishes without triangles)"},
          note="Here the code *is* the formula; the independent side is the brute-force physics of eq 6.2.")
def _(rng):
    n, edges = 5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)]
    Cm = np.diag([sum(format(z, "05b")[i] != format(z, "05b")[j] for i, j in edges) for z in range(32)]).astype(float)
    B = sum(kron_all(*[q.X if j == k else q.I2 for k in range(n)]) for j in range(n))
    formula, code = [], []
    for g, b in rng.uniform(0, np.pi, (6, 2)):
        psi = expm_hermitian(B, b) @ expm_hermitian(Cm, g) @ np.full(32, 1 / np.sqrt(32), complex)
        formula.append(np.vdot(psi, Cm @ psi).real)
        code.append(q.qaoa_p1_formula(n, edges, g, b))
    return np.array(formula), np.array(code)


# =============================================================== Part 7 ====
@equation("7.1", "Black-Scholes price of a European call",
          term("7.1b", r"C = S_0\,N(d_1) - K e^{-rT} N(d_1 - \sigma\sqrt T)") + r" \\ "
          + term("7.1a", r"d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt T}"),
          "C = S₀ N(d₁) − K e^{−rT} N(d₂),  d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)", "european_call",
          {"7.1a": "d₁", "7.1b": "the price"}, tol=1e-6,
          note="Formula side: integrate the discounted payoff against the lognormal density, numerically.")
def _(rng):
    S0, K, sigma, r, T = 100.0, 105.0, 0.2, 0.05, 1.0
    x = np.linspace(-10, 10, 400_001)  # standard normal variable
    S = S0 * np.exp((r - sigma ** 2 / 2) * T + sigma * np.sqrt(T) * x)
    integrand = np.maximum(S - K, 0) * np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)
    integral = np.sum((integrand[1:] + integrand[:-1]) / 2 * np.diff(x))  # trapezoid rule
    return math.exp(-r * T) * integral, q.european_call()[3]


@equation("7.2", "Loading the price into an amplitude",
          term("7.2a", r"A|0\rangle = \sum_i \sqrt{p_i}\,|i\rangle\big(\sqrt{1-f_i}\,|0\rangle + \sqrt{f_i}\,|1\rangle\big)")
          + r" \\ " + term("7.2b", r"a = \sum_i p_i f_i") + r" = P(\text{last qubit} = 1)",
          "A|0⟩ = Σ_i √p_i |i⟩ (√(1−f_i)|0⟩ + √f_i|1⟩),  a = Σ p_i f_i = P(last qubit = 1)", "amplitude_estimation",
          {"7.2a": "the state, built in one line", "7.2b": "the number we want to estimate"},
          note="Formula side: build each |i⟩ ⊗ (…) with Kronecker products and measure the last qubit.")
def _(rng):
    p, f, *_ = q.european_call()
    state = sum(np.sqrt(p[i]) * np.kron(np.eye(16)[i], [np.sqrt(1 - f[i]), np.sqrt(f[i])]) for i in range(16))
    return np.sum(np.abs(state[1::2]) ** 2), q.amplitude_estimation(p, f, 3)[2]


@equation("7.3", "A Householder reflection prepares the state",
          term("7.3b", r"A = I - 2\,\frac{w w^{T}}{w^{T} w}") + r",\quad " + term("7.3a", r"w = |0\rangle - |\psi\rangle")
          + r" \\ \Rightarrow\; A|0\rangle = |\psi\rangle,\quad A^{T}A = I",
          "A = I − 2 wwᵀ/(wᵀw),  w = |0⟩ − |ψ⟩  ⇒  A|0⟩ = |ψ⟩,  AᵀA = I", "amplitude_estimation",
          {"7.3a": "the mirror direction", "7.3b": "the reflection: real, symmetric, its own inverse"},
          note="An identity the code relies on; checked on a random real unit vector.")
def _(rng):
    psi = rng.normal(size=16)
    psi /= np.linalg.norm(psi)
    w = np.eye(16)[0] - psi
    A = np.eye(16) - 2 * np.outer(w, w) / (w @ w)
    return np.concatenate([psi, np.eye(16).ravel()]), np.concatenate([A[:, 0], (A.T @ A).ravel()])


@equation("7.4", "Amplitude estimation: two Fejér peaks",
          r"\mathrm{eig}\big(" + term("7.4c", r"Q = -A" + term("7.4a", "S_0") + r"A^{T}" + term("7.4b", r"S_\chi"))
          + r"\big) \ni e^{\pm 2i\theta_a},\; \sin^2\theta_a = a \\ \Rightarrow\; "
          + term("7.4d", r"P(y) = \tfrac12\big[F_M\!\big(\tfrac{\theta_a}{\pi} - \tfrac{y}{M}\big) + F_M\!\big(-\tfrac{\theta_a}{\pi} - \tfrac{y}{M}\big)\big]"),
          "Q = −A S₀ Aᵀ S_χ has eigenvalues e^{±2iθ_a}, so P(y) = ½[F_M(θ_a/π − y/M) + F_M(−θ_a/π − y/M)]",
          "amplitude_estimation",
          {"7.4a": "S₀ flips the sign of |0⟩", "7.4b": "S_χ flips the 'good' states (last qubit 1)",
           "7.4c": "Grover's operator for this problem", "7.4d": "phase estimation on A|0⟩"},
          note="F_M is the Fejér kernel of eq 2.2; A|0⟩ splits evenly between the two eigenvectors.")
def _(rng):
    p, f, *_ = q.european_call()
    t = int(rng.integers(3, 8))
    M, a = 2 ** t, float(p @ f)
    th, y = np.arcsin(np.sqrt(a)), np.arange(M)
    return 0.5 * (fejer(th / np.pi - y / M, M) + fejer(-th / np.pi - y / M, M)), q.amplitude_estimation(p, f, t)[1]


@equation("7.5", "Brassard-Høyer-Mosca-Tapp guarantee",
          r"P\Big(|\tilde a - a| \le \tfrac{2\pi\sqrt{a(1-a)}}{M} + \tfrac{\pi^2}{M^2}\Big) \ge \tfrac{8}{\pi^2} \\ "
          + term("7.5a", r"\tilde a_y = \sin^2(\pi y/M)"),
          "P(|ã − a| ≤ 2π√(a(1−a))/M + π²/M²) ≥ 8/π²,  ã_y = sin²(πy/M)", "amplitude_estimation",
          {"7.5a": "turn the measured y into an estimate"}, relation="≥",
          note="Code value: the worst case over M = 8 … 512.")
def _(rng):
    p, f, *_ = q.european_call()
    worst = 1.0
    for t in range(3, 10):
        est, probs, a = q.amplitude_estimation(p, f, t)
        M = 2 ** t
        worst = min(worst, probs[np.abs(est - a) <= 2 * np.pi * np.sqrt(a * (1 - a)) / M + np.pi ** 2 / M ** 2].sum())
    return 8 / np.pi ** 2, worst


@equation("7.6", "Monte Carlo error is binomial",
          r"P\big(\hat a = \tfrac{k}{M}\big) = " + term("7.6a", r"\binom{M}{k}") + term("7.6b", r"a^{k}(1-a)^{M-k}"),
          "P(â = k/M) = C(M, k) a^k (1 − a)^{M−k}", "monte_carlo_error",
          {"7.6a": "binomial coefficient, via log-gamma", "7.6b": "probability of k successes"},
          note="Formula side: exact integers with math.comb, then the 80% error by brute force.")
def _(rng):
    a, M = rng.uniform(0.05, 0.95), int(rng.integers(20, 120))
    pmf = {k: math.comb(M, k) * a ** k * (1 - a) ** (M - k) for k in range(M + 1)}
    for eps in sorted({abs(k / M - a) for k in pmf}):
        if sum(v for k, v in pmf.items() if abs(k / M - a) <= eps + 1e-15) >= 0.8:
            break
    return eps, q.monte_carlo_error(a, M)


# ============================================================ the engine ===
BY_ID = {e.id: e for e in EQS}


def run(eq, seed=0):
    """Evaluate one equation. Returns a dict with both sides, the gap and the verdict."""
    formula, code = eq.check(np.random.default_rng(seed))
    formula, code = np.asarray(formula), np.asarray(code)
    if eq.relation == "=":
        gap = float(np.max(np.abs(formula - code))) if formula.size else 0.0
        ok = formula.shape == code.shape and gap <= eq.tol
    elif eq.relation == "≥":
        gap, ok = float(code - formula), bool(code >= formula - eq.tol)
    else:
        gap, ok = float(formula - code), bool(code <= formula + eq.tol)
    return {"id": eq.id, "ok": bool(ok), "relation": eq.relation, "gap": gap,
            "formula": summarize(formula), "code": summarize(code), "shape": list(formula.shape)}


def summarize(x, k=3):
    flat = np.atleast_1d(x).ravel()
    fmt = lambda v: (f"{v.real:+.6f}{v.imag:+.6f}i" if abs(v.imag) > 1e-12 else f"{v.real:.6g}")  # noqa: E731
    head = ", ".join(fmt(v) for v in flat[:k])
    return head if flat.size <= k else f"[{head}, … {flat.size} values]"


def source(eq):
    """The implementing function's lines: [(line number, text, [tags of this equation])]."""
    lines, start = inspect.getsourcelines(getattr(q, eq.func))
    out = []
    for i, text in enumerate(lines):
        tags = [t.strip() for m in TAG.finditer(text) for t in m.group(1).split(",")]
        mine = [t for t in tags if re.sub(r"[a-z]$", "", t) == eq.id]
        out.append((start + i, text.rstrip("\n"), mine))
    return out


def all_tags():
    """Every tag that appears anywhere in nanoqalgos.py, with its line numbers."""
    found = {}
    for n, text in enumerate(inspect.getsource(q).splitlines(), 1):
        for m in TAG.finditer(text):
            for t in m.group(1).split(","):
                found.setdefault(t.strip(), []).append(n)
    return found


def plain_tex(tex, underbrace=False):
    """Remove the \\htmlClass wrappers (GitHub's MathJax does not know them). With
    underbrace=True, keep the link visible as a letter under each term instead."""
    out, i = "", 0
    while True:
        j = tex.find(r"\htmlClass{", i)
        if j < 0:
            return out + tex[i:]
        k = tex.index("}{", j)
        tag = tex[j + 11:k][1:].replace("_", ".")
        depth, m = 1, k + 2
        while depth:  # find the matching closing brace of the content
            depth += {"{": 1, "}": -1}.get(tex[m], 0)
            m += 1
        inner = plain_tex(tex[k + 2:m - 1], underbrace)
        label = tag[-1] if tag[-1].isalpha() else tag  # "3.2a" -> "a", "3.3" -> "3.3"
        out += tex[i:j] + (r"\underbrace{" + inner + r"}_{\text{" + label + "}}" if underbrace else inner)
        i = m


def line_ranges(numbers):
    """[154, 155, 158] -> [(154, 155), (158, 158)]"""
    out = []
    for n in sorted(set(numbers)):
        if out and n == out[-1][1] + 1:
            out[-1] = (out[-1][0], n)
        else:
            out.append((n, n))
    return out


def markdown(seed=0, repo="nanoqalgos.py"):
    """MATH.md: every equation, its terms, the exact code, and the check, for GitHub."""
    md = ["# The math, next to the code",
          "",
          "Every equation in [`nanoqalgos.py`](nanoqalgos.py), with the lines that implement each term and a",
          "numerical check that the two agree. Letters under the formula (a, b, c) match the `# [eq X.Ya]`",
          "markers in the source. The *formula* side of each check is computed straight from the equation",
          "with plain NumPy, never by calling the code under test.",
          "",
          "Regenerate with `nanoqalgos verify --markdown > MATH.md`; run the checks with `nanoqalgos verify`.",
          "Interactive version, where hovering a term lights up its code: [docs/math.html](https://nakka1.github.io/nano-qalgos/math.html).",
          ""]
    for part, name in PARTS.items():
        md.append(f"- **Part {part} · {name}**: " + ", ".join(
            f"[{e.id}](#eq-{e.id.replace('.', '')})" for e in EQS if e.part == part))
    for e in EQS:
        r = run(e, seed)
        src = source(e)
        md += ["", f'<a id="eq-{e.id.replace(".", "")}"></a>', f"## eq {e.id} · {e.title}", "",
               "```math", plain_tex(e.tex, underbrace=True), "```", "",
               "| term | what it is | lines |", "|---|---|---|"]
        for tag, meaning in e.terms.items():
            nums = [n for n, _, tags in src if tag in tags]
            links = ", ".join(f"[{a}]({repo}#L{a})" if a == b else f"[{a}–{b}]({repo}#L{a}-L{b})"
                              for a, b in line_ranges(nums))
            label = tag[-1] if tag[-1].isalpha() else tag
            md.append(f"| **{label}** | {meaning} | {links} |")
        md += ["", f"`{repo}`, lines {src[0][0]}–{src[-1][0]}:", "", "```python"]
        md += [text for _, text, _ in src]
        md += ["```", ""]
        verdict = "✅" if r["ok"] else "❌"
        if r["relation"] == "=":
            md.append(f"{verdict} **formula = code**: max |formula − code| = {r['gap']:.1e} "
                      f"over {max(1, int(np.prod(r['shape'])))} value(s) (tolerance {e.tol:g}).")
        else:
            md.append(f"{verdict} **code {r['relation']} bound**: bound {r['formula']}, code {r['code']}.")
        if e.note:
            md += ["", f"<sub>{e.note}</sub>"]
    return "\n".join(md) + "\n"
