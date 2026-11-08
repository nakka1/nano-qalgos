"""Every claim the README makes is checked here.  Run: pytest -q"""
import math

import numpy as np
import pytest

import nanoqalgos as q
import nanoqalgos_cli

rng = np.random.default_rng(0)


def random_state(n):
    v = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    return v / np.linalg.norm(v)


# --- Part 0: simulator --------------------------------------------------------
def test_apply_matches_kronecker_product():
    psi = random_state(3)
    U = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0]
    want = np.kron(np.eye(2), U) @ psi  # U on qubits (1, 2)
    assert np.allclose(q.apply(psi.reshape(2, 2, 2), U, 1, 2).reshape(-1), want)


# --- Part 1: QFT ----------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 5, 9])
def test_qft_is_the_dft(n):
    v = random_state(n)
    out = q.qft(v.reshape((2,) * n)).reshape(-1)
    assert np.allclose(out, np.sqrt(2 ** n) * np.fft.ifft(v), atol=1e-12)
    back = q.qft(out.reshape((2,) * n), inverse=True).reshape(-1)
    assert np.allclose(back, v, atol=1e-12)


# --- Part 2: phase estimation ---------------------------------------------------
@pytest.mark.parametrize("phi", [0.0, 0.25, 0.3141, 0.7777])
def test_qpe_matches_fejer_kernel(phi):
    t, M = 6, 64
    probs = q.phase_estimation(np.diag([np.exp(2j * np.pi * phi), 1]), np.array([1, 0]), t)
    y = np.arange(M)
    fejer = np.abs(np.exp(2j * np.pi * np.outer(y * 0 + 1, np.arange(M))
                          * (phi - y[:, None] / M)).sum(1) / M) ** 2
    assert np.allclose(probs, fejer, atol=1e-12)
    assert probs.max() >= 4 / np.pi ** 2 - 1e-12


# --- Part 3: Grover ---------------------------------------------------------------
@pytest.mark.parametrize("n,marked", [(3, [5]), (6, [1, 40]), (10, [777])])
def test_grover_rotation_formula(n, marked):
    for k in range(0, 6):
        psi, theta = q.grover(n, marked, iterations=k)
        p = q.probabilities(psi)[marked].sum()
        assert p == pytest.approx(np.sin((2 * k + 1) * theta) ** 2, abs=1e-10)


# --- Part 4: Shor -----------------------------------------------------------------
def test_convergents():
    assert list(q.convergents(415, 93)) == [(4, 1), (9, 2), (58, 13), (415, 93)]


@pytest.mark.parametrize("N,a,r", [(15, 7, 4), (21, 2, 6), (35, 3, 12)])
def test_order_finding_peaks_at_multiples_of_one_over_r(N, a, r):
    probs, t = q.order_finding(a, N)
    peaks = np.argsort(probs)[::-1][:r]
    for y in peaks:  # every peak is within 1/2 of some s * 2^t / r
        assert min(abs(y - s * 2 ** t / r) for s in range(r + 1)) <= 0.5
    assert probs[peaks].sum() > 0.6


@pytest.mark.parametrize("N", [15, 21, 33, 35])
def test_shor_factors(N):
    fac, *_ = q.shor(N, seed=1)
    assert fac[0] * fac[1] == N and 1 < fac[0] < N


# --- Part 5: VQE ------------------------------------------------------------------
def test_h2_hamiltonian_and_vqe():
    exact = np.linalg.eigvalsh(q.hamiltonian_matrix(q.H2))[0]
    assert exact == pytest.approx(-1.857275030202, abs=1e-9)
    theta, hist = q.vqe(q.H2)
    assert np.all(hist >= exact - 1e-12)  # variational principle
    assert hist[-1] - exact < 1.6e-3  # chemical accuracy


def test_parameter_shift_is_the_exact_derivative():
    th, h = rng.uniform(-1, 1, 4), 1e-6
    E = lambda x: q.expectation(q.ansatz(x, 2, 1), q.H2)  # noqa: E731
    for i in range(4):
        e = np.eye(4)[i]
        shift = (E(th + e * np.pi / 2) - E(th - e * np.pi / 2)) / 2
        fd = (E(th + e * h) - E(th - e * h)) / (2 * h)
        assert shift == pytest.approx(fd, abs=1e-7)


# --- Part 6: QAOA -----------------------------------------------------------------
GRAPHS = [(3, [(0, 1), (1, 2), (0, 2)]), (6, [(i, (i + 1) % 6) for i in range(6)]),
          nanoqalgos_cli.CUBE, (5, [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (1, 4)])]


@pytest.mark.parametrize("graph", GRAPHS)
def test_qaoa_matches_closed_form(graph):
    n, edges = graph
    for g, b in rng.uniform(0, np.pi, (6, 2)):
        assert q.qaoa_expectation(n, edges, [g], [b]) == pytest.approx(
            q.qaoa_p1_formula(n, edges, g, b), abs=1e-10)


def test_qaoa_three_regular_bound():
    n, edges = nanoqalgos_cli.CUBE  # triangle-free 3-regular: optimum ratio 0.6924
    _, _, val = q.qaoa_maxcut(n, edges)
    assert val / len(edges) == pytest.approx(0.6924, abs=2e-4)


# --- Part 7: amplitude estimation ---------------------------------------------------
def test_grover_operator_eigenphases():
    p, f, *_ = q.european_call(n=3)
    a = p @ f
    psi = (np.sqrt(p)[:, None] * np.stack([np.sqrt(1 - f), np.sqrt(f)], 1)).reshape(-1)
    est, probs, a2 = q.amplitude_estimation(p, f, t=3)
    assert a2 == pytest.approx(a)
    assert psi @ psi == pytest.approx(1)


@pytest.mark.parametrize("t", range(3, 11))
def test_qae_brassard_guarantee(t):
    p, f, *_ = q.european_call()
    est, probs, a = q.amplitude_estimation(p, f, t)
    M = 2 ** t
    bound = 2 * np.pi * np.sqrt(a * (1 - a)) / M + np.pi ** 2 / M ** 2
    assert probs[np.abs(est - a) <= bound].sum() >= 8 / np.pi ** 2


def test_qae_beats_monte_carlo_at_equal_cost():
    p, f, *_ = q.european_call()
    est, probs, a = q.amplitude_estimation(p, f, 10)
    calls = 2 * (2 ** 10 - 1) + 1
    assert q.error_at_confidence(est, probs, a) < q.monte_carlo_error(a, calls) / 5


def test_monte_carlo_error_matches_normal_approximation():
    a, M = 0.3, 4000
    z = 1.2815515655446004
    assert q.monte_carlo_error(a, M) == pytest.approx(z * math.sqrt(a * (1 - a) / M), rel=0.05)


# --- CLI ------------------------------------------------------------------------------
@pytest.mark.parametrize("argv", [["demo"], ["shor", "21"], ["grover", "4"]])
def test_cli_runs(argv, capsys):
    nanoqalgos_cli.main(argv)
    assert capsys.readouterr().out
