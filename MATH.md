# The math, next to the code

Every equation in [`nanoqalgos.py`](nanoqalgos.py), with the lines that implement each term and a
numerical check that the two agree. Letters under the formula (a, b, c) match the `# [eq X.Ya]`
markers in the source. The *formula* side of each check is computed straight from the equation
with plain NumPy, never by calling the code under test.

Regenerate with `nanoqalgos verify --markdown > MATH.md`; run the checks with `nanoqalgos verify`.
Interactive version, where hovering a term lights up its code: [docs/math.html](https://nakka1.github.io/nano-qalgos/math.html).

- **Part 0 · Simulator**: [0.1](#eq-01), [0.2](#eq-02), [0.3](#eq-03), [0.4](#eq-04)
- **Part 1 · Quantum Fourier transform**: [1.1](#eq-11), [1.2](#eq-12), [1.3](#eq-13)
- **Part 2 · Phase estimation**: [2.1](#eq-21), [2.2](#eq-22), [2.3](#eq-23)
- **Part 3 · Grover search**: [3.1](#eq-31), [3.2](#eq-32), [3.3](#eq-33), [3.4](#eq-34)
- **Part 4 · Shor's algorithm**: [4.1](#eq-41), [4.2](#eq-42), [4.3](#eq-43)
- **Part 5 · VQE**: [5.1](#eq-51), [5.2](#eq-52), [5.3](#eq-53)
- **Part 6 · QAOA**: [6.1](#eq-61), [6.2](#eq-62), [6.3](#eq-63)
- **Part 7 · Amplitude estimation**: [7.1](#eq-71), [7.2](#eq-72), [7.3](#eq-73), [7.4](#eq-74), [7.5](#eq-75), [7.6](#eq-76)

<a id="eq-01"></a>
## eq 0.1 · A gate is a contraction on its qubit's axis

```math
(U_{q}\psi)_{i_1\dots \underbrace{i_q}_{\text{c}}\dots i_n} = \underbrace{\sum_{j}}_{\text{b}}\;\underbrace{U_{i_q j}}_{\text{a}}\;\psi_{i_1\dots j \dots i_n}
```

| term | what it is | lines |
|---|---|---|
| **a** | U reshaped into output and input legs | [78](nanoqalgos.py#L78) |
| **b** | sum over the input leg (np.tensordot) | [79](nanoqalgos.py#L79) |
| **c** | output leg moved back to the qubit's axis | [80](nanoqalgos.py#L80) |

`nanoqalgos.py`, lines 75–80:

```python
def apply(psi, U, *qubits):
    """Apply a k-qubit gate U to the listed axes of psi."""
    k = len(qubits)
    U = U.reshape((2,) * 2 * k)  # k output legs, then k input legs [eq 0.1a]
    psi = np.tensordot(U, psi, axes=(list(range(k, 2 * k)), list(qubits)))  # [eq 0.1b]
    return np.moveaxis(psi, list(range(k)), list(qubits))  # put legs back [eq 0.1c]
```

✅ **formula = code**: max |formula − code| = 5.6e-17 over 8 value(s) (tolerance 1e-09).

<sub>Checked on a 2-qubit gate acting on qubits (2, 0) of 3, out of order on purpose.</sub>

<a id="eq-02"></a>
## eq 0.2 · Controlled gate

```math
C(U) = \underbrace{|0\rangle\langle 0|\otimes I}_{\text{a}} + \underbrace{|1\rangle\langle 1|\otimes U}_{\text{b}}
```

| term | what it is | lines |
|---|---|---|
| **a** | start from the identity (the |0⟩ block) | [64](nanoqalgos.py#L64) |
| **b** | overwrite the |1⟩ block with U | [65](nanoqalgos.py#L65) |

`nanoqalgos.py`, lines 61–66:

```python
def controlled(U):
    """|0><0| ⊗ I + |1><1| ⊗ U. The control is the first qubit."""
    d = U.shape[0]
    out = np.eye(2 * d, dtype=complex)  # [eq 0.2a]
    out[d:, d:] = U  # [eq 0.2b]
    return out
```

✅ **formula = code**: max |formula − code| = 0.0e+00 over 64 value(s) (tolerance 1e-09).

<a id="eq-03"></a>
## eq 0.3 · Y rotation

```math
\begin{gathered}R_Y(\theta) = e^{-i\theta Y/2} = \underbrace{\cos\tfrac{\theta}{2}\,I - i\sin\tfrac{\theta}{2}\,Y}_{\text{a}} \\ = \underbrace{\begin{pmatrix}\cos\frac{\theta}{2} & -\sin\frac{\theta}{2}\\ \sin\frac{\theta}{2} & \cos\frac{\theta}{2}\end{pmatrix}}_{\text{b}}\end{gathered}
```

| term | what it is | lines |
|---|---|---|
| **a** | cos and sin of the half angle | [57](nanoqalgos.py#L57) |
| **b** | the resulting real 2×2 matrix | [58](nanoqalgos.py#L58) |

`nanoqalgos.py`, lines 55–58:

```python
def ry(theta):
    """exp(-iθY/2): a real rotation, cos(θ/2)|0> + sin(θ/2)|1> from |0>."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)  # [eq 0.3a]
    return np.array([[c, -s], [s, c]], dtype=complex)  # [eq 0.3b]
```

✅ **formula = code**: max |formula − code| = 2.2e-16 over 4 value(s) (tolerance 1e-09).

<sub>The formula side computes the matrix exponential by diagonalizing Y.</sub>

<a id="eq-04"></a>
## eq 0.4 · Born rule

```math
P(x) = \underbrace{\big|\langle x|\psi\rangle\big|^{2}}_{\text{0.4}}
```

| term | what it is | lines |
|---|---|---|
| **0.4** | squared magnitude of every amplitude | [85](nanoqalgos.py#L85) |

`nanoqalgos.py`, lines 83–85:

```python
def probabilities(psi):
    """Born rule: P(outcome) = |amplitude|^2, in flat (binary) order."""
    return np.abs(psi.reshape(-1)) ** 2  # [eq 0.4]
```

✅ **formula = code**: max |formula − code| = 1.1e-16 over 16 value(s) (tolerance 1e-09).

<a id="eq-11"></a>
## eq 1.1 · QFT circuit = DFT matrix

```math
\begin{gathered}\underbrace{\mathrm{REV}}_{\text{c}}\;\prod_{j}\Big[\prod_{k>j}\underbrace{CR^{\,k\to j}_{k-j+1}}_{\text{b}}\Big]\,\underbrace{H_j}_{\text{a}}\;=\;F \\ F_{kj} = \tfrac{1}{\sqrt N}\,e^{2\pi i\,jk/N},\quad R_m = \mathrm{diag}(1, e^{2\pi i/2^m})\end{gathered}
```

| term | what it is | lines |
|---|---|---|
| **a** | Hadamard on qubit j | [104](nanoqalgos.py#L104) |
| **b** | controlled phase R_m from every later qubit | [105–107](nanoqalgos.py#L105-L107) |
| **c** | swaps reverse the qubit order | [108–109](nanoqalgos.py#L108-L109) |

`nanoqalgos.py`, lines 98–110:

```python
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
```

✅ **formula = code**: max |formula − code| = 5.6e-15 over 1024 value(s) (tolerance 1e-09).

<sub>The code side builds the full matrix by running the circuit on every basis state (n = 5).</sub>

<a id="eq-12"></a>
## eq 1.2 · QFT output is a product state

```math
\begin{gathered}\mathrm{QFT}\,|j\rangle = \underbrace{\bigotimes_{\ell=1}^{n}}_{\text{c}}\tfrac{1}{\sqrt2}\Big(|0\rangle + \underbrace{e^{i\pi j_{\ell-1}}}_{\text{a}}\underbrace{\prod_{m<\ell-1} e^{2\pi i\, j_m 2^{m-\ell}}}_{\text{b}}\,|1\rangle\Big) \\ j = \textstyle\sum_m j_m 2^m\end{gathered}
```

| term | what it is | lines |
|---|---|---|
| **a** | Hadamard: the sign from bit j_{ℓ−1} | [104](nanoqalgos.py#L104) |
| **b** | controlled phases: the finer bits | [105–107](nanoqalgos.py#L105-L107) |
| **c** | factors come out reversed; swaps put them in order | [108–109](nanoqalgos.py#L108-L109) |

`nanoqalgos.py`, lines 98–110:

```python
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
```

✅ **formula = code**: max |formula − code| = 3.3e-15 over 64 value(s) (tolerance 1e-09).

<a id="eq-13"></a>
## eq 1.3 · Inverse QFT

```math
\underbrace{\mathrm{QFT}^{\dagger} = \overline{\mathrm{QFT}}}_{\text{1.3}}\quad\text{because}\quad F^{T} = F
```

| term | what it is | lines |
|---|---|---|
| **1.3** | conjugate, transform, conjugate back | [99](nanoqalgos.py#L99) |

`nanoqalgos.py`, lines 98–110:

```python
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
```

✅ **formula = code**: max |formula − code| = 4.0e-15 over 32 value(s) (tolerance 1e-09).

<a id="eq-21"></a>
## eq 2.1 · Phase estimation, any input state

```math
P(y) = \underbrace{\Big\|}_{\text{d}}\underbrace{\tfrac{1}{\sqrt M}\sum_{k=0}^{M-1} e^{-2\pi i\,ky/M}}_{\text{c}}\,\underbrace{\tfrac{1}{\sqrt M}}_{\text{b}}\,\underbrace{U^{k}|\psi\rangle}_{\text{a}}\underbrace{\Big\|^{2}}_{\text{d}}
```

| term | what it is | lines |
|---|---|---|
| **a** | row k of the register holds U^k ψ | [132–133](nanoqalgos.py#L132-L133) |
| **b** | uniform superposition amplitude 1/√M | [134](nanoqalgos.py#L134) |
| **c** | inverse QFT on the counting register | [135](nanoqalgos.py#L135) |
| **d** | probability of y: sum over the work register | [136](nanoqalgos.py#L136) |

`nanoqalgos.py`, lines 127–136:

```python
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
```

✅ **formula = code**: max |formula − code| = 4.2e-16 over 16 value(s) (tolerance 1e-09).

<sub>Random 4×4 unitary and random ψ (not an eigenvector), t = 4.</sub>

<a id="eq-22"></a>
## eq 2.2 · Eigenvector input: the Fejér kernel

```math
U|u\rangle = e^{2\pi i\varphi}|u\rangle \;\Rightarrow\; P(y) = \Big|\tfrac1M\sum_{k}\underbrace{e^{2\pi i\varphi k}}_{\text{a}}\underbrace{e^{-2\pi i\,ky/M}}_{\text{b}}\Big|^{2}
```

| term | what it is | lines |
|---|---|---|
| **a** | U^k on an eigenvector is just a phase | [132–133](nanoqalgos.py#L132-L133) |
| **b** | inverse QFT | [135](nanoqalgos.py#L135) |

`nanoqalgos.py`, lines 127–136:

```python
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
```

✅ **formula = code**: max |formula − code| = 4.8e-15 over 64 value(s) (tolerance 1e-09).

<a id="eq-23"></a>
## eq 2.3 · Nearest outcome has probability ≥ 4/π²

```math
\underbrace{\max_y P(y)}_{\text{2.3}} \;\ge\; \frac{4}{\pi^{2}} \approx 0.405
```

| term | what it is | lines |
|---|---|---|
| **2.3** | the output distribution | [136](nanoqalgos.py#L136) |

`nanoqalgos.py`, lines 127–136:

```python
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
```

✅ **code ≥ bound**: bound 0.405285, code 0.423905.

<sub>Code value: the worst case over 50 random phases.</sub>

<a id="eq-31"></a>
## eq 3.1 · Oracle: flip the answers

```math
O = I - 2\!\!\sum_{x\in\text{marked}}\!\!\underbrace{|x\rangle\langle x|}_{\text{3.1}}
```

| term | what it is | lines |
|---|---|---|
| **3.1** | negate the marked amplitudes | [154](nanoqalgos.py#L154) |

`nanoqalgos.py`, lines 152–155:

```python
def oracle(psi, marked):
    flat = psi.reshape(-1).copy()
    flat[list(marked)] *= -1  # [eq 3.1]
    return flat.reshape(psi.shape)
```

✅ **formula = code**: max |formula − code| = 0.0e+00 over 16 value(s) (tolerance 1e-09).

<a id="eq-32"></a>
## eq 3.2 · Diffusion is a reflection about |s⟩

```math
\underbrace{H^{\otimes n}}_{\text{a}}\big(\underbrace{2|0\rangle\langle 0|}_{\text{c}}\underbrace{\,-\,I}_{\text{b}}\big)\underbrace{H^{\otimes n}}_{\text{a}} = 2|s\rangle\langle s| - I,\qquad |s\rangle = H^{\otimes n}|0\rangle
```

| term | what it is | lines |
|---|---|---|
| **a** | Hadamard on every qubit | [160–161](nanoqalgos.py#L160-L161), [164–165](nanoqalgos.py#L164-L165) |
| **b** | − I: negate everything | [162](nanoqalgos.py#L162) |
| **c** | 2|0⟩⟨0|: restore |0⟩'s sign | [163](nanoqalgos.py#L163) |

`nanoqalgos.py`, lines 158–166:

```python
def diffusion(psi):
    n = psi.ndim
    for q in range(n):  # [eq 3.2a]
        psi = apply(psi, H, q)  # [eq 3.2a]
    psi = -psi  # [eq 3.2b]
    psi[(0,) * n] *= -1  # 2|0><0| - I [eq 3.2c, 3.3]
    for q in range(n):  # [eq 3.2a]
        psi = apply(psi, H, q)  # [eq 3.2a]
    return psi
```

✅ **formula = code**: max |formula − code| = 3.3e-16 over 32 value(s) (tolerance 1e-09).

<a id="eq-33"></a>
## eq 3.3 · Inversion about the mean

```math
a_x \;\mapsto\; 2\,\underbrace{\bar a}_{\text{3.3}} - a_x,\qquad \bar a = \tfrac1N\textstyle\sum_x a_x
```

| term | what it is | lines |
|---|---|---|
| **3.3** | this sign flip is what produces the mean | [163](nanoqalgos.py#L163) |

`nanoqalgos.py`, lines 158–166:

```python
def diffusion(psi):
    n = psi.ndim
    for q in range(n):  # [eq 3.2a]
        psi = apply(psi, H, q)  # [eq 3.2a]
    psi = -psi  # [eq 3.2b]
    psi[(0,) * n] *= -1  # 2|0><0| - I [eq 3.2c, 3.3]
    for q in range(n):  # [eq 3.2a]
        psi = apply(psi, H, q)  # [eq 3.2a]
    return psi
```

✅ **formula = code**: max |formula − code| = 2.9e-16 over 32 value(s) (tolerance 1e-09).

<a id="eq-34"></a>
## eq 3.4 · Grover success probability

```math
\begin{gathered}P_k = \sin^{2}\!\big(\underbrace{(2k+1)}_{\text{c}}\underbrace{\theta}_{\text{a}}\big) \\ \underbrace{\theta = \arcsin\sqrt{m/N}}_{\text{a}},\quad \underbrace{k^{\star} = \lfloor \pi/(4\theta)\rfloor}_{\text{b}}\end{gathered}
```

| term | what it is | lines |
|---|---|---|
| **a** | the rotation angle | [171](nanoqalgos.py#L171) |
| **b** | the best number of rounds | [173](nanoqalgos.py#L173) |
| **c** | each round: oracle + diffusion | [179](nanoqalgos.py#L179), [181](nanoqalgos.py#L181) |

`nanoqalgos.py`, lines 169–183:

```python
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
```

✅ **formula = code**: max |formula − code| = 1.7e-14 over 11 value(s) (tolerance 1e-09).

<sub>N = 128, 2 marked items, k = 0 … 10.</sub>

<a id="eq-41"></a>
## eq 4.1 · Eigenvectors of modular multiplication

```math
\begin{gathered}\underbrace{U}_{\text{b}}|y\rangle = \underbrace{|a y \bmod N\rangle}_{\text{a}} \\ U|u_s\rangle = e^{2\pi i s/r}|u_s\rangle \\ |u_s\rangle = \tfrac{1}{\sqrt r}\sum_{k=0}^{r-1} e^{-2\pi i sk/r}|a^{k} \bmod N\rangle\end{gathered}
```

| term | what it is | lines |
|---|---|---|
| **a** | the permutation y → a·y mod N | [201](nanoqalgos.py#L201) |
| **b** | written as a matrix, one 1 per column | [203](nanoqalgos.py#L203) |

`nanoqalgos.py`, lines 198–204:

```python
def modmul_matrix(a, N, n):
    """Permutation matrix of y -> a·y mod N on n qubits (y >= N left alone)."""
    image = np.arange(2 ** n)
    image[:N] = (a * image[:N]) % N  # [eq 4.1a]
    U = np.zeros((2 ** n, 2 ** n))
    U[image, np.arange(2 ** n)] = 1  # [eq 4.1b]
    return U
```

✅ **formula = code**: max |formula − code| = 1.1e-15 over 192 value(s) (tolerance 1e-09).

<sub>N = 21, a = 2 (order r = 6): every |u_s⟩ is an eigenvector of the code's matrix.</sub>

<a id="eq-42"></a>
## eq 4.2 · Continued-fraction convergents

```math
\begin{gathered}\frac{y}{2^{t}} = [a_0; a_1, a_2, \dots],\quad \underbrace{a_k = \lfloor x_k \rfloor}_{\text{a}} \\ \underbrace{h_k = a_k h_{k-1} + h_{k-2},\;\; q_k = a_k q_{k-1} + q_{k-2}}_{\text{b}}\end{gathered}
```

| term | what it is | lines |
|---|---|---|
| **a** | next partial quotient (Euclid's algorithm) | [213](nanoqalgos.py#L213) |
| **b** | the recurrence for numerator and denominator | [214](nanoqalgos.py#L214) |

`nanoqalgos.py`, lines 207–215:

```python
def convergents(num, den):
    """Best rational approximations p/q of num/den, from its continued fraction
    [a0; a1, a2, ...] and the recurrence h_k = a_k h_{k-1} + h_{k-2} (same for q)."""
    num, den = int(num), int(den)
    h0, h1, k0, k1 = 0, 1, 1, 0
    while den:
        a, (num, den) = num // den, (den, num % den)  # [eq 4.2a]
        h0, h1, k0, k1 = h1, a * h1 + h0, k1, a * k1 + k0  # [eq 4.2b]
        yield h1, k1
```

✅ **formula = code**: max |formula − code| = 0.0e+00 over 16 value(s) (tolerance 1e-09).

<sub>Formula side: evaluate each truncated fraction [a0; …, ak] directly with exact rationals.</sub>

<a id="eq-43"></a>
## eq 4.3 · From the order to the factors

```math
\begin{gathered}\underbrace{a^{r} \equiv 1 \pmod N}_{\text{a}},\;\; \underbrace{r\text{ even},\; a^{r/2} \not\equiv -1}_{\text{b}} \\ \Rightarrow\; (a^{r/2}-1)(a^{r/2}+1) \equiv 0 \\ \Rightarrow\; \underbrace{\gcd(a^{r/2}-1,\,N)}_{\text{c}}\text{ divides } N\end{gathered}
```

| term | what it is | lines |
|---|---|---|
| **a** | r is a period of a mod N | [230](nanoqalgos.py#L230) |
| **b** | the two conditions that make it useful | [231–232](nanoqalgos.py#L231-L232) |
| **c** | the factor, by Euclid's gcd | [233](nanoqalgos.py#L233) |

`nanoqalgos.py`, lines 227–237:

```python
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
```

✅ **formula = code**: max |formula − code| = 0.0e+00 over 8 value(s) (tolerance 1e-09).

<sub>Eight values of N; the code is fed the ideal measurement y = round(2^t / r).</sub>

<a id="eq-51"></a>
## eq 5.1 · Energy of a Pauli-sum Hamiltonian

```math
E = \langle\psi|H|\psi\rangle = \sum_{P} c_P\,\underbrace{\langle\psi|}_{\text{b}}\underbrace{P|\psi\rangle}_{\text{a}}
```

| term | what it is | lines |
|---|---|---|
| **a** | apply the Pauli string one qubit at a time | [279](nanoqalgos.py#L279) |
| **b** | overlap with ⟨ψ|, weighted by c_P | [280](nanoqalgos.py#L280) |

`nanoqalgos.py`, lines 272–281:

```python
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
```

✅ **formula = code**: max |formula − code| = 1.1e-16 over 1 value(s) (tolerance 1e-09).

<sub>Formula side: build H as a 4×4 matrix with Kronecker products, then ψ†Hψ.</sub>

<a id="eq-52"></a>
## eq 5.2 · Parameter-shift gradient

```math
\frac{\partial E}{\partial\theta_i} = \underbrace{\tfrac12\Big[E\big(\theta + \underbrace{\tfrac{\pi}{2}e_i}_{\text{a}}\big) - E\big(\theta - \tfrac{\pi}{2}e_i\big)\Big]}_{\text{b}}
```

| term | what it is | lines |
|---|---|---|
| **a** | shift one angle by ±π/2 | [314](nanoqalgos.py#L314) |
| **b** | half the difference: an exact derivative | [315](nanoqalgos.py#L315) |

`nanoqalgos.py`, lines 308–318:

```python
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
```

✅ **formula = code**: max |formula − code| = 1.0e-11 over 4 value(s) (tolerance 1e-08).

<sub>One step of vqe() must equal θ − lr·∇E, with ∇E from a plain finite difference.</sub>

<a id="eq-53"></a>
## eq 5.3 · Variational principle

```math
\underbrace{E(\theta)}_{\text{5.3}} = \langle\psi(\theta)|H|\psi(\theta)\rangle \;\ge\; E_0 = \lambda_{\min}(H)
```

| term | what it is | lines |
|---|---|---|
| **5.3** | the energy VQE minimizes | [310](nanoqalgos.py#L310) |

`nanoqalgos.py`, lines 308–318:

```python
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
```

✅ **code ≥ bound**: bound -1.85728, code -1.84648.

<sub>Code value: the lowest energy over 300 random circuits; bound: exact eigenvalue.</sub>

<a id="eq-61"></a>
## eq 6.1 · MaxCut cost

```math
C(z) = \sum_{(i,j)\in E}\underbrace{[\,z_i \neq z_j\,]}_{\text{b}},\qquad \underbrace{z_i = \text{bit } i \text{ of } z}_{\text{a}}
```

| term | what it is | lines |
|---|---|---|
| **a** | read the bits of every z at once | [333](nanoqalgos.py#L333) |
| **b** | count the edges whose ends differ | [334](nanoqalgos.py#L334) |

`nanoqalgos.py`, lines 332–334:

```python
def cut_values(n, edges):
    z = (np.arange(2 ** n)[:, None] >> (n - 1 - np.arange(n))) & 1  # bits of z [eq 6.1a]
    return sum((z[:, i] != z[:, j]).astype(int) for i, j in edges)  # [eq 6.1b]
```

✅ **formula = code**: max |formula − code| = 0.0e+00 over 64 value(s) (tolerance 1e-09).

<a id="eq-62"></a>
## eq 6.2 · QAOA state

```math
|\gamma,\beta\rangle = \prod_{\ell=1}^{p}\Big[\underbrace{e^{-i\beta_\ell \sum_j X_j}}_{\text{c}}\,\underbrace{e^{-i\gamma_\ell C}}_{\text{b}}\Big]\,\underbrace{|+\rangle^{\otimes n}}_{\text{a}}
```

| term | what it is | lines |
|---|---|---|
| **a** | uniform start, amplitude 2^{−n/2} | [339](nanoqalgos.py#L339) |
| **b** | cost layer: a phase e^{−iγC(z)} per basis state | [341](nanoqalgos.py#L341) |
| **c** | mixer: e^{−iβX} = cos β I − i sin β X on every qubit | [342–344](nanoqalgos.py#L342-L344) |

`nanoqalgos.py`, lines 337–345:

```python
def qaoa_state(n, edges, gammas, betas):
    c = cut_values(n, edges)
    psi = np.full(2 ** n, 2 ** (-n / 2), dtype=complex).reshape((2,) * n)  # [eq 6.2a]
    for g, b in zip(gammas, betas):
        psi = (np.exp(-1j * g * c) * psi.reshape(-1)).reshape(psi.shape)  # [eq 6.2b]
        mixer = np.cos(b) * I2 - 1j * np.sin(b) * X  # [eq 6.2c]
        for q in range(n):  # [eq 6.2c]
            psi = apply(psi, mixer, q)  # [eq 6.2c]
    return psi
```

✅ **formula = code**: max |formula − code| = 7.0e-16 over 32 value(s) (tolerance 1e-09).

<sub>Formula side exponentiates the full 2ⁿ×2ⁿ matrices; random graph on 5 nodes, p = 2.</sub>

<a id="eq-63"></a>
## eq 6.3 · QAOA p = 1 closed form (Wang et al. 2018)

```math
\begin{gathered}\langle C_{uv}\rangle = \tfrac12 + \underbrace{\tfrac14\sin 4\beta\,\sin\gamma\,(\cos^{d}\gamma + \cos^{e}\gamma)}_{\text{b}} \\ - \underbrace{\tfrac14\sin^{2}2\beta\,\cos^{d+e-2f}\gamma\,(1-\cos^{f}2\gamma)}_{\text{c}} \\ \underbrace{d, e, f}_{\text{a}}:\ \text{other neighbours of } u, v;\ \text{triangles on } uv\end{gathered}
```

| term | what it is | lines |
|---|---|---|
| **a** | other neighbours of u and v, and triangles on the edge | [360](nanoqalgos.py#L360) |
| **b** | first correction | [362](nanoqalgos.py#L362) |
| **c** | second correction (vanishes without triangles) | [363–364](nanoqalgos.py#L363-L364) |

`nanoqalgos.py`, lines 352–365:

```python
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
```

✅ **formula = code**: max |formula − code| = 5.3e-15 over 6 value(s) (tolerance 1e-09).

<sub>Here the code *is* the formula; the independent side is the brute-force physics of eq 6.2.</sub>

<a id="eq-71"></a>
## eq 7.1 · Black-Scholes price of a European call

```math
\begin{gathered}\underbrace{C = S_0\,N(d_1) - K e^{-rT} N(d_1 - \sigma\sqrt T)}_{\text{b}} \\ \underbrace{d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt T}}_{\text{a}}\end{gathered}
```

| term | what it is | lines |
|---|---|---|
| **a** | d₁ | [407](nanoqalgos.py#L407) |
| **b** | the price | [408](nanoqalgos.py#L408) |

`nanoqalgos.py`, lines 397–409:

```python
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
```

✅ **formula = code**: max |formula − code| = 1.5e-09 over 1 value(s) (tolerance 1e-06).

<sub>Formula side: integrate the discounted payoff against the lognormal density, numerically.</sub>

<a id="eq-72"></a>
## eq 7.2 · Loading the price into an amplitude

```math
\begin{gathered}\underbrace{A|0\rangle = \sum_i \sqrt{p_i}\,|i\rangle\big(\sqrt{1-f_i}\,|0\rangle + \sqrt{f_i}\,|1\rangle\big)}_{\text{a}} \\ \underbrace{a = \sum_i p_i f_i}_{\text{b}} = P(\text{last qubit} = 1)\end{gathered}
```

| term | what it is | lines |
|---|---|---|
| **a** | the state, built in one line | [414](nanoqalgos.py#L414) |
| **b** | the number we want to estimate | [424](nanoqalgos.py#L424) |

`nanoqalgos.py`, lines 412–424:

```python
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
```

✅ **formula = code**: max |formula − code| = 2.8e-17 over 1 value(s) (tolerance 1e-09).

<sub>Formula side: build each |i⟩ ⊗ (…) with Kronecker products and measure the last qubit.</sub>

<a id="eq-73"></a>
## eq 7.3 · A Householder reflection prepares the state

```math
\begin{gathered}\underbrace{A = I - 2\,\frac{w w^{T}}{w^{T} w}}_{\text{b}},\quad \underbrace{w = |0\rangle - |\psi\rangle}_{\text{a}} \\ \Rightarrow\; A|0\rangle = |\psi\rangle,\quad A^{T}A = I\end{gathered}
```

| term | what it is | lines |
|---|---|---|
| **a** | the mirror direction | [416](nanoqalgos.py#L416) |
| **b** | the reflection: real, symmetric, its own inverse | [417](nanoqalgos.py#L417) |

`nanoqalgos.py`, lines 412–424:

```python
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
```

✅ **formula = code**: max |formula − code| = 2.2e-16 over 272 value(s) (tolerance 1e-09).

<sub>An identity the code relies on; checked on a random real unit vector.</sub>

<a id="eq-74"></a>
## eq 7.4 · Amplitude estimation: two Fejér peaks

```math
\begin{gathered}\mathrm{eig}\big(\underbrace{Q = -A\underbrace{S_0}_{\text{a}}A^{T}\underbrace{S_\chi}_{\text{b}}}_{\text{c}}\big) \ni e^{\pm 2i\theta_a},\; \sin^2\theta_a = a \\ \Rightarrow\; \underbrace{P(y) = \tfrac12\big[F_M\!\big(\tfrac{\theta_a}{\pi} - \tfrac{y}{M}\big) + F_M\!\big(-\tfrac{\theta_a}{\pi} - \tfrac{y}{M}\big)\big]}_{\text{d}}\end{gathered}
```

| term | what it is | lines |
|---|---|---|
| **a** | S₀ flips the sign of |0⟩ | [418–419](nanoqalgos.py#L418-L419) |
| **b** | S_χ flips the 'good' states (last qubit 1) | [420](nanoqalgos.py#L420) |
| **c** | Grover's operator for this problem | [421](nanoqalgos.py#L421) |
| **d** | phase estimation on A|0⟩ | [422](nanoqalgos.py#L422) |

`nanoqalgos.py`, lines 412–424:

```python
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
```

✅ **formula = code**: max |formula − code| = 1.1e-14 over 128 value(s) (tolerance 1e-09).

<sub>F_M is the Fejér kernel of eq 2.2; A|0⟩ splits evenly between the two eigenvectors.</sub>

<a id="eq-75"></a>
## eq 7.5 · Brassard-Høyer-Mosca-Tapp guarantee

```math
\begin{gathered}P\Big(|\tilde a - a| \le \tfrac{2\pi\sqrt{a(1-a)}}{M} + \tfrac{\pi^2}{M^2}\Big) \ge \tfrac{8}{\pi^2} \\ \underbrace{\tilde a_y = \sin^2(\pi y/M)}_{\text{a}}\end{gathered}
```

| term | what it is | lines |
|---|---|---|
| **a** | turn the measured y into an estimate | [423](nanoqalgos.py#L423) |

`nanoqalgos.py`, lines 412–424:

```python
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
```

✅ **code ≥ bound**: bound 0.810569, code 0.82281.

<sub>Code value: the worst case over M = 8 … 512.</sub>

<a id="eq-76"></a>
## eq 7.6 · Monte Carlo error is binomial

```math
P\big(\hat a = \tfrac{k}{M}\big) = \underbrace{\binom{M}{k}}_{\text{a}}\underbrace{a^{k}(1-a)^{M-k}}_{\text{b}}
```

| term | what it is | lines |
|---|---|---|
| **a** | binomial coefficient, via log-gamma | [438–439](nanoqalgos.py#L438-L439) |
| **b** | probability of k successes | [440](nanoqalgos.py#L440) |

`nanoqalgos.py`, lines 434–441:

```python
def monte_carlo_error(a, samples, confidence=0.8):
    """The same quantity for classical Monte Carlo: k ~ Binomial(samples, a), â = k/samples.
    Exact binomial pmf (via log-gamma), no normal approximation."""
    k = np.arange(samples + 1)
    log_choose = np.array([math.lgamma(samples + 1) - math.lgamma(i + 1)  # [eq 7.6a]
                           - math.lgamma(samples - i + 1) for i in k])  # [eq 7.6a]
    pmf = np.exp(log_choose + k * np.log(a) + (samples - k) * np.log1p(-a))  # [eq 7.6b]
    return error_at_confidence(k / samples, pmf, a, confidence)
```

✅ **formula = code**: max |formula − code| = 0.0e+00 over 1 value(s) (tolerance 1e-09).

<sub>Formula side: exact integers with math.comb, then the 80% error by brute force.</sub>
