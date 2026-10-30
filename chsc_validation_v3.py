# -*- coding: utf-8 -*-
# =============================================================================
#  Validation ("正当化コード") for chsc_switch_v3.py (Pauli-proxy / Stim engine; engine identical to v2)
#   The v2 engine replaces the exact quantum state by a Pauli-stabilizer "proxy" state plus an error frame
#   (Takada-Bartlett-Williamson).  The claim is that every measurement outcome distribution and the final logical
#   state class are EXACTLY those of the true non-Clifford protocol.  This is checked shot by shot against two
#   independent exact references, driven with identical inputs (noise, decoders, random outcome bits):
#     (A) brute force : dense state vector, every operator applied literally (v1 validation code)       d = 2
#     (B) v1 engine   : Z4 twisted-frame engine of v1 (itself validated against (A) in v1)               d = 3, 4
#   Compared outputs: every measurement (deterministic / random, and the outcome), every Z check, the X frame used by
#   the D4 read-out, and the final logical state class (I, X, Y, Z, S, collapsed, flux_D4).
#   (C) direct brute-force check that U = W V is the logical T^dagger of the D4 code.
#   (D) production path (lockstep=False: batched MPP + sign bits, Stim's own random outcomes) replayed into the
#       brute force: every outcome produced by the production engine must be possible (deterministic ones equal).
#   assert_validation() turns all of this into a pass/fail check.
#  Requires chsc_switch_v3.py (or the main notebook cells) to be loaded first; (B) requires numba.
# =============================================================================

# %% [Cell V1] Reference 1: brute-force state vector (from v1, unchanged) + classify (from v1)
import time

import numpy as np

try:
    from chsc_switch_v3 import *  # noqa: F401,F403
except ImportError:  # inside the notebook everything is already defined
    pass

VCFG = dict(
    bf_d=2,                    # brute force: d=2 (k=12 -> 4096 amplitudes)
    tf_d=[3, 4],               # v1 twisted-frame engine (exact, polynomial): larger d
    cases=["in_only", "in_out", "in_T_out"],
    boundaries=["ideal", "noisy"],
    p_values=[0.02, 0.05, 0.1, 0.15],
    bf_shots_per_point=150,
    replay_shots_per_point=50,
    tf_shots_per_point=100,
    seed=1017,
    tol=1e-9,
)


def _parity64(v: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return (np.bitwise_count(v) & 1).astype(np.uint8)
    v = v.copy()
    out = np.zeros(v.shape, np.uint8)
    while np.any(v):
        out ^= (v & 1).astype(np.uint8)
        v >>= 1
    return out


class BFBackend:
    """Dense state vector over the configurations a = M0 t + x_cum  (t in GF(2)^k), operators applied literally."""

    name = "brute_force"

    def __init__(self, code, tol: float = 1e-9):
        self.code = code
        self.k = code.k
        if self.k > 30:
            raise ValueError("brute force limited to k <= 30")
        self.N = 1 << self.k
        self.idx = np.arange(self.N, dtype=np.int64)
        self.tol = tol
        self.masks = np.zeros(code.nq, np.int64)
        for q in range(code.nq):
            for j in np.nonzero(code.M0[q])[0]:
                self.masks[q] |= np.int64(1) << np.int64(j)
        P = code.M0plus.astype(np.int64)
        self.shift_int = []
        for g in code.gens:
            s = np.zeros(code.nq, np.uint8)
            s[list(g.supp)] = 1
            st = (P @ s) % 2
            if not np.array_equal((code.M0.astype(np.int64) @ st) % 2, s):
                raise AssertionError("generator support not in C0")
            self.shift_int.append(int(sum(1 << int(j) for j in np.nonzero(st)[0])))
        # V in coordinates: V M0 = M0 A
        A = (P @ code.V_rows(code.M0).astype(np.int64)) % 2
        if not np.array_equal((code.M0.astype(np.int64) @ A) % 2, code.V_rows(code.M0)):
            raise AssertionError("V does not preserve C0")
        self.Acols = [int(sum(1 << int(i) for i in np.nonzero(A[:, j])[0])) for j in range(self.k)]
        self.init_cols = [col for col, gi in enumerate(code.basis_gens) if code.gens[gi].kind == "g"]
        self.reset()

    def reset(self):
        self.x = np.zeros(self.code.nq, np.uint8)
        ok = np.ones(self.N, bool)
        for c in self.init_cols:
            ok &= ((self.idx >> c) & 1) == 0
        self.psi = ok.astype(np.complex128)
        self.psi /= np.linalg.norm(self.psi)
        self.after_U = False

    def a(self, q: int) -> np.ndarray:
        return _parity64(self.idx & self.masks[q]) ^ self.x[q]

    def _support(self):
        pr = np.abs(self.psi) ** 2
        return pr > self.tol * pr.max()

    # ---- Pauli / unitaries
    def apply_x(self, qs):
        for q in qs:
            self.x[int(q)] ^= 1

    def apply_z(self, qs):
        for q in qs:
            self.psi *= 1.0 - 2.0 * self.a(int(q))

    def _dress_sign(self, pairs):
        s = np.zeros(self.N, np.uint8)
        for i, j in pairs:
            s ^= self.a(i) & self.a(j)
        return 1.0 - 2.0 * s

    def apply_gen_unitaries(self, gen_idx):
        for gi in gen_idx:
            g = self.code.gens[int(gi)]
            for q in g.supp:
                self.x[q] ^= 1
            self.psi *= self._dress_sign(g.pairs)

    def apply_U(self):
        code = self.code
        # V: CNOT(r->g) CNOT(b->g) on every edge with a green qubit (literal action on configurations)
        self.x = code.apply_V_vec(self.x)
        new = np.zeros(self.N, np.int64)
        for j, col in enumerate(self.Acols):
            new ^= ((self.idx >> j) & 1) * np.int64(col)
        psi = np.zeros_like(self.psi)
        psi[new] = self.psi
        self.psi = psi
        # W: CS / CS^dagger on plaquettes, T^dagger on L_rb red qubits (literal phases)
        e = np.zeros(self.N, np.int64)
        for qa, qb, s in code.cs_terms:
            e += 2 * s * (self.a(qa).astype(np.int64) & self.a(qb).astype(np.int64))
        for q in code.tdag_qubits:
            e -= self.a(q).astype(np.int64)
        self.psi = self.psi * np.exp(1j * np.pi / 4 * (e % 8))
        self.after_U = True

    # ---- measurements
    def zchecks(self, idx):
        sup = self._support()
        out = np.zeros(len(idx), np.uint8)
        for u, c in enumerate(idx):
            v = np.zeros(self.N, np.uint8)
            for q in self.code.zchecks[int(c)][2]:
                v ^= self.a(q)
            vals = np.unique(v[sup])
            if vals.size != 1:
                raise AssertionError("Z check not deterministic (state left the flat subspace?)")
            out[u] = vals[0]
        return out

    def _measure_op(self, sh: int, sign: np.ndarray, rbit: int):
        Opsi = sign * self.psi[self.idx ^ sh]
        nrm = np.vdot(self.psi, self.psi).real
        p0 = 0.5 * (1.0 + np.vdot(self.psi, Opsi).real / nrm)
        if abs(p0 - 1) < 1e-7:
            m, kind = 0, "det"
        elif abs(p0) < 1e-7:
            m, kind = 1, "det"
        elif abs(p0 - 0.5) < 1e-7:
            m, kind = int(rbit), "rand"
        else:
            raise AssertionError(f"outcome probability {p0} not in {{0, 1/2, 1}}")
        self.psi = 0.5 * (self.psi + (1 - 2 * m) * Opsi)
        self.psi /= np.linalg.norm(self.psi)
        return m, kind, p0

    def measure_gens(self, gen_idx, rbits, bare: bool, kinds: list = None):
        out = np.zeros(len(gen_idx), np.uint8)
        for u, gi in enumerate(gen_idx):
            g = self.code.gens[int(gi)]
            sign = np.ones(self.N) if bare else self._dress_sign(g.pairs)
            m, kind, _ = self._measure_op(self.shift_int[int(gi)], sign, int(rbits[u]))
            out[u] = m
            if kinds is not None:
                kinds.append(kind)
        return out

    def measure_z_single(self, qs, rbits, kinds: list = None):
        out = np.zeros(len(qs), np.uint8)
        for u, q in enumerate(qs):
            aq = self.a(int(q))
            pr = np.abs(self.psi) ** 2
            p0 = pr[aq == 0].sum() / pr.sum()
            if abs(p0 - 1) < 1e-7:
                m, kind = 0, "det"
            elif abs(p0) < 1e-7:
                m, kind = 1, "det"
            elif abs(p0 - 0.5) < 1e-7:
                m, kind = int(rbits[u]), "rand"
            else:
                raise AssertionError(f"Z outcome probability {p0}")
            self.psi = np.where(aq == m, self.psi, 0)
            self.psi /= np.linalg.norm(self.psi)
            out[u] = m
            if kinds is not None:
                kinds.append(kind)
        return out

    def x_frame(self):
        return self.x.copy()

    # ---- evaluation
    def evaluate(self, basis: str):
        code = self.code
        sup = self._support()
        if basis == "SC":
            zb = np.zeros(self.N, np.uint8)
            for q in np.nonzero(code.zbar)[0]:
                zb ^= self.a(int(q))
            for q in code.g_qubits:  # green must be in a product state
                vals = np.unique(self.a(int(q))[sup])
                if vals.size != 1:
                    raise AssertionError("green qubits not pinned")
            present, exps = [False, False], [0, 0]
            for s in (0, 1):
                sel = sup & (zb == s)
                if not np.any(sel):
                    continue
                amps = self.psi[sel]
                if np.max(np.abs(amps - amps[0])) > 1e-6 * np.abs(amps[0]):
                    raise AssertionError("amplitudes not uniform within a sector (not a code state)")
                present[s] = True
                exps[s] = int(np.round(np.angle(amps[0]) / (np.pi / 4))) % 8
                if abs(np.angle(amps[0]) / (np.pi / 4) - np.round(np.angle(amps[0]) / (np.pi / 4))) > 1e-6:
                    raise AssertionError("non-Z8 phase")
            xflip = int(code.zbar @ self.x) % 2
            return dict(present=present, exp=exps, xflip=xflip, amps=None)
        if np.any(self.x):
            raise AssertionError("D4 evaluation requires x = 0")
        present, exps = [False, False], [0, 0]
        for h in (0, 1):
            v = np.zeros(self.N, np.complex128)
            v[(1 << code.kh) if h else 0] = 1.0
            for gi, g in enumerate(code.gens):
                sign = np.ones(self.N)
                s = np.zeros(self.N, np.uint8)
                for i, j in g.pairs:
                    s ^= (_parity64(self.idx & self.masks[i]) & _parity64(self.idx & self.masks[j]))
                sign = 1.0 - 2.0 * s
                v = 0.5 * (v + sign * v[self.idx ^ self.shift_int[gi]])
            v /= np.linalg.norm(v)
            beta = np.vdot(v, self.psi)
            if abs(beta) > 1e-6:
                present[h] = True
                ang = np.angle(beta) / (np.pi / 4)
                if abs(ang - np.round(ang)) > 1e-6:
                    raise AssertionError("non-Z8 D4 phase")
                exps[h] = int(np.round(ang)) % 8
        return dict(present=present, exp=exps, xflip=0)


def classify(ev: dict, target: int) -> Tuple[str, str]:
    """Logical classification of the final state, ideal = |0> + omega^target |1>  (omega = e^{i pi/4}).

    state class (primary): final state  ~  P * ideal  with P in {I, X, Y, Z}  (decided from the state only);
        "S" = equatorial state not of this form (|+-i> for the |+> target), "collapsed" = only one logical
        Z sector survived (logical Z was measured).   P_L = 1 - Pr(state class = I).
        For the T case every equatorial outcome is P*ideal for exactly one Pauli P; for the |+> target X ~ I
        and Y ~ Z as states.
    history class (diagnostic): E = X^f * diag(1, omega^delta), f = homology of the accumulated Pauli-X frame,
        delta in {0,2,4,6} from the sector phases (e.g. "XS": m-type logical error before U; note XS T^dag|+>
        ~ T^dag|+> so it is NOT a state deviation)."""
    pres = ev["present"]
    if not (pres[0] and pres[1]):
        return "collapsed", "collapsed"
    r = (ev["exp"][1] - ev["exp"][0]) % 8
    xf = int(ev.get("xflip", 0))
    delta = (r - target) % 8 if xf == 0 else (-r - target) % 8
    if delta % 2:
        raise AssertionError(f"odd logical phase delta={delta}: inconsistent state")
    diag = {0: "I", 4: "Z", 2: "S", 6: "Sdg"}[delta]
    hist = diag if xf == 0 else {"I": "X", "Z": "Y", "S": "XS", "Sdg": "XSdg"}[diag]
    st = {target % 8: "I", (target + 4) % 8: "Z", (-target) % 8: "X", (4 - target) % 8: "Y"}
    # for an even target X~I and Y~Z coincide: the dict keeps the first (I / Z) only if inserted first
    if target % 2 == 0:
        st = {target % 8: "I", (target + 4) % 8: "Z"}
    state = st.get(r, "S")
    return state, hist


# %% [Cell V2] Reference 2: the v1 Z4 twisted-frame engine (copied unchanged from v1 Cells 5-6; needs numba)
from numba import njit


def pack_bits(bits: np.ndarray) -> np.ndarray:
    """bits (..., k) in {0,1} -> (..., ceil(k/64)) uint64, bit j of the vector = bit (j % 64) of word j // 64."""
    bits = np.asarray(bits, dtype=np.uint8) & 1
    k = bits.shape[-1]
    nw = max(1, (k + 63) // 64)
    pad = np.zeros(bits.shape[:-1] + (nw * 64,), np.uint8)
    pad[..., :k] = bits
    by = np.packbits(pad, axis=-1, bitorder="little")
    return np.ascontiguousarray(by).view("<u8").astype(np.uint64).reshape(bits.shape[:-1] + (nw,))


def unpack_bits(packed: np.ndarray, k: int) -> np.ndarray:
    packed = np.ascontiguousarray(np.asarray(packed, dtype=np.uint64)).astype("<u8")
    by = packed.view(np.uint8).reshape(packed.shape[:-1] + (packed.shape[-1] * 8,))
    return np.unpackbits(by, axis=-1, bitorder="little")[..., :k].copy()


# ---- (v1 Cell 5) Z4 twisted-frame numba kernels ----
#
#  State (up to normalisation and a global phase):
#     psi(a) = omega^{tpow * t_h} * (-1)^{phi0(b) + Q(t)} * i^{l . t} * 1[t in B],     a = b + x,  b = M t
#  x: Pauli-X frame;  B: affine subspace {G t = rhs};  Q(t) = sum_{i<j} W_ij t_i t_j + d . t  (mod 2);
#  l . t: integer sum (mod 4) -> S-type phases created by U acting on X errors;  t_h: logical coordinate
#  (coefficient of eta), never shifted by any X-type operator, so omega^{tpow t_h} (= logical T^dagger phase)
#  is a passive factor that never influences probabilities or updates.
#  v7 kernels are reused (case (i)/(ii) of a dressed measurement) and extended by the l-vector.

@njit(inline="always")
def _popcount64(v):
    v = v - ((v >> np.uint64(1)) & np.uint64(0x5555555555555555))
    v = (v & np.uint64(0x3333333333333333)) + ((v >> np.uint64(2)) & np.uint64(0x3333333333333333))
    v = (v + (v >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return int((v * np.uint64(0x0101010101010101)) >> np.uint64(56))


@njit(inline="always")
def _getbit(row, j):
    return int((row[j >> 6] >> np.uint64(j & 63)) & np.uint64(1))


@njit
def _dot_parity(a, b):
    s = 0
    for w in range(a.shape[0]):
        s += _popcount64(a[w] & b[w])
    return s & 1


@njit
def _is_zero(a):
    for w in range(a.shape[0]):
        if a[w] != np.uint64(0):
            return False
    return True


@njit
def _first_bit(a):
    for w in range(a.shape[0]):
        v = a[w]
        if v != np.uint64(0):
            b = 0
            while ((v >> np.uint64(b)) & np.uint64(1)) == np.uint64(0):
                b += 1
            return w * 64 + b
    return -1


@njit
def _rank2_update(W, a, b, k):
    """W += a b^T + b a^T (diagonal stays 0)."""
    for j in range(k):
        if _getbit(a, j):
            for w in range(W.shape[1]):
                W[j, w] ^= b[w]
        if _getbit(b, j):
            for w in range(W.shape[1]):
                W[j, w] ^= a[w]


@njit
def tf_qval(W, d, t, k):
    s = 0
    for j in range(k):
        if _getbit(t, j):
            for w in range(W.shape[1]):
                s += _popcount64(W[j, w] & t[w])
    s = s // 2
    for w in range(d.shape[0]):
        s += _popcount64(d[w] & t[w])
    return s & 1


@njit
def tf_add_product(W, d, a, a0, b, b0, k):
    """Q += (a.t + a0)(b.t + b0)."""
    _rank2_update(W, a, b, k)
    for w in range(d.shape[0]):
        d[w] ^= a[w] & b[w]
        if b0:
            d[w] ^= a[w]
        if a0:
            d[w] ^= b[w]


@njit
def tf_mul_i_lift(W, d, l, Lam, c0, power, k):
    """psi *= i^{power * lift(c0 + Lam.t)}   (power = 1 or 3, lift: {0,1} -> {0,1} in Z)."""
    nw = d.shape[0]
    for w in range(nw):
        d[w] ^= l[w] & Lam[w]
        l[w] ^= Lam[w]
    for j in range(k):
        if _getbit(Lam, j):
            for w in range(nw):
                W[j, w] ^= Lam[w]
            W[j, j >> 6] ^= np.uint64(1) << np.uint64(j & 63)
    flip = c0
    if power == 3:
        flip ^= 1
    if flip:
        for w in range(nw):
            d[w] ^= Lam[w]


@njit
def tf_apply_cz(Mp, x, W, d, i, j, k):
    """physical CZ(i,j) = (-1)^{a_i a_j}, a = M t + x."""
    tf_add_product(W, d, Mp[i], x[i], Mp[j], x[j], k)


@njit
def tf_apply_cz_b(Mp, W, d, i, j, k):
    """(-1)^{b_i b_j} on the flat part b = M t."""
    tf_add_product(W, d, Mp[i], 0, Mp[j], 0, k)


@njit
def tf_apply_z_list(Mp, d, qs):
    for u in range(qs.shape[0]):
        q = qs[u]
        for w in range(d.shape[0]):
            d[w] ^= Mp[q, w]


@njit
def _reduce_row(v, G, rhs, piv, nrows):
    """Reduce a linear form v against the (fully reduced) constraint rows of B; returns (rest, acc)."""
    r = v.copy()
    acc = 0
    for i in range(nrows):
        if _getbit(r, piv[i]):
            for w in range(r.shape[0]):
                r[w] ^= G[i, w]
            acc ^= rhs[i]
    return r, acc


@njit
def _add_row(G, rhs, piv, nrows_arr, lr, val):
    """Add constraint lr . t = val (lr already reduced, nonzero)."""
    nr = nrows_arr[0]
    p = _first_bit(lr)
    for r in range(nr):
        if _getbit(G[r], p):
            for w in range(G.shape[1]):
                G[r, w] ^= lr[w]
            rhs[r] ^= val
    for w in range(G.shape[1]):
        G[nr, w] = lr[w]
    rhs[nr] = val
    piv[nr] = p
    nrows_arr[0] = nr + 1


@njit
def tf_measure_xop(Mp, x, W, d, l, G, rhs, piv, nrows_arr, stats, k,
                   sh, shc, nshc, cp, ncp, qp, nqp, rbit):
    """Ideal projective measurement of O = D X^s (X-type, Hermitian, O^2 = 1).

    sh/shc: shift in t-coordinates (packed / list of coordinates);
    cp: 'common' pairs = CZ dressing that the measured operator carries AND whose phi0 identity is used
        (dressed generator, v7 formula);  qp: pairs of the generator whose phi0 identity is used but which the
        measured operator does NOT carry (bare X star measured in the surface code): needs one end pinned on B.
    Returns outcome m (eigenvalue (-1)^m), or -1 on an internal error (non-affine ratio).
    stats: [deterministic, random case (i), random case (ii), random (i) with i-phase, errors, case (ii) with i-phase]
    """
    nw = W.shape[1]
    lam = np.zeros(nw, np.uint64)
    for u in range(nshc):
        j = shc[u]
        for w in range(nw):
            lam[w] ^= W[j, w]
    kap = tf_qval(W, d, sh, k)
    c = 0
    for w in range(nw):
        c += _popcount64(l[w] & sh[w])
        lam[w] ^= l[w] & sh[w]
    codd = c & 1
    kap ^= (c >> 1) & 1
    for u in range(ncp):
        i = cp[u, 0]
        j = cp[u, 1]
        if x[j]:
            for w in range(nw):
                lam[w] ^= Mp[i, w]
        if x[i]:
            for w in range(nw):
                lam[w] ^= Mp[j, w]
        kap ^= (x[i] & x[j])
    nr = nrows_arr[0]
    for u in range(nqp):
        i = qp[u, 0]
        j = qp[u, 1]
        ri, ai = _reduce_row(Mp[i], G, rhs, piv, nr)
        if _is_zero(ri):
            if ai:
                for w in range(nw):
                    lam[w] ^= Mp[j, w]
        else:
            rj, aj = _reduce_row(Mp[j], G, rhs, piv, nr)
            if not _is_zero(rj):
                stats[4] += 1
                return -1
            if aj:
                for w in range(nw):
                    lam[w] ^= Mp[i, w]
    r0 = -1
    for r in range(nr):
        if _dot_parity(G[r], sh):
            r0 = r
            break
    if r0 < 0:
        if codd == 0:
            lr, kr = _reduce_row(lam, G, rhs, piv, nr)
            kr ^= kap
            if _is_zero(lr):
                stats[0] += 1
                return kr
            m = rbit
            _add_row(G, rhs, piv, nrows_arr, lr, m ^ kr)
            stats[1] += 1
            return m
        # ratio = i * (-1)^{lam.t + kap}: outcome uniform, B unchanged, phases rotate by omega^{+-1}
        m = rbit
        tf_mul_i_lift(W, d, l, lam, m ^ kap, 3, k)
        stats[3] += 1
        return m
    # case (ii): s not in dir(B)
    m = rbit
    g = G[r0].copy()
    cc = rhs[r0]
    tf_add_product(W, d, g, cc, lam, (kap ^ m) & 1, k)
    if codd:
        tf_mul_i_lift(W, d, l, g, cc, 1, k)
        stats[5] += 1
    for r in range(nr):
        if r != r0 and _dot_parity(G[r], sh):
            for w in range(nw):
                G[r, w] ^= g[w]
            rhs[r] ^= cc
    last = nr - 1
    if r0 != last:
        for w in range(nw):
            G[r0, w] = G[last, w]
        rhs[r0] = rhs[last]
        piv[r0] = piv[last]
    nrows_arr[0] = last
    stats[2] += 1
    return m


@njit
def tf_measure_xops(Mp, x, W, d, l, G, rhs, piv, nrows_arr, stats, k,
                    SH, SHC, NSHC, CP, NCP, QP, NQP, ops, rbits, out):
    for u in range(ops.shape[0]):
        o = ops[u]
        m = tf_measure_xop(Mp, x, W, d, l, G, rhs, piv, nrows_arr, stats, k,
                           SH[o], SHC[o], NSHC[o], CP[o], NCP[o], QP[o], NQP[o], rbits[u])
        if m < 0:
            return -1
        out[u] = m
    return 0


@njit
def tf_measure_z_single(Mp, x, G, rhs, piv, nrows_arr, stats, qs, rbits, out):
    for u in range(qs.shape[0]):
        q = qs[u]
        lr, acc = _reduce_row(Mp[q], G, rhs, piv, nrows_arr[0])
        if _is_zero(lr):
            out[u] = acc ^ x[q]
            stats[0] += 1
        else:
            m = rbits[u]
            _add_row(G, rhs, piv, nrows_arr, lr, m ^ x[q] ^ acc)
            out[u] = m
            stats[1] += 1


@njit
def tf_zchecks(x, qmat, qlen, out):
    for c in range(qmat.shape[0]):
        v = 0
        for u in range(qlen[c]):
            v ^= x[qmat[c, u]]
        out[c] = v


@njit
def tf_apply_gen_unitary(Mp, x, W, d, supp, nsupp, pairs, npairs, k):
    """Apply a dressed generator S = X^supp prod CZ(pairs) as a unitary (pairs disjoint from supp)."""
    for u in range(nsupp):
        x[supp[u]] ^= 1
    for u in range(npairs):
        tf_apply_cz(Mp, x, W, d, pairs[u, 0], pairs[u, 1], k)


# ---- (v1 Cell 6) twisted-frame backend ----
class OpTable:
    """Measurement table for a list of X-type operators (numba-friendly arrays)."""

    def __init__(self, k: int, shifts: List[np.ndarray], common: List[List[Tuple[int, int]]],
                 quad: List[List[Tuple[int, int]]]):
        nops = len(shifts)
        self.nops = nops
        self.SH = pack_bits(np.array(shifts, np.uint8).reshape(nops, k))
        maxc = max(1, max(len(np.nonzero(s)[0]) for s in shifts))
        self.SHC = np.zeros((nops, maxc), np.int32)
        self.NSHC = np.zeros(nops, np.int32)
        for o, s in enumerate(shifts):
            nz = np.nonzero(s)[0]
            self.SHC[o, :len(nz)] = nz
            self.NSHC[o] = len(nz)
        mp = max(1, max([len(c) for c in common] + [len(q) for q in quad]))
        self.CP = np.zeros((nops, mp, 2), np.int32)
        self.NCP = np.zeros(nops, np.int32)
        self.QP = np.zeros((nops, mp, 2), np.int32)
        self.NQP = np.zeros(nops, np.int32)
        for o in range(nops):
            for u, pr in enumerate(common[o]):
                self.CP[o, u] = pr
            self.NCP[o] = len(common[o])
            for u, pr in enumerate(quad[o]):
                self.QP[o, u] = pr
            self.NQP[o] = len(quad[o])


class TFTables:
    """Per-geometry tables shared by all shots."""

    def __init__(self, code: TriangleCode):
        self.code = code
        k = code.k
        P = code.M0plus.astype(np.int64)
        # shifts of every generator in original coordinates and after U (M' = V M0  =>  s_t = M0^+ V s)
        sh0, sh1 = [], []
        for g in code.gens:
            s = np.zeros(code.nq, np.uint8)
            s[list(g.supp)] = 1
            sh0.append(((P @ s) % 2).astype(np.uint8))
            sh1.append(((P @ code.apply_V_vec(s)) % 2).astype(np.uint8))
        pairs = [list(g.pairs) for g in code.gens]
        empty = [[] for _ in code.gens]
        self.dressed0 = OpTable(k, sh0, pairs, empty)      # dressed generators, before U
        self.bare0 = OpTable(k, sh0, empty, pairs)         # bare X stars (surface code), no U
        self.bare1 = OpTable(k, sh1, empty, pairs)         # bare X stars (surface code), after U
        nz = len(code.zchecks)
        ml = max(len(z[2]) for z in code.zchecks)
        self.zq = np.zeros((nz, ml), np.int32)
        self.zl = np.zeros(nz, np.int32)
        for c, (_, _, qs) in enumerate(code.zchecks):
            self.zq[c, :len(qs)] = qs
            self.zl[c] = len(qs)
        mg = max(len(g.supp) for g in code.gens)
        mp = max(1, max(len(g.pairs) for g in code.gens))
        self.gsupp = np.zeros((code.ngen, mg), np.int32)
        self.gnsupp = np.zeros(code.ngen, np.int32)
        self.gpairs = np.zeros((code.ngen, mp, 2), np.int32)
        self.gnpairs = np.zeros(code.ngen, np.int32)
        for gi, g in enumerate(code.gens):
            self.gsupp[gi, :len(g.supp)] = g.supp
            self.gnsupp[gi] = len(g.supp)
            for u, pr in enumerate(g.pairs):
                self.gpairs[gi, u] = pr
            self.gnpairs[gi] = len(g.pairs)
        self.M0p = pack_bits(np.vstack([code.M0, np.zeros((1, k), np.uint8)]))  # + dummy row
        self.M1p = pack_bits(np.vstack([code.V_rows(code.M0), np.zeros((1, k), np.uint8)]))
        self.M1dense = code.V_rows(code.M0)
        self.kappa = logical_U_phase(code)
        # initial B: all green generator coordinates = 0  (green qubits in |0>)
        self.init_rows = []
        for col, gi in enumerate(code.basis_gens):
            if code.gens[gi].kind == "g":
                self.init_rows.append(col)
        nw = max(1, (k + 63) // 64)
        self.G_init = np.zeros((k + 1, nw), np.uint64)
        self.rhs_init = np.zeros(k + 1, np.uint8)
        self.piv_init = np.zeros(k + 1, np.int32)
        for r_, col in enumerate(self.init_rows):  # unit rows e_col: already fully reduced
            v = np.zeros(k, np.uint8)
            v[col] = 1
            self.G_init[r_] = pack_bits(v)
            self.piv_init[r_] = col
        self.nrows_init = len(self.init_rows)


class TFBackend:
    """Z4 twisted-frame backend (exact for this protocol under Pauli noise, see report Sec. 3)."""

    name = "twisted_frame"

    def __init__(self, tab: TFTables):
        self.tab = tab
        code = tab.code
        self.code = code
        k = code.k
        self.k = k
        self.nw = max(1, (k + 63) // 64)
        self.x = np.zeros(code.nq + 1, np.uint8)
        self.W = np.zeros((k, self.nw), np.uint64)
        self.d = np.zeros(self.nw, np.uint64)
        self.l = np.zeros(self.nw, np.uint64)
        self.G = np.zeros((k + 1, self.nw), np.uint64)
        self.rhs = np.zeros(k + 1, np.uint8)
        self.piv = np.zeros(k + 1, np.int32)
        self.nrows = np.zeros(1, np.int32)
        self.stats = np.zeros(6, np.int64)
        self.reset()

    def reset(self):
        self.x[:] = 0
        self.W[:] = 0
        self.d[:] = 0
        self.l[:] = 0
        self.G[:] = 0
        self.rhs[:] = 0
        self.nrows[0] = 0
        self.tpow = 0
        self.after_U = False
        self.Mp = self.tab.M0p
        self.G[:] = self.tab.G_init
        self.rhs[:] = self.tab.rhs_init
        self.piv[:] = self.tab.piv_init
        self.nrows[0] = self.tab.nrows_init

    # ---- Pauli errors / corrections
    def apply_x(self, qs):
        if len(qs):
            self.x[np.asarray(qs, np.int64)] ^= 1

    def apply_z(self, qs):
        if len(qs):
            tf_apply_z_list(self.Mp, self.d, np.asarray(qs, np.int32))

    # ---- measurements
    def zchecks(self, idx: np.ndarray) -> np.ndarray:
        out = np.zeros(len(idx), np.uint8)
        tf_zchecks(self.x, self.tab.zq[idx], self.tab.zl[idx], out)
        return out

    def measure_gens(self, gen_idx: np.ndarray, rbits: np.ndarray, bare: bool) -> np.ndarray:
        if bare:
            tb = self.tab.bare1 if self.after_U else self.tab.bare0
        else:
            if self.after_U:
                raise RuntimeError("dressed measurements after U are not used by this protocol")
            tb = self.tab.dressed0
        out = np.zeros(len(gen_idx), np.uint8)
        err = tf_measure_xops(self.Mp, self.x, self.W, self.d, self.l, self.G, self.rhs, self.piv, self.nrows,
                              self.stats, self.k, tb.SH, tb.SHC, tb.NSHC, tb.CP, tb.NCP, tb.QP, tb.NQP,
                              np.asarray(gen_idx, np.int32), np.asarray(rbits, np.uint8), out)
        if err < 0:
            raise RuntimeError("non-affine measurement ratio (bare star with unpinned dressing qubits)")
        return out

    def measure_z_single(self, qs: np.ndarray, rbits: np.ndarray) -> np.ndarray:
        out = np.zeros(len(qs), np.uint8)
        tf_measure_z_single(self.Mp, self.x, self.G, self.rhs, self.piv, self.nrows, self.stats,
                            np.asarray(qs, np.int32), np.asarray(rbits, np.uint8), out)
        return out

    def apply_gen_unitaries(self, gen_idx):
        t = self.tab
        for gi in gen_idx:
            tf_apply_gen_unitary(self.Mp, self.x, self.W, self.d, t.gsupp[gi], t.gnsupp[gi],
                                 t.gpairs[gi], t.gnpairs[gi], self.k)

    # ---- the ideal logical gate U = W V
    def apply_U(self):
        code = self.code
        k = self.k
        xn = code.apply_V_vec(self.x[:code.nq])
        self.x[:code.nq] = xn
        self.Mp = self.tab.M1p
        self.after_U = True
        Mp = self.Mp
        for qa, qb, s in code.cs_terms:  # W(b'+x')/W(b')  for CS^{s}
            xa, xb = int(xn[qa]), int(xn[qb])
            if xa and not xb:
                tf_mul_i_lift(self.W, self.d, self.l, Mp[qb], 0, 1 if s > 0 else 3, k)
                tf_apply_cz_b(Mp, self.W, self.d, qa, qb, k)
            elif xb and not xa:
                tf_mul_i_lift(self.W, self.d, self.l, Mp[qa], 0, 1 if s > 0 else 3, k)
                tf_apply_cz_b(Mp, self.W, self.d, qa, qb, k)
            elif xa and xb:
                tf_mul_i_lift(self.W, self.d, self.l, Mp[qa], 0, 3 if s > 0 else 1, k)
                tf_mul_i_lift(self.W, self.d, self.l, Mp[qb], 0, 3 if s > 0 else 1, k)
        for q in code.tdag_qubits:  # T^dagger on a flipped qubit = S on b (up to a global phase)
            if xn[q]:
                tf_mul_i_lift(self.W, self.d, self.l, Mp[q], 0, 1, k)
        self.tpow = (self.tpow + self.tab.kappa) % 8  # c_1/c_0 = omega^kappa (passive logical phase)

    # ---- evaluation
    def _dense(self):
        k = self.k
        nr = int(self.nrows[0])
        G = unpack_bits(self.G[:nr], k)
        rhs = self.rhs[:nr].copy()
        piv = self.piv[:nr].copy()
        W = unpack_bits(self.W, k)
        d = unpack_bits(self.d, k)
        l = unpack_bits(self.l, k)
        return G, rhs, piv, W, d, l

    def _rep(self, G, rhs, piv, h):
        """A point t of B with t_h = h, or None."""
        k = self.k
        kh = self.code.kh
        t = np.zeros(k, np.uint8)
        pivset = {int(p): i for i, p in enumerate(piv)}
        if kh in pivset:
            r = pivset[kh]
            free = [c for c in np.nonzero(G[r])[0] if c != kh]
            if not free:
                return t if int(rhs[r]) == h else None
            t[free[0]] = (int(rhs[r]) ^ h) & 1
        else:
            t[kh] = h
        for i, p in enumerate(piv):
            row = G[i].copy()
            row[p] = 0
            t[p] = (int(rhs[i]) + int(row @ t)) % 2
        assert np.all((G.astype(np.int64) @ t) % 2 == rhs) and t[kh] == h
        return t

    def _phase_exp(self, W, d, l, t, with_phi0: bool, Mdense) -> int:
        q = (int(t @ W.astype(np.int64) @ t) // 2 + int(d @ t)) % 2
        lt = int(l.astype(np.int64) @ t) % 4
        e = (self.tpow * int(t[self.code.kh]) + 4 * q + 2 * lt) % 8
        if with_phi0:
            b = (Mdense.astype(np.int64) @ t) % 2
            e = (e + 4 * phi0_walk(self.code, b)) % 8
        return e

    def evaluate(self, basis: str):
        """Logical amplitudes as Z8 exponents.  basis="SC": physical surface-code sectors (uniform basis);
        "D4": D4 basis sum_b (-1)^phi0(b)|b> (requires x = 0).  Returns dict(present=[bool,bool], exp=[e0,e1])."""
        code = self.code
        G, rhs, piv, W, d, l = self._dense()
        Mdense = self.tab.M1dense if self.after_U else code.M0
        present = [False, False]
        exps = [0, 0]
        xflip = int(code.zbar @ self.x[:code.nq]) % 2 if basis == "SC" else 0
        for h in (0, 1):
            t = self._rep(G, rhs, piv, h)
            if t is None:
                continue
            s = h ^ xflip
            present[s] = True
            exps[s] = self._phase_exp(W, d, l, t, basis == "SC", Mdense)
        return dict(present=present, exp=exps, xflip=xflip)

    def evaluate_check_uniform(self, basis: str, n_extra: int = 3, rng=None) -> bool:
        """Self-check: phases at several random points of the same sector agree (state is a code state)."""
        code = self.code
        G, rhs, piv, W, d, l = self._dense()
        Mdense = self.tab.M1dense if self.after_U else code.M0
        rng = rng or np.random.default_rng(0)
        k = self.k
        free = [c for c in range(k) if c not in set(int(p) for p in piv) and c != code.kh]
        for h in (0, 1):
            t0 = self._rep(G, rhs, piv, h)
            if t0 is None:
                continue
            e0 = self._phase_exp(W, d, l, t0, basis == "SC", Mdense)
            for _ in range(n_extra):
                t = t0.copy()
                for c in free:
                    t[c] = rng.integers(2)
                for i, p in enumerate(piv):
                    row = G[i].copy()
                    row[p] = 0
                    t[p] = (int(rhs[i]) + int(row @ t)) % 2
                if t[code.kh] != h:
                    continue
                if self._phase_exp(W, d, l, t, basis == "SC", Mdense) != e0:
                    return False
        return True

    def x_frame(self) -> np.ndarray:
        return self.x[:self.code.nq].copy()




# %% [Cell V3] Lock-step comparison: Pauli-proxy engine vs a reference, identical inputs, every output compared
class Mismatch(Exception):
    pass


class RefAdapter:
    """Gives a v1 backend (brute force 'bf' or twisted frame 'tf') the v2 backend interface."""

    needs_rbits = True

    def __init__(self, be, kind: str):
        self.be = be
        self.kind = kind
        self.after_U = False
        self.last_kinds: List[str] = []

    def reset(self, seed=None):
        self.be.reset()
        self.after_U = False

    def apply_x(self, qs):
        self.be.apply_x(qs)

    def apply_z(self, qs):
        self.be.apply_z(qs)

    def apply_gen_unitaries(self, g):
        self.be.apply_gen_unitaries(g)

    def apply_U(self):
        self.be.apply_U()
        self.after_U = True

    def x_frame(self):
        return self.be.x_frame()

    def zchecks(self, idx):
        return self.be.zchecks(idx)

    def _one(self, fn, *args):
        if self.kind == "bf":
            k = []
            m = fn(*args, k)[0]
            self.last_kinds.append(k[0])
        else:
            before = int(self.be.stats[0])
            m = fn(*args)[0]
            self.last_kinds.append("det" if int(self.be.stats[0]) > before else "rand")
        return m

    def measure_gens(self, gen_idx, bare, rbits):
        self.last_kinds = []
        out = np.zeros(len(gen_idx), np.uint8)
        for u, gi in enumerate(gen_idx):
            if self.kind == "bf":
                out[u] = self._one(lambda a, b, k: self.be.measure_gens(a, b, bare, k), np.array([gi]), np.array([rbits[u]]))
            else:
                out[u] = self._one(lambda a, b: self.be.measure_gens(a, b, bare), np.array([gi]), np.array([rbits[u]]))
        return out

    def result(self, h):
        return h

    def measure_green_switch_out(self, gq, rbits):
        self.last_kinds = []
        out = np.zeros(len(gq), np.uint8)
        for u, q in enumerate(gq):
            out[u] = self._one(self.be.measure_z_single, np.array([q]), np.array([rbits[u]]))
        return out

    def evaluate_class(self, basis):
        ev = self.be.evaluate(basis)
        target = 7 if (basis == "SC" and self.after_U) else 0
        return classify(ev, target)[0]


class LockstepBackend:
    needs_rbits = True

    def __init__(self, px, ref):
        self.px = px
        self.ref = ref
        self.n_compared = 0
        self.n_random = 0

    def reset(self, seed=None):
        self.px.reset(seed)
        self.ref.reset(seed)

    def apply_x(self, qs):
        self.px.apply_x(qs)
        self.ref.apply_x(qs)

    def apply_z(self, qs):
        self.px.apply_z(qs)
        self.ref.apply_z(qs)

    def apply_gen_unitaries(self, g):
        self.px.apply_gen_unitaries(g)
        self.ref.apply_gen_unitaries(g)

    def apply_U(self):
        self.px.apply_U()
        self.ref.apply_U()

    def x_frame(self):
        a, b = self.px.x_frame(), self.ref.x_frame()
        if not np.array_equal(a, b):
            raise Mismatch("X frames differ")
        return a

    def zchecks(self, idx):
        a, b = self.px.zchecks(idx), self.ref.zchecks(idx)
        if not np.array_equal(a, b):
            raise Mismatch("Z checks differ")
        self.n_compared += len(idx)
        return a

    def _compare(self, what, a, b, ka, kb):
        if not np.array_equal(a, b) or list(ka) != list(kb):
            bad = [u for u in range(len(a)) if a[u] != b[u] or ka[u] != kb[u]]
            u = bad[0]
            raise Mismatch(f"{what}: first difference at #{u}: proxy ({a[u]},{ka[u]}) reference ({b[u]},{kb[u]})")
        self.n_compared += len(a)
        self.n_random += sum(k == "rand" for k in ka)

    def measure_gens(self, gen_idx, bare, rbits):
        a = self.px.measure_gens(gen_idx, bare, rbits)
        ka = list(self.px.last_kinds)
        b = self.ref.measure_gens(gen_idx, bare, rbits)
        self._compare(f"X-type measurements (bare={bare})", a, b, ka, self.ref.last_kinds)
        return a

    def result(self, h):
        return h

    def measure_green_switch_out(self, gq, rbits):
        a = self.px.measure_green_switch_out(gq, rbits)
        ka = list(self.px.last_kinds)
        b = self.ref.measure_green_switch_out(gq, rbits)
        self._compare("Z^g measurements (switch-out)", a, b, ka, self.ref.last_kinds)
        return a

    def evaluate_class(self, basis):
        a = self.px.evaluate_class(basis)
        b = self.ref.evaluate_class(basis)
        if a != b:
            raise Mismatch(f"final logical class differs: proxy {a} reference {b}")
        self.n_compared += 1
        return a


# (C) U = W V is the logical T^dagger (brute force, no phi0 bookkeeping) -- from v1
def check_U_bruteforce(d: int = 2) -> dict:
    code, tab = get_code(d)
    bf = BFBackend(code)
    res = {}
    vecs = []
    for h in (0, 1):
        bf.reset()
        bf.x[:] = 0
        v = np.zeros(bf.N, np.complex128)
        v[(1 << code.kh) if h else 0] = 1.0
        for gi, g in enumerate(code.gens):
            s = np.zeros(bf.N, np.uint8)
            for i, j in g.pairs:
                s ^= bf.a(i) & bf.a(j)
            v = 0.5 * (v + (1.0 - 2.0 * s) * v[bf.idx ^ bf.shift_int[gi]])
        v /= np.linalg.norm(v)
        vecs.append(v)
    cs = []
    for h in (0, 1):
        bf.psi = vecs[h].copy()
        bf.x[:] = 0
        bf.apply_U()
        if np.any(bf.x):
            raise AssertionError
        c = np.vdot(vecs[h], bf.psi)
        res[f"|<{h}|U|{h}>|"] = float(abs(c))
        cs.append(c)
    ratio = cs[1] / cs[0]
    res["c1/c0 (units of pi/4)"] = float(np.angle(ratio) / (np.pi / 4))
    res["logical gate"] = "T^dagger" if abs(np.angle(ratio) / (np.pi / 4) + 1) < 1e-9 else "other"
    res["kappa used by the proxy engine (T_h)"] = int(tab.kappa)
    return res


# %% [Cell V4] Drivers: lock-step (A, B), production-path replay (D), pass/fail check
_TF_CACHE: Dict[int, object] = {}


def _reference(kind: str, code, vcfg):
    if kind == "bf":
        return RefAdapter(BFBackend(code, tol=vcfg.get("tol", 1e-9)), "bf")
    if code.d not in _TF_CACHE:
        _TF_CACHE[code.d] = TFTables(code)
    return RefAdapter(TFBackend(_TF_CACHE[code.d]), "tf")


def run_lockstep(ref: str = "bf", ds=None, vcfg: dict = None) -> list:
    vcfg = VCFG if vcfg is None else vcfg
    if ds is None:
        ds = [vcfg["bf_d"]] if ref == "bf" else list(vcfg["tf_d"])
    shots = int(vcfg["bf_shots_per_point"] if ref == "bf" else vcfg["tf_shots_per_point"])
    rows = []
    for d in ds:
        code, tab = get_code(int(d))
        for boundary in vcfg["boundaries"]:
            cfg = dict(CFG)
            cfg["switch_in_boundary"] = boundary
            for case in vcfg["cases"]:
                for p in vcfg["p_values"]:
                    runner = ShotRunner(code, p, cfg)
                    px = ProxyBackend(tab, lockstep=True)
                    ls = LockstepBackend(px, _reference(ref, code, vcfg))
                    counts = {c: 0 for c in STATE_CLASSES}
                    t0 = time.time()
                    mism = 0
                    msg = ""
                    for s in range(shots):
                        rng = np.random.default_rng([int(vcfg["seed"]), s, int(p * 1e6), len(case), int(d)])
                        try:
                            counts[runner.run(ls, case, rng)] += 1
                        except Mismatch as exc:
                            mism += 1
                            msg = str(exc)
                    rows.append(dict(reference=ref, boundary=boundary, case=case, d=int(d), p=p, shots=shots,
                                     mismatches=mism, compared_outputs=ls.n_compared, random_outcomes=ls.n_random,
                                     frac_random=ls.n_random / max(1, ls.n_compared),
                                     **{f"n_{c}": counts[c] for c in STATE_CLASSES if counts[c]},
                                     seconds=round(time.time() - t0, 1), last_mismatch=msg))
                    print(f"[{ref} {boundary:5s}] {case:8s} d={d} p={p:.3f}: mismatches={mism}  compared={ls.n_compared}"
                          f"  random={ls.n_random}  classes={ {c: k for c, k in counts.items() if k} }  ({time.time() - t0:.1f}s)")
                    if msg:
                        print("    last mismatch:", msg)
    return rows


class ReplayBackend:
    """(D): production ProxyBackend (lockstep=False) drives the protocol with Stim's own random outcomes; every
    measurement outcome is replayed into the reference (forced if random there, compared if deterministic there)."""

    needs_rbits = False

    def __init__(self, px, ref):
        self.px = px
        self.ref = ref
        self.n_compared = 0

    def reset(self, seed=None):
        self.px.reset(seed)
        self.ref.reset(seed)

    def apply_x(self, qs):
        self.px.apply_x(qs)
        self.ref.apply_x(qs)

    def apply_z(self, qs):
        self.px.apply_z(qs)
        self.ref.apply_z(qs)

    def apply_gen_unitaries(self, g):
        self.px.apply_gen_unitaries(g)
        self.ref.apply_gen_unitaries(g)

    def apply_U(self):
        self.px.apply_U()
        self.ref.apply_U()

    def x_frame(self):
        a, b = self.px.x_frame(), self.ref.x_frame()
        if not np.array_equal(a, b):
            raise Mismatch("X frames differ")
        return a

    def zchecks(self, idx):
        a, b = self.px.zchecks(idx), self.ref.zchecks(idx)
        if not np.array_equal(a, b):
            raise Mismatch("Z checks differ")
        self.n_compared += len(idx)
        return a

    def _replay(self, what, a, fn):
        b = fn(a)
        if not np.array_equal(a, b):
            u = int(np.nonzero(a != b)[0][0])
            raise Mismatch(f"{what}: production outcome #{u}={a[u]} is impossible (reference deterministic {b[u]})")
        self.n_compared += len(a)
        return a

    def measure_gens(self, gen_idx, bare, rbits=None):
        a = self.px.result(self.px.measure_gens(gen_idx, bare))   # batched MPP path + sign bits
        return self._replay(f"X-type measurements (bare={bare})", a,
                            lambda r: self.ref.measure_gens(gen_idx, bare, r))

    def result(self, h):
        return h

    def measure_green_switch_out(self, gq, rbits=None):
        a = self.px.measure_green_switch_out(gq)                    # measure_many path
        return self._replay("Z^g measurements (switch-out)", a, lambda r: self.ref.measure_green_switch_out(gq, r))

    def evaluate_class(self, basis):
        a = self.px.evaluate_class(basis)
        b = self.ref.evaluate_class(basis)
        if a != b:
            raise Mismatch(f"final logical class differs: production {a} reference {b}")
        self.n_compared += 1
        return a


def run_replay(vcfg: dict = None) -> list:
    """(D) at d = bf_d against the brute force."""
    vcfg = VCFG if vcfg is None else vcfg
    d = int(vcfg["bf_d"])
    code, tab = get_code(d)
    shots = int(vcfg["replay_shots_per_point"])
    rows = []
    for boundary in vcfg["boundaries"]:
        cfg = dict(CFG)
        cfg["switch_in_boundary"] = boundary
        for case in vcfg["cases"]:
            for p in vcfg["p_values"]:
                runner = ShotRunner(code, p, cfg)
                rb = ReplayBackend(ProxyBackend(tab, lockstep=False), _reference("bf", code, vcfg))
                mism, msg, t0 = 0, "", time.time()
                counts = {c: 0 for c in STATE_CLASSES}
                for s in range(shots):
                    rng = np.random.default_rng([int(vcfg["seed"]) + 1, s, int(p * 1e6), len(case), d])
                    try:
                        counts[runner.run(rb, case, rng)] += 1
                    except Mismatch as exc:
                        mism += 1
                        msg = str(exc)
                rows.append(dict(reference="bf_replay", boundary=boundary, case=case, d=d, p=p, shots=shots,
                                 mismatches=mism, compared_outputs=rb.n_compared, random_outcomes=0,
                                 **{f"n_{c}": counts[c] for c in STATE_CLASSES if counts[c]},
                                 seconds=round(time.time() - t0, 1), last_mismatch=msg))
                print(f"[replay {boundary:5s}] {case:8s} d={d} p={p:.3f}: mismatches={mism}  compared={rb.n_compared}"
                      f"  classes={ {c: k for c, k in counts.items() if k} }  ({time.time() - t0:.1f}s)")
                if msg:
                    print("    last mismatch:", msg)
    return rows


def assert_validation(val_df, u_check: dict):
    """Pass/fail: raises AssertionError unless every comparison agreed and U is the logical T^dagger."""
    assert len(val_df) > 0, "no validation rows"
    assert int(val_df["mismatches"].sum()) == 0, f"{int(val_df['mismatches'].sum())} mismatching shots"
    n_cls = val_df.filter(regex=r"^n_").fillna(0).sum(axis=1)
    assert (n_cls + val_df["mismatches"] == val_df["shots"]).all(), "class counts do not add up to the shots"
    for ref in ("bf", "tf", "bf_replay"):
        assert (val_df["reference"] == ref).any(), f"reference {ref} was not run"
    assert u_check["logical gate"] == "T^dagger", u_check
    assert abs(u_check["|<0|U|0>|"] - 1) < 1e-9 and abs(u_check["|<1|U|1>|"] - 1) < 1e-9, u_check
    return "VALIDATION PASSED"


if __name__ == "__main__":
    import pandas as pd
    u = check_U_bruteforce(2)
    print(u)
    df = pd.DataFrame(run_lockstep("bf") + run_lockstep("tf") + run_replay())
    print(assert_validation(df, u))
