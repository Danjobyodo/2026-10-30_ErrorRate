# -*- coding: utf-8 -*-
# =============================================================================
#  CHSC code switching (folded surface code <-> D4 Clifford stabilizer code), version 3
#  phenomenological-noise logical failure rate of
#     |+>  --(noisy switch-in)-->  D4  --(ideal logical T^dagger = U)-->  --(noisy switch-out)-->  SC
#  Protocol : R. Kobayashi, G. Zhu, P.-S. Hsin, "Clifford Hierarchy Stabilizer Codes: Transversal
#             Non-Clifford Gates and Magic" (main text Fig. 4, SM "Details of code switching", SM Fig. 6).
#  Simulator: Y. Takada, S. D. Bartlett, D. J. Williamson, "Exact efficient simulation of noisy logical
#             magic states using Clifford stabilizers" -- Clifford-stabilizer simulation:
#             measurement-conjugated picture + Pauli dressings + Pauli proxies (Sec. VII.A-C, Algorithm 1),
#             transversal non-Clifford gate / branch-flipping corrections / mid-circuit Clifford corrections
#             (Sec. VII.D, VIII.A-B, Algorithm 3), fictitious Clifford-stabilizer measurement for the logical error (Sec. VI.D).
#             The Pauli-proxy Clifford circuit is executed with Stim (stim.TableauSimulator).
#  Geometry and decoders are taken over unchanged from v1 (chsc_switch_v1); the engine is unchanged from v2.
#  v3 (2026-10-20): run management and reporting fixed (condition-checked resume, split runs by d keep earlier rows,
#  ideal/noisy run side by side, self-checks for every d, P_L-only plots).  See README_v3.md
#  Author: Claude (Opus 5.5)
# =============================================================================

# %% [Cell 1] Imports & configuration
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import pymatching
except ImportError:  # pragma: no cover
    pymatching = None

import stim

try:
    from scipy.stats import norm as _norm
    _SCIPY = True
except ImportError:  # pragma: no cover
    _SCIPY = False

CODE_VERSION = "chsc_switch_v3"

CFG = dict(
    # ===== what to simulate =====
    case="in_T_out",            # "in_only"   : switch-in only (evaluated in the D4 code, target |+>_D4)
                                # "in_out"    : switch-in -> switch-out (no T^dagger, target |+>)
                                # "in_T_out"  : switch-in -> ideal U (logical T^dagger) -> switch-out (target T^dagger|+>)
    cases_to_run=None,          # e.g. ["in_only", "in_out", "in_T_out"] to run all three in one call (overrides `case`)
    noise_model="phenomenological",  # only phenomenological is implemented (see README_v3.md)
    d_values=[3, 5, 7],         # code distance d = number of plaquettes per side of the square (= triangle) patch
    rounds_in=None,             # number of noisy switch-in rounds   (None -> d ;  "O(d) rounds" of the paper)
    rounds_out=None,            # number of noisy surface-code rounds after switch-out (None -> d)
    shots=20000,
    p_min=None, p_max=None, p_n=8,   # None -> RECOMMENDED_P_RANGE
    seed=20261017,
    # ===== protocol details NOT fixed by the paper (same meaning as v1, see README_v3.md) =====
    switch_in_boundary="ideal",  # how the switch-in phase ends before U (README_v3.md Sec. 2 for what each one means):
                                 #  "ideal": one extra round of perfect Z and dressed-X measurements, flux closing and
                                 #           charge clean-up (U then acts on a code state)
                                 #  "noisy": no perfect round. pending JIT fluxes are left uncorrected, charges are cleaned
                                 #           with a decoder whose last layer may end on a "future" boundary
    boundaries_to_run=None,      # e.g. ["ideal", "noisy"]: run both settings in one call (overrides switch_in_boundary)
    x_check_order="rgb",         # order of colour layers of the dressed X checks within a round (rb-combos go with r)
    jit_boundary_offset=1.0,     # JIT waiting cost  w_t * (t - tau + offset)
    green_noise_after_switch_out=False,  # green qubits are discarded after the Z^g measurement (no further noise)
    # ===== output (local files only; nothing is written to Google Drive) =====
    out_dir="chsc_v3_results",   # all CSV / PNG / PDF / JSON files go into this folder (created if missing)
    resume=True,                 # skip (d, p) points already in the CSV IF they were computed under the same conditions
                                 # (all keys of RUN_CONDITION_KEYS + code version); a CSV with other conditions -> error
    confidence_level=0.95,       # confidence level of the Agresti-Coull interval of P_L
    save_pdf=True,
    save_manifest=True,
    run_self_checks=True,
    self_check_shots=100,
    verbose=True,
)

# pilot crossing region for phenomenological noise (v1 pilot); p grid is log-spaced
# keys that define the computed quantity: a CSV is only resumed / extended if all of them agree
RUN_CONDITION_KEYS = ("noise_model", "rounds_in", "rounds_out", "switch_in_boundary", "x_check_order",
                      "jit_boundary_offset", "green_noise_after_switch_out", "seed")

RECOMMENDED_P_RANGE = {"phenomenological": (8e-3, 5e-2)}

CASE_ALIASES = {
    "in_only": "in_only", "switch_in_only": "in_only", "1": "in_only",
    "in_out": "in_out", "switch_in_out": "in_out", "2": "in_out",
    "in_T_out": "in_T_out", "in_t_out": "in_T_out", "switch_in_T_out": "in_T_out", "3": "in_T_out",
}

COLORS = ("r", "g", "b")


# %% [Cell 2] GF(2) linear algebra helpers
def gf2_rref(A: np.ndarray):
    """Row-reduced echelon form over GF(2). Returns (R, pivots)."""
    R = (np.array(A, dtype=np.uint8) & 1).copy()
    piv = []
    r = 0
    nr, nc = R.shape
    for c in range(nc):
        if r >= nr:
            break
        rows = np.nonzero(R[r:, c])[0]
        if rows.size == 0:
            continue
        p = r + rows[0]
        if p != r:
            R[[r, p]] = R[[p, r]]
        others = np.nonzero(R[:, c])[0]
        others = others[others != r]
        if others.size:
            R[others] ^= R[r]
        piv.append(c)
        r += 1
    return R[:r], piv


def gf2_rank(A: np.ndarray) -> int:
    if A.size == 0:
        return 0
    return len(gf2_rref(A)[1])


def gf2_nullspace(A: np.ndarray) -> np.ndarray:
    """Basis (rows) of {x : A x = 0}."""
    A = np.array(A, dtype=np.uint8) & 1
    n = A.shape[1]
    R, piv = gf2_rref(A)
    free = [c for c in range(n) if c not in piv]
    basis = []
    for f in free:
        v = np.zeros(n, np.uint8)
        v[f] = 1
        for i, pc in enumerate(piv):
            if R[i, f]:
                v[pc] = 1
        basis.append(v)
    return np.array(basis, np.uint8).reshape(len(basis), n)


def gf2_solve(A: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    """One solution x of A x = b over GF(2) (None if inconsistent)."""
    A = np.array(A, dtype=np.uint8) & 1
    b = np.array(b, dtype=np.uint8).reshape(-1) & 1
    aug = np.concatenate([A, b[:, None]], axis=1)
    R, piv = gf2_rref(aug)
    n = A.shape[1]
    if n in piv:
        return None
    x = np.zeros(n, np.uint8)
    for i, pc in enumerate(piv):
        x[pc] = R[i, n]
    return x


def gf2_left_inverse(M: np.ndarray) -> np.ndarray:
    """P (k x n) with P M = I_k for a full-column-rank M (n x k)."""
    M = np.array(M, dtype=np.uint8) & 1
    n, k = M.shape
    aug = np.concatenate([M, np.eye(n, dtype=np.uint8)], axis=1)
    R, piv = gf2_rref(aug)
    if piv[:k] != list(range(k)):
        raise AssertionError("M does not have full column rank")
    return R[:k, k:].copy()


# %% [Cell 3] Geometry: square-lattice triangle patch of Fig. 4 / SM Fig. 6
@dataclass
class XGen:
    """X-type stabilizer generator S = X^{supp} * prod_{pairs} CZ  (vertex v, kind r/g/b/rb)."""
    kind: str
    vertex: Tuple[int, int]
    supp: Tuple[int, ...]
    pairs: Tuple[Tuple[int, int], ...]
    comp: int  # 0: red/blue sector (r, b, rb), 1: green sector


class TriangleCode:
    """Square lattice with (n+1)x(n+1) vertices (i, j), 0 <= i, j <= n  (j increases upwards).

    Boundaries (paper Fig. 4): top j=n : L_b (a_b = 0)   left i=0 : L_r (a_r = 0)
                               bottom j=0 and right i=n : L_rb (a_r + a_b = 0, a_g = 0).
    Qubits fixed by single-qubit Z stabilizers in Fig. 4 (b on top edges, r on left edges, g on L_rb edges)
    are omitted (= ideal |0>, see report Sec. 4, A1).  Branching: from the lower-left to the upper-right corner
    of every square; square p(i,j) has corners 0=(i,j), 1=(i+1,j), 2=(i,j+1), 3=(i+1,j+1),
    e01 = bottom, e13 = right, e02 = left, e23 = top (paper Fig. 1).
    """

    def __init__(self, n: int, junction: Optional[Tuple[int, int]] = None):
        if n < 2:
            raise ValueError("n (= d) must be >= 2")
        self.n = n
        self.d = n
        edges = []
        eid = {}
        for j in range(n + 1):
            for i in range(n):
                eid[("h", i, j)] = len(edges)
                edges.append(("h", i, j))
        for i in range(n + 1):
            for j in range(n):
                eid[("v", i, j)] = len(edges)
                edges.append(("v", i, j))
        self.edges = edges
        self.eid = eid
        self.E = len(edges)

        def bclass(e):
            kind, i, j = e
            if kind == "h" and j == n:
                return "top"
            if kind == "v" and i == 0:
                return "left"
            if kind == "h" and j == 0:
                return "bottom"
            if kind == "v" and i == n:
                return "right"
            return "bulk"

        removed = {"top": "b", "left": "r", "bottom": "g", "right": "g"}
        self.edge_class = [bclass(e) for e in edges]
        qubits = []
        qid = {}
        for e_idx, e in enumerate(edges):
            for c in COLORS:
                if removed.get(self.edge_class[e_idx]) == c:
                    continue
                qid[(e_idx, c)] = len(qubits)
                qubits.append((e_idx, c))
        self.qubits = qubits
        self.qid = qid
        self.nq = len(qubits)
        self.q_color = np.array([COLORS.index(c) for _, c in qubits], np.int32)
        self.q_edge = np.array([e for e, _ in qubits], np.int32)

        def Q(kind, i, j, c):
            key = (kind, i, j)
            if key not in eid:
                return -1
            return qid.get((eid[key], c), -1)

        self._Q = Q

        def ends(e):
            kind, i, j = e
            return ((i, j), (i + 1, j)) if kind == "h" else ((i, j), (i, j + 1))

        vert_edges: Dict[Tuple[int, int], List[int]] = {}
        for e_idx, e in enumerate(edges):
            for v in ends(e):
                vert_edges.setdefault(v, []).append(e_idx)
        self.vert_edges = vert_edges

        on_Lrb = lambda v: v[1] == 0 or v[0] == n  # noqa: E731

        # ----- Z checks -----
        zchecks = []  # (label, comp, qubits)
        for j in range(n):
            for i in range(n):
                es = [eid[("h", i, j)], eid[("v", i + 1, j)], eid[("v", i, j)], eid[("h", i, j + 1)]]
                for c in COLORS:
                    qs = [qid[(e, c)] for e in es if (e, c) in qid]
                    zchecks.append((f"P{c}({i},{j})", 1 if c == "g" else 0, tuple(qs)))
        for e_idx, cl in enumerate(self.edge_class):
            if cl in ("bottom", "right"):
                zchecks.append((f"ZZ{self.edges[e_idx]}", 0, (qid[(e_idx, "r")], qid[(e_idx, "b")])))
        self.zchecks = zchecks

        # ----- X-type generators (dressed) -----
        def red_pairs(i, j):
            if i <= n - 1 and j <= n - 1:  # v = corner 0 of p(i,j)
                return [(Q("h", i, j, "g"), Q("v", i + 1, j, "b")), (Q("v", i, j, "g"), Q("h", i, j + 1, "b"))]
            return []

        def green_pairs(i, j):
            out = []
            if i >= 1 and j <= n - 1:  # v = corner 1 of p(i-1, j):  a_r(01) v(1) a_b(13)
                out.append((Q("h", i - 1, j, "r"), Q("v", i, j, "b")))
            if i <= n - 1 and j >= 1:  # v = corner 2 of p(i, j-1):  a_r(02) v(2) a_b(23)
                out.append((Q("v", i, j - 1, "r"), Q("h", i, j, "b")))
            return out

        def blue_pairs(i, j):
            if i >= 1 and j >= 1:  # v = corner 3 of p(i-1, j-1)
                return [(Q("h", i - 1, j - 1, "r"), Q("v", i, j - 1, "g")),
                        (Q("v", i - 1, j - 1, "r"), Q("h", i - 1, j, "g"))]
            return []

        def clean(prs):
            return tuple((a, b) for a, b in prs if a >= 0 and b >= 0)

        Zmat = np.zeros((len(zchecks), self.nq), np.uint8)
        for r_, (_, _, qs) in enumerate(zchecks):
            Zmat[r_, list(qs)] = 1
        self.Zmat = Zmat

        gens: List[XGen] = []
        self.dropped_combos = []
        for j in range(n + 1):
            for i in range(n + 1):
                v = (i, j)
                inc = vert_edges[v]
                if not on_Lrb(v):
                    if i >= 1:
                        supp = tuple(qid[(e, "r")] for e in inc if (e, "r") in qid)
                        gens.append(XGen("r", v, supp, clean(red_pairs(i, j)), 0))
                    supp = tuple(qid[(e, "g")] for e in inc if (e, "g") in qid)
                    gens.append(XGen("g", v, supp, clean(green_pairs(i, j)), 1))
                    if j <= n - 1:
                        supp = tuple(qid[(e, "b")] for e in inc if (e, "b") in qid)
                        gens.append(XGen("b", v, supp, clean(blue_pairs(i, j)), 0))
                else:
                    supp = tuple(qid[(e, c)] for e in inc for c in ("r", "b") if (e, c) in qid)
                    sv = np.zeros(self.nq, np.uint8)
                    sv[list(supp)] = 1
                    if np.any((Zmat @ sv) % 2):
                        self.dropped_combos.append(v)  # does not commute with Z checks (corners (0,0), (n,n))
                        continue
                    gens.append(XGen("rb", v, supp, clean(red_pairs(i, j) + blue_pairs(i, j)), 0))
        self.gens = gens
        self.ngen = len(gens)
        for g in gens:
            sv = np.zeros(self.nq, np.uint8)
            sv[list(g.supp)] = 1
            if np.any((Zmat @ sv) % 2):
                raise AssertionError(f"X part of generator {g.kind}{g.vertex} anticommutes with a Z check")
            for a, b in g.pairs:
                if a in g.supp or b in g.supp:
                    raise AssertionError("dressing pair overlaps X support")
        self.gen_supp_mat = np.zeros((self.ngen, self.nq), np.uint8)
        for gi, g in enumerate(gens):
            self.gen_supp_mat[gi, list(g.supp)] = 1

        # ----- logical Z (SM Fig. 6, right): junction J -----
        i0, j0 = junction if junction is not None else (max(1, n // 2), max(1, n // 2))
        zl = []
        for i in range(i0):
            zl.append(Q("h", i, j0, "r"))
        for j in range(j0, n):
            zl.append(Q("v", i0, j, "b"))
        for j in range(j0):
            zl.append(Q("v", i0, j, "r"))
            zl.append(Q("v", i0, j, "b"))
        if min(zl) < 0:
            raise AssertionError("logical Z uses an absent qubit")
        self.zbar = np.zeros(self.nq, np.uint8)
        self.zbar[zl] ^= 1
        if np.any((self.gen_supp_mat @ self.zbar) % 2):
            raise AssertionError("logical Z anticommutes with an X generator")

        # ----- flat configurations C0 = ker(Z checks) and coordinates -----
        ker = gf2_nullspace(Zmat)
        rank_gens = gf2_rank(self.gen_supp_mat)
        self.n_logical = ker.shape[0] - rank_gens
        if self.n_logical != 1:
            raise AssertionError(f"expected 1 logical qubit, found {self.n_logical}")
        # eta: flat, green part 0, odd overlap with Zbar (nontrivial configuration of SM Sec. II)
        green_rows = np.zeros((int(np.sum(self.q_color == 1)), self.nq), np.uint8)
        for r_, q in enumerate(np.nonzero(self.q_color == 1)[0]):
            green_rows[r_, q] = 1
        A = np.concatenate([Zmat, green_rows, self.zbar[None, :]], axis=0)
        rhs = np.zeros(A.shape[0], np.uint8)
        rhs[-1] = 1
        eta = gf2_solve(A, rhs)
        if eta is None:
            raise AssertionError("no nontrivial flat configuration with a_g = 0")
        # make eta sparse-ish: reduce by r/b generator supports greedily (cosmetic)
        self.eta = eta.astype(np.uint8)
        # basis of C0: independent generator supports (in generator order) + eta (last coordinate)
        basis_idx = []
        cur = np.zeros((0, self.nq), np.uint8)
        for gi in range(self.ngen):
            cand = np.concatenate([cur, self.gen_supp_mat[gi:gi + 1]], axis=0)
            if gf2_rank(cand) > cur.shape[0]:
                cur = cand
                basis_idx.append(gi)
        self.basis_gens = basis_idx
        self.k = len(basis_idx) + 1
        self.kh = self.k - 1
        M = np.zeros((self.nq, self.k), np.uint8)
        for col, gi in enumerate(basis_idx):
            M[:, col] = self.gen_supp_mat[gi]
        M[:, self.kh] = self.eta
        self.M0 = M
        self.M0plus = gf2_left_inverse(M)
        if gf2_rank(np.concatenate([M.T, ker], axis=0)) != self.k or self.k != ker.shape[0]:
            raise AssertionError("coordinates do not span C0")
        if int(self.zbar @ self.eta) % 2 != 1:
            raise AssertionError("eta must anticommute with Zbar")
        self.coord_of_gen = {gi: col for col, gi in enumerate(basis_idx)}

        # ----- U = W V ingredients (main text Eqs. (6),(8)) -----
        cs = []  # (q_r, q_b, sign) : sign=+1 CS, -1 CS^dagger
        for j in range(n):
            for i in range(n):
                cs.append((Q("h", i, j, "r"), Q("v", i + 1, j, "b"), +1))   # CS(r e01, b e13)
                cs.append((Q("v", i, j, "r"), Q("h", i, j + 1, "b"), -1))   # CS^dag(r e02, b e23)
        self.cs_terms = [(a, b, s) for a, b, s in cs if a >= 0 and b >= 0]
        self.tdag_qubits = [qid[(e, "r")] for e, cl in enumerate(self.edge_class) if cl in ("bottom", "right")]
        # V = prod_e CNOT(r->g) CNOT(b->g); on L_rb edges the green qubit is absent (A1)
        vmap = []
        for e_idx in range(self.E):
            if (e_idx, "g") in qid:
                vmap.append((qid[(e_idx, "g")], qid.get((e_idx, "r"), -1), qid.get((e_idx, "b"), -1)))
        self.vmap = vmap

        # ----- components for decoders -----
        self.rb_qubits = np.nonzero(self.q_color != 1)[0].astype(np.int32)
        self.g_qubits = np.nonzero(self.q_color == 1)[0].astype(np.int32)
        self.zc_rb = [i for i, z in enumerate(zchecks) if z[1] == 0]
        self.zc_g = [i for i, z in enumerate(zchecks) if z[1] == 1]
        self.xg_rb = [i for i, g in enumerate(gens) if g.comp == 0]
        self.xg_g = [i for i, g in enumerate(gens) if g.comp == 1]
        order = {"r": [], "g": [], "b": []}
        for i, g in enumerate(gens):
            order["r" if g.kind in ("r", "rb") else g.kind].append(i)
        self.order_by_color = order

    # --------------------------------------------------------------------- utilities
    def apply_V_vec(self, a: np.ndarray) -> np.ndarray:
        """Physical V on a configuration / X-frame vector (a_g += a_r + a_b, green present)."""
        out = np.array(a, dtype=np.uint8).copy()
        for qg, qr, qb in self.vmap:
            v = 0
            if qr >= 0:
                v ^= int(a[qr])
            if qb >= 0:
                v ^= int(a[qb])
            out[qg] ^= v
        return out

    def V_rows(self, M: np.ndarray) -> np.ndarray:
        """Rows of V M (qubit-indexed linear forms)."""
        out = np.array(M, dtype=np.uint8).copy()
        for qg, qr, qb in self.vmap:
            if qr >= 0:
                out[qg] ^= M[qr]
            if qb >= 0:
                out[qg] ^= M[qb]
        return out

    def W_exponent(self, a: np.ndarray) -> int:
        """W(a) = omega^{e}, omega = exp(i pi/4):  CS -> 2 a a',  CS^dag -> -2 a a',  T^dag -> -a."""
        e = 0
        for qa, qb, s in self.cs_terms:
            e += 2 * s * int(a[qa]) * int(a[qb])
        for q in self.tdag_qubits:
            e -= int(a[q])
        return e % 8

    def summary(self) -> Dict[str, object]:
        kinds = {}
        for g in self.gens:
            kinds[g.kind] = kinds.get(g.kind, 0) + 1
        return dict(d=self.n, qubits=self.nq, z_checks=len(self.zchecks), x_generators=self.ngen,
                    generators_by_kind=kinds, coords_k=self.k, dropped_rb_corners=self.dropped_combos,
                    weight_Zbar=int(self.zbar.sum()), weight_eta=int(self.eta.sum()),
                    cs_terms=len(self.cs_terms), tdag_qubits=len(self.tdag_qubits))


# %% [Cell 4] phi0 (code-state phase) bookkeeping, logical action of U, Theorem-2 check
def phi0_walk(code: TriangleCode, b: np.ndarray) -> int:
    """phi0(b) for a flat configuration b (convention phi0(0) = phi0(eta) = 0).

    Uses the stabilizer identity phi0(b + s) = phi0(b) + q_s(b) of every dressed generator S = X^s CZ^{pairs}
    (q_s(b) = sum_{pairs} b_i b_j), applied generator by generator from h*eta.  Never needed by the sampler
    itself except in the final evaluation in the surface code and in the self-checks.
    """
    b = np.asarray(b, dtype=np.uint8) & 1
    u = (code.M0plus.astype(np.int64) @ b) % 2
    if not np.array_equal((code.M0.astype(np.int64) @ u) % 2, b):
        raise AssertionError("configuration is not flat (not in C0)")
    cur = (code.eta * u[code.kh]).astype(np.uint8)
    phi = 0
    for col, gi in enumerate(code.basis_gens):
        if u[col]:
            g = code.gens[gi]
            for qa, qb in g.pairs:
                phi ^= int(cur[qa]) & int(cur[qb])
            cur[list(g.supp)] ^= 1
    return phi


def logical_U_phase(code: TriangleCode) -> int:
    """kappa with  U |1_D4> = omega^kappa U |0_D4>-phase, i.e. c_1/c_0 = omega^kappa (paper: kappa = -1 = 7, T^dagger)."""
    b = code.eta.copy()
    vb = code.apply_V_vec(b)
    return (code.W_exponent(vb) + 4 * (phi0_walk(code, b) ^ phi0_walk(code, vb))) % 8


def check_theorem_pointwise(code: TriangleCode, n_samples: int = 50, seed: int = 1) -> Dict[str, object]:
    """Check W(Vb) (-1)^{phi0(b)-phi0(Vb)} = omega^{kappa h(b)} for random flat b (Theorems 1-2 of the paper)."""
    rng = np.random.default_rng(seed)
    kappa = logical_U_phase(code)
    bad = 0
    for _ in range(n_samples):
        t = rng.integers(0, 2, code.k).astype(np.uint8)
        b = (code.M0.astype(np.int64) @ t) % 2
        vb = code.apply_V_vec(b)
        e = (code.W_exponent(vb) + 4 * (phi0_walk(code, b) ^ phi0_walk(code, vb))) % 8
        if e != (kappa * int(t[code.kh])) % 8:
            bad += 1
    return dict(kappa=kappa, logical_gate={7: "T^dagger", 1: "T", 0: "I"}.get(kappa, f"omega^{kappa}"),
                samples=n_samples, violations=bad)


def check_flat_commutation(code: TriangleCode) -> Tuple[int, int]:
    """(# operator-level non-commuting generator pairs, # pairs non-commuting on the flat subspace (must be 0))."""
    n = code.nq
    F = code.Zmat
    rF = gf2_rank(F)
    data = []
    for g in code.gens:
        sv = np.zeros(n, np.uint8)
        sv[list(g.supp)] = 1
        data.append((sv, g.pairs))
    op_nc = flat_nc = 0
    for A in range(len(data)):
        for B in range(A + 1, len(data)):
            (sA, pA), (sB, pB) = data[A], data[B]
            y = np.zeros(n, np.uint8)
            c = 0
            for (i, j), sv in [(pr, sB) for pr in pA] + [(pr, sA) for pr in pB]:
                y[i] ^= sv[j]
                y[j] ^= sv[i]
                c ^= int(sv[i]) & int(sv[j])
            if y.any() or c:
                op_nc += 1
                if c or gf2_rank(np.vstack([F, y])) > rF:
                    flat_nc += 1
    return op_nc, flat_nc



def check_intertwiner_conditions(code: TriangleCode, n_samples: int = 10, seed: int = 2) -> Dict[str, object]:
    """Conditions under which the Pauli-proxy reduction is exact for this protocol (README_v3.md Sec. 3.3).

    (C1) phi0(a + s_G) = phi0(a) + q_G(a) (mod 2) for every dressed generator G = X^{s_G} CZ^{pairs} and every flat a,
         i.e. C X^{s_G} C^dagger = G on the flat subspace, C = diag((-1)^phi0)  ->  the Pauli proxy of G is X^{s_G};
    (C2) phi0(b) = 0 for every flat b with a_g = 0  ->  C acts trivially on folded-surface-code states;
    (C3) U C = C V T_h on the flat subspace with T_h = diag(omega^{kappa h}), h = Zbar-parity
         (= check_theorem_pointwise)  ->  the Pauli proxy of U is V, the logical T^dagger is kept as T_h."""
    rng = np.random.default_rng(seed)
    M0 = code.M0.astype(np.int64)
    bad1 = bad2 = 0
    green_cols = [col for col, gi in enumerate(code.basis_gens) if code.gens[gi].kind == "g"]
    for _ in range(n_samples):
        t = rng.integers(0, 2, code.k).astype(np.uint8)
        a = ((M0 @ t) % 2).astype(np.uint8)
        pa = phi0_walk(code, a)
        for g in code.gens:
            q = 0
            for i, j in g.pairs:
                q ^= int(a[i]) & int(a[j])
            a2 = a.copy()
            a2[list(g.supp)] ^= 1
            if phi0_walk(code, a2) != pa ^ q:
                bad1 += 1
        t0 = t.copy()
        t0[green_cols] = 0
        b = ((M0 @ t0) % 2).astype(np.uint8)
        if np.any(b[code.g_qubits]) or phi0_walk(code, b) != 0:
            bad2 += 1
    th = check_theorem_pointwise(code, n_samples=max(20, n_samples), seed=seed + 1)
    return dict(samples=n_samples, C1_violations=bad1, C2_violations=bad2, C3_violations=th["violations"],
                kappa=th["kappa"])


# %% [Cell 5] Pauli-proxy engine (Clifford-stabilizer simulation of Takada-Bartlett-Williamson, run on Stim)
#
#  Measurement-conjugated picture (paper Sec. VII.A): the physical state is  E |frame>,  E = accumulated errors and
#  corrections.  Measuring M on it  ==  measuring  E^dagger M E  on |frame>  (then E is applied again).
#  E is kept in the form  E = X^x diag(omega^gamma(a)),  gamma(a) = sum_i l_i a_i + 4 sum_{i<j} Q_ij a_i a_j (mod 8),
#  omega = e^{i pi/4}: a Pauli X frame times a diagonal Clifford (Z: l=4, S: l=2, S^dag: l=6, CZ: Q=1).
#  Before U, E is Pauli.  Diagonal Clifford parts appear when Pauli X errors pass the transversal non-Clifford
#  W (CS, T^dagger) and through the green correction S^g_X (CZ^{b,r}).
#  For a factorised measured operator  M = X^s Delta  (Delta = its diagonal Clifford factor, e.g. the CZ dressing of
#  a D4 generator)
#        E^dagger M E = X^s Delta diag(omega^{ [delta(a+x) - delta(a)] + [gamma(a) - gamma(a+s)] })
#  the second factor is a Pauli dressing (Z string with a phase).  Replacing Delta by its Pauli proxy (= identity:
#  the proxy of a dressed generator is its bare X star) gives a Pauli operator, measured with Stim.
#  The frame state is a Pauli-stabilizer state in the proxy picture (the "Pauli-proxy tableau").
#  Why this is exact for this protocol: README_v3.md Sec. 3 (conditions C1-C3 are checked in run_self_checks).

class EngineTables:
    """Per-geometry tables shared by all shots (supports, CZ pairs, U, green solver, initial proxy state)."""

    def __init__(self, code: TriangleCode):
        self.code = code
        n = code.nq
        self.n = n
        self.S = code.gen_supp_mat.astype(np.uint8)                       # (ngen, n) X supports
        mp = max(1, max(len(g.pairs) for g in code.gens))
        self.PA = np.full((code.ngen, mp), n, np.int64)                    # CZ pairs (padded with dummy qubit n)
        self.PB = np.full((code.ngen, mp), n, np.int64)
        for gi, g in enumerate(code.gens):
            for u, (a, b) in enumerate(g.pairs):
                self.PA[gi, u] = a
                self.PB[gi, u] = b
        self.Zmat = code.Zmat.astype(np.int64)
        self.kappa = logical_U_phase(code)
        # U = W V :  V = CNOT(r->g) CNOT(b->g) (proxy of U),  W = CS / CS^dag / T^dag (removed in the proxy picture)
        cx = []
        for qg, qr, qb in code.vmap:
            if qr >= 0:
                cx += [qr, qg]
            if qb >= 0:
                cx += [qb, qg]
        self.cx_text = "CX " + " ".join(str(q) for q in cx)
        # green correction S^g_X(lambda): solve  sum_v lambda_v supp_v = m  on green qubits
        Sg = code.gen_supp_mat[code.xg_g][:, code.g_qubits]
        if gf2_rank(Sg) != Sg.shape[0]:
            raise AssertionError("green generators are not independent")
        self.green_S = Sg.astype(np.int64)
        self.green_left = gf2_left_inverse(Sg.T.copy()).astype(np.int64)
        self.green_gens = np.asarray(code.xg_g, np.int64)
        # Stim tokens per qubit: 1 = X, 2 = Z, 3 = Y
        self.tok = [None, [f"X{q}" for q in range(n)], [f"Z{q}" for q in range(n)], [f"Y{q}" for q in range(n)]]
        # initial frame = initial physical state |+>_SC (r,b) x |0>^g  (a Pauli-stabilizer state)
        stabs = []
        for gi in code.xg_rb:
            stabs.append(self._ps(xs=code.gens[gi].supp))
        for c in code.zc_rb:
            stabs.append(self._ps(zs=code.zchecks[c][2]))
        stabs.append(self._ps(xs=np.nonzero(code.eta)[0]))
        for q in code.g_qubits:
            stabs.append(self._ps(zs=[q]))
        tab = stim.Tableau.from_stabilizers(stabs, allow_redundant=True, allow_underconstrained=False)
        sim = stim.TableauSimulator()
        sim.set_inverse_tableau(tab.inverse())
        self.sim0 = sim
        self.eta = code.eta.astype(np.uint8)
        self.zbar = code.zbar.astype(np.uint8)
        self.zbar_ps = self._ps(zs=np.nonzero(code.zbar)[0])

    def _ps(self, xs=(), zs=(), sign=1) -> stim.PauliString:
        xa = np.zeros(self.n, bool)
        za = np.zeros(self.n, bool)
        xa[list(xs)] = True
        za[list(zs)] = True
        return stim.PauliString.from_numpy(xs=xa, zs=za, sign=sign)


class ErrorFrame:
    """E = X^x diag(omega^gamma(a)),  gamma(a) = l . a + 4 sum_{i<j} Q_ij a_i a_j  (mod 8; global constant dropped).

    Arrays carry one dummy qubit (index n, always 0) used for padding."""

    def __init__(self, n: int):
        self.n = n
        self.x = np.zeros(n + 1, np.uint8)
        self.l = np.zeros(n + 1, np.int64)
        self.Q = np.zeros((n + 1, n + 1), np.uint8)
        self.q_dirty = False

    def reset(self):
        self.x[:] = 0
        self.l[:] = 0
        if self.q_dirty:
            self.Q[:] = 0
            self.q_dirty = False

    def is_pauli(self) -> bool:
        return (not self.q_dirty or not self.Q.any()) and not np.any(self.l % 4)

    # ---- left multiplication  E <- F E
    def left_x(self, qs):
        if len(qs):
            np.bitwise_xor.at(self.x, np.asarray(qs, np.int64), 1)

    def left_z(self, qs):
        if len(qs):
            np.add.at(self.l, np.asarray(qs, np.int64), 4)
            self.l %= 8

    def left_cz(self, pairs):
        """E <- prod CZ_{ij} E :  gamma(a) += 4 (a_i+x_i)(a_j+x_j)."""
        x = self.x
        for i, j in pairs:
            self.Q[i, j] ^= 1
            self.Q[j, i] ^= 1
            self.l[i] += 4 * int(x[j])
            self.l[j] += 4 * int(x[i])
            self.q_dirty = True
        self.l %= 8

    # ---- right multiplication  E <- E X^c CZ^{pairs}   (branch-flipping operator, Takada et al. Algorithm 3)
    def right_xcz(self, c: np.ndarray, pairs):
        """gamma'(a) = gamma(a + c) + 4 sum_{pairs} a_i a_j ,  x' = x + c."""
        n = self.n
        cm = np.zeros(n + 1, np.int64)
        cm[:n] = np.asarray(c, np.int64) & 1
        add = np.zeros(n + 1, np.int64)
        if self.q_dirty:
            add = 4 * ((self.Q.astype(np.int64) @ cm) % 2)
        self.l = (np.where(cm == 1, -self.l, self.l) + add) % 8
        self.x ^= cm.astype(np.uint8)
        for i, j in pairs:
            self.Q[i, j] ^= 1
            self.Q[j, i] ^= 1
            self.q_dirty = True

    # ---- the ideal logical gate:  E <- U E U^dagger  (E must be Pauli here)
    def conj_U(self, code: TriangleCode):
        if not self.is_pauli():
            raise AssertionError("E must be a Pauli operator when U is applied")
        n = self.n
        z = (self.l[:n] == 4).astype(np.int64)
        xv = code.apply_V_vec(self.x[:n])
        lnew = np.zeros(n + 1, np.int64)
        lnew[:n] = 4 * z
        for qg, qr, qb in code.vmap:          # Z^g -> Z^r Z^g Z^b  under V
            if z[qg]:
                if qr >= 0:
                    lnew[qr] += 4
                if qb >= 0:
                    lnew[qb] += 4
        Q = self.Q
        for qa, qb, s in code.cs_terms:       # X W X W^dagger for CS^{s}:  S-type and CZ-type Clifford errors
            xa, xb = int(xv[qa]), int(xv[qb])
            if xa and not xb:
                lnew[qb] += 2 * s
                Q[qa, qb] ^= 1
                Q[qb, qa] ^= 1
                self.q_dirty = True
            elif xb and not xa:
                lnew[qa] += 2 * s
                Q[qa, qb] ^= 1
                Q[qb, qa] ^= 1
                self.q_dirty = True
            elif xa and xb:
                lnew[qa] -= 2 * s
                lnew[qb] -= 2 * s
        for q in code.tdag_qubits:            # X T^dag X T = (phase) S
            if xv[q]:
                lnew[q] += 2
        self.x[:n] = xv
        self.l = lnew % 8

    # ---- Pauli-dressed measured operators
    def measured(self, S: np.ndarray, PA: Optional[np.ndarray] = None, PB: Optional[np.ndarray] = None):
        """Pauli-proxy version of E^dagger M E for factorised operators M = X^{S[g]} CZ^{(PA[g],PB[g])}.

        Returns (L, ph): the operator is  omega^ph * P  with P the Pauli string with X on S\\L, Y on S&L, Z on L\\S."""
        n = self.n
        G = S.shape[0]
        x = self.x
        L = np.zeros((G, n + 1), np.uint8)
        c = np.zeros(G, np.int64)
        if PA is not None:                    # X frame through the CZ factors:  Z dressing on the partners
            xa = x[PA]
            xb = x[PB]
            rows = np.broadcast_to(np.arange(G)[:, None], PA.shape)
            np.bitwise_xor.at(L, (rows, PB), xa)
            np.bitwise_xor.at(L, (rows, PA), xb)
            c += 4 * (xa & xb).sum(axis=1).astype(np.int64)
        L = L[:, :n]
        l = self.l[:n]
        if np.any(l):                         # Z (sign) and S-type (Z dressing) parts of gamma
            c -= S.astype(np.int64) @ l
            L ^= (S & ((l % 4) == 2).astype(np.uint8)[None, :])
        if self.q_dirty:                      # CZ-type parts of gamma
            R = np.nonzero(self.Q[:n, :n].any(axis=1))[0]
            if R.size:
                QR = self.Q[R, :n].astype(np.int64)
                A = S[:, R].astype(np.int64)
                L ^= ((A @ QR) % 2).astype(np.uint8)
                inner = ((S.astype(np.int64) @ QR.T) * A).sum(axis=1) // 2
                c += 4 * inner
        k = (S & L).sum(axis=1).astype(np.int64)
        ph = (c - 2 * k) % 8
        return L, ph


class _Handle:
    __slots__ = ("sign", "out")

    def __init__(self, sign):
        self.sign = sign
        self.out = None


_PHASE = {0: 1, 2: 1j, 4: -1, 6: -1j}


class ProxyBackend:
    """Pauli-proxy Clifford circuit on stim.TableauSimulator + error frame E (exact, see README_v3.md Sec. 3).

    lockstep=False (production): measurements are queued as MPP instructions and run in one Stim call per segment.
    lockstep=True  (validation): every measurement is executed immediately; random outcomes are forced to the
                                 supplied bits (peek + postselect) and the kind (det/rand) is recorded."""

    name = "pauli_proxy"

    def __init__(self, tab: EngineTables, lockstep: bool = False):
        self.tab = tab
        self.code = tab.code
        self.n = tab.n
        self.E = ErrorFrame(self.n)
        self.lockstep = lockstep
        self.needs_rbits = lockstep
        self.last_kinds: List[str] = []
        self.stats = np.zeros(2, np.int64)  # (#deterministic, #random) proxy measurements (lockstep mode only)
        self.reset()

    def reset(self, seed: Optional[int] = None):
        self.sim = self.tab.sim0.copy(seed=seed)
        self.E.reset()
        self.after_U = False
        self.queue: List[str] = []
        self.pending: List[_Handle] = []
        self.nrec = 0

    # ---- Pauli errors / corrections (left multiplication of E)
    def apply_x(self, qs):
        self.E.left_x(qs)

    def apply_z(self, qs):
        self.E.left_z(qs)

    def apply_gen_unitaries(self, gen_idx):
        """Dressed generators applied as unitaries (green correction S^g_X(lambda), D4 read-out): E <- G E."""
        for gi in gen_idx:
            g = self.code.gens[int(gi)]
            self.E.left_cz(g.pairs)
            self.E.left_x(list(g.supp))

    def x_frame(self) -> np.ndarray:
        return self.E.x[:self.n].copy()

    # ---- Z checks: deterministic in the frame (frame Z checks are +1), outcome = parity of the X frame
    def zchecks(self, idx: np.ndarray) -> np.ndarray:
        return ((self.tab.Zmat[idx] @ self.E.x[:self.n].astype(np.int64)) % 2).astype(np.uint8)

    # ---- X-type measurements (dressed D4 generators or bare surface-code stars)
    def _pauli(self, Srow, Lrow, ph) -> stim.PauliString:
        return stim.PauliString.from_numpy(xs=Srow.astype(bool), zs=Lrow.astype(bool), sign=_PHASE[int(ph)])

    def measure_gens(self, gen_idx, bare: bool, rbits: Optional[np.ndarray] = None):
        gen_idx = np.asarray(gen_idx, np.int64)
        S = self.tab.S[gen_idx]
        if bare:
            L, ph = self.E.measured(S)
        else:
            L, ph = self.E.measured(S, self.tab.PA[gen_idx], self.tab.PB[gen_idx])
        if np.any((ph != 0) & (ph != 4)):
            raise AssertionError("non-Hermitian Pauli-dressed operator")
        sign = (ph == 4).astype(np.uint8)
        if self.lockstep:
            out = np.zeros(len(gen_idx), np.uint8)
            self.last_kinds = []
            for u in range(len(gen_idx)):
                out[u] = self._measure_now(self._pauli(S[u], L[u], ph[u]), int(rbits[u]))
            return out
        P = S + 2 * L
        tok = self.tab.tok
        for u in range(len(gen_idx)):
            row = P[u]
            idx = np.flatnonzero(row)
            self.queue.append("*".join([tok[row[q]][q] for q in idx]))
        h = _Handle(sign)
        self.pending.append(h)
        return h

    def _measure_now(self, ps: stim.PauliString, rbit: int) -> int:
        e = self.sim.peek_observable_expectation(ps)
        if e == 0:
            self.sim.postselect_observable(ps, desired_value=bool(rbit))
            self.last_kinds.append("rand")
            self.stats[1] += 1
            return int(rbit)
        self.last_kinds.append("det")
        self.stats[0] += 1
        return 0 if e > 0 else 1

    def result(self, h) -> np.ndarray:
        if isinstance(h, np.ndarray):
            return h
        if h.out is None:
            self.flush()
        return h.out

    def flush(self):
        if not self.queue:
            return
        self.sim.do(stim.Circuit("MPP " + " ".join(self.queue)))
        rec = self.sim.current_measurement_record()
        new = np.asarray(rec[self.nrec:], np.uint8)
        self.nrec = len(rec)
        pos = 0
        for h in self.pending:
            k = h.sign.shape[0]
            h.out = new[pos:pos + k] ^ h.sign
            pos += k
        if pos != new.shape[0]:
            raise AssertionError("measurement record length mismatch")
        self.queue = []
        self.pending = []

    # ---- the ideal logical gate U = W V :  proxy picture applies V only (README_v3.md Sec. 3.3)
    def apply_U(self):
        self.flush()
        self.sim.do(stim.Circuit(self.tab.cx_text))
        self.E.conj_U(self.code)
        self.after_U = True

    # ---- switch-out: Z^g on every green qubit + branch-flipping normalisation (README_v3.md Sec. 3.4)
    def measure_green_switch_out(self, gq: np.ndarray, rbits: Optional[np.ndarray] = None) -> np.ndarray:
        """Returns the physical outcomes (before read-out flips).  Afterwards the frame is projected onto Z^g = +1
        (proxy: X on the flipped green qubits) and the true branch-flipping operator S^g_X(lambda_fr) is moved
        into E (right multiplication), exactly as in Algorithm 3 of Takada et al."""
        self.flush()
        gq = np.asarray(gq, np.int64)
        sx = self.E.x[gq].copy()          # E^dagger Z_q E = (-1)^{x_q} Z_q
        if self.lockstep:
            phys = np.zeros(gq.shape[0], np.uint8)
            self.last_kinds = []
            for u, q in enumerate(gq):
                zs = np.zeros(self.n, bool)
                zs[q] = True
                ps = stim.PauliString.from_numpy(xs=np.zeros(self.n, bool), zs=zs, sign=-1 if sx[u] else 1)
                phys[u] = self._measure_now(ps, int(rbits[u]))
            m = phys ^ sx
        else:
            m = np.asarray(self.sim.measure_many(*[int(q) for q in gq]), np.uint8)
            self.nrec += gq.shape[0]
            phys = m ^ sx
        tab = self.tab
        lam = (tab.green_left @ m.astype(np.int64)) % 2
        if not np.array_equal((tab.green_S.T @ lam) % 2, m):
            raise AssertionError("frame Z^g pattern is not a coboundary (frame left the flat subspace?)")
        flip = gq[m == 1]
        if flip.size:
            self.sim.do(stim.Circuit("X " + " ".join(str(int(q)) for q in flip)))
        c = np.zeros(self.n, np.uint8)
        c[gq] = m
        pairs = []
        for gi in tab.green_gens[np.nonzero(lam)[0]]:
            pairs.extend(self.code.gens[int(gi)].pairs)
        self.E.right_xcz(c, pairs)
        return phys

    # ---- logical evaluation: fictitious measurement of the Clifford stabilizer (paper Sec. VI.D)
    def logical_operators(self, basis: str):
        """(O, Zt, Yt): Pauli-proxy versions of the Clifford stabilizer of the target state and of the logical Z, Y
        conjugated by the error frame (and by T_h for the T^dagger case)."""
        self.flush()
        n = self.n
        x = self.E.x[:n]
        if basis == "D4" and np.any(x):
            raise AssertionError("D4 evaluation requires x = 0")
        tab = self.tab
        L, ph = self.E.measured(tab.eta[None, :])
        O = self._pauli(tab.eta, L[0], ph[0])
        hx = int(tab.zbar @ x) % 2
        kap = tab.kappa if (basis == "SC" and self.after_U) else 0
        if hx and kap % 2:  # X frame anticommuting with Zbar passes T_h: extra factor i^{-kappa} Zbar
            O = O * tab.zbar_ps * (1j ** ((-kap) % 4))
        Zt = tab.zbar_ps * (-1 if hx else 1)
        Yt = O * Zt * 1j
        for P in (O, Zt, Yt):
            if P.sign not in (1, -1):
                raise AssertionError("non-Hermitian logical operator")
        return O, Zt, Yt, kap

    def evaluate_class(self, basis: str) -> str:
        O, Zt, Yt, kap = self.logical_operators(basis)
        eo = self.sim.peek_observable_expectation(O)
        if eo == 1:
            return "I"
        if eo == -1:
            return "Z"
        if self.sim.peek_observable_expectation(Zt) != 0:
            return "collapsed"
        ey = self.sim.peek_observable_expectation(Yt)
        if ey == 0:
            raise AssertionError("logical state is not a pure single-qubit stabilizer state (not in the code space?)")
        if kap % 2:
            return "X" if ey == 1 else "Y"
        return "S"

    def code_space_check(self) -> bool:
        """After the final perfect round and correction: every surface-code stabilizer is deterministically +1."""
        self.flush()
        code = self.code
        if np.any(self.zchecks(np.asarray(code.zc_rb, np.int64))):
            return False
        gens = np.asarray(sorted(code.xg_rb), np.int64)
        S = self.tab.S[gens]
        L, ph = self.E.measured(S)
        for u in range(len(gens)):
            if self.sim.peek_observable_expectation(self._pauli(S[u], L[u], ph[u])) != 1:
                return False
        return True


# %% [Cell 6] Decoders: generic space-time MWPM and just-in-time (JIT) flux decoder
def _llr(p, cap=30.0):
    p = float(p)
    if p <= 0.0:
        return cap
    p = min(p, 0.499)
    return float(min(cap, max(1e-3, math.log((1.0 - p) / p))))


class Graph:
    """Graph-like error model of one component: nodes = checks, edges = qubits (1 or 2 incident checks)."""

    def __init__(self, check_qubits: List[Sequence[int]], qubits: Sequence[int]):
        self.nn = len(check_qubits)
        qpos = {int(q): i for i, q in enumerate(qubits)}
        inc = [[] for _ in qubits]
        for c, qs in enumerate(check_qubits):
            for q in qs:
                if int(q) in qpos:
                    inc[qpos[int(q)]].append(c)
        for i, lst in enumerate(inc):
            if len(lst) == 0 or len(lst) > 2:
                raise AssertionError(f"qubit {qubits[i]} has {len(lst)} incident checks (not graph-like)")
        self.qubits = np.asarray(qubits, np.int32)
        self.inc = inc
        self.nq = len(qubits)
        # dedupe parallel edges (pymatching drops fault ids of merged edges)
        seen = {}
        self.edges = []  # (u, v or -1, local qubit index)
        for i, lst in enumerate(inc):
            key = tuple(sorted(lst)) if len(lst) == 2 else (lst[0], -1)
            if key in seen:
                continue
            seen[key] = i
            self.edges.append((key[0], key[1], i))
        self.Hc = np.zeros((self.nn, self.nq), np.uint8)
        for i, lst in enumerate(inc):
            for c in lst:
                self.Hc[c, i] ^= 1

    def syndrome_of(self, local_bits: np.ndarray) -> np.ndarray:
        return (self.Hc.astype(np.int64) @ local_bits.astype(np.int64)) % 2


def build_st_matching(gr: Graph, layers: int, w_s: float, w_t: float, past_boundary: bool = False,
                      future_boundary: Optional[float] = None, future_ids: bool = False):
    """Space-time matching graph. Node (layer, check) -> layer*nn + check.
    Fault ids: local qubit index for every space-like edge (any layer); nq + check for future-boundary edges."""
    m = pymatching.Matching()
    nn = gr.nn
    for L in range(layers):
        base = L * nn
        for u, v, qi in gr.edges:
            if v < 0:
                m.add_boundary_edge(base + u, fault_ids={qi}, weight=w_s, merge_strategy="disallow")
            else:
                m.add_edge(base + u, base + v, fault_ids={qi}, weight=w_s, merge_strategy="disallow")
        if L + 1 < layers:
            for c in range(nn):
                m.add_edge(base + c, base + nn + c, weight=w_t, merge_strategy="disallow")
    # time boundaries via virtual nodes (never detection events) so that they are distinguishable from the
    # spatial boundary:  node -- virtual (weight w) -- boundary (weight 0)
    nxt = layers * nn
    if past_boundary:
        for c in range(nn):
            m.add_edge(c, nxt, weight=w_t, merge_strategy="disallow")
            m.add_boundary_edge(nxt, weight=0.0, merge_strategy="disallow")
            nxt += 1
    if future_boundary is not None:
        base = (layers - 1) * nn
        for c in range(nn):
            fid = {gr.nq + c} if future_ids else set()
            m.add_edge(base + c, nxt, fault_ids=fid, weight=max(1e-6, future_boundary), merge_strategy="disallow")
            m.add_boundary_edge(nxt, weight=0.0, merge_strategy="disallow")
            nxt += 1
    return m


def decode_ids(m, det: np.ndarray, n_ids: int) -> np.ndarray:
    if not np.any(det):
        return np.zeros(n_ids, np.uint8)
    full = np.zeros(m.num_detectors, np.uint8)
    full[:det.shape[0]] = det
    out = np.asarray(m.decode(full), np.uint8)
    if out.shape[0] < n_ids:
        out = np.concatenate([out, np.zeros(n_ids - out.shape[0], np.uint8)])
    return out[:n_ids]


class JITDecoder:
    """Just-in-time flux decoder (v7 algorithm, generalised to planar graphs with spatial boundaries).

    Round t: corrected syndrome sigma'_t = sigma_t xor (flux of corrections so far); detection events = changes;
    MWPM over layers 0..t with a 'future' boundary on layer t (cost offset*w_t); every matched pair / spatial-boundary
    match is corrected immediately (X on the spatial edges of the matching); events matched to the future stay pending.
    """

    def __init__(self, gr: Graph, rounds: int, w_s: float, w_t: float, offset: float):
        self.gr = gr
        self.R = rounds
        self.w_t = w_t
        self.matchers = [build_st_matching(gr, t + 1, w_s, w_t, future_boundary=offset * w_t, future_ids=True)
                         for t in range(rounds)]
        self.n_ids = gr.nq + gr.nn

    def new_shot(self):
        self.pending: List[Tuple[int, int]] = []
        self.prev = np.zeros(self.gr.nn, np.uint8)
        self.corr = np.zeros(self.gr.nn, np.uint8)

    def step(self, syn: np.ndarray, t: int) -> np.ndarray:
        gr = self.gr
        s = (syn ^ self.corr).astype(np.uint8)
        ev = np.nonzero(s ^ self.prev)[0]
        self.prev = s
        self.pending.extend((int(c), t) for c in ev)
        if not self.pending:
            return np.zeros(0, np.int32)
        det = np.zeros((t + 1) * gr.nn, np.uint8)
        for c, tau in self.pending:
            det[tau * gr.nn + c] ^= 1
        ids = decode_ids(self.matchers[t], det, self.n_ids)
        flips = np.nonzero(ids[:gr.nq])[0]
        fut = ids[gr.nq:]
        newp = []
        latest = {}
        for c, tau in self.pending:
            if fut[c]:
                latest[c] = max(latest.get(c, -1), tau)
        for c in np.nonzero(fut)[0]:
            newp.append((int(c), latest.get(int(c), t)))
        self.pending = newp
        if flips.size:
            self.corr ^= gr.syndrome_of(np.bincount(flips, minlength=gr.nq) % 2).astype(np.uint8)
        return gr.qubits[flips]


class Decoders:
    """All decoders of one (code, p, rounds) point."""

    def __init__(self, code: TriangleCode, p: float, R_in: int, R_out: int, cfg: dict):
        self.code = code
        zq = [z[2] for z in code.zchecks]
        self.flux_rb = Graph([zq[i] for i in code.zc_rb], code.rb_qubits)
        self.flux_g = Graph([zq[i] for i in code.zc_g], code.g_qubits)
        gq = [g.supp for g in code.gens]
        self.chg_rb = Graph([gq[i] for i in code.xg_rb], code.rb_qubits)
        self.chg_g = Graph([gq[i] for i in code.xg_g], code.g_qubits)
        px = 2.0 * p / 3.0
        pm = p
        self.w_s = _llr(px)
        self.w_t = _llr(pm)
        off = float(cfg.get("jit_boundary_offset", 1.0))
        self.jit_rb = JITDecoder(self.flux_rb, R_in, self.w_s, self.w_t, off)
        self.jit_g = JITDecoder(self.flux_g, R_in, self.w_s, self.w_t, off)
        ideal = cfg.get("switch_in_boundary", "ideal") == "ideal"
        self.ideal = ideal
        Lin = R_in + (1 if ideal else 0)
        # switch-in charge clean-up
        self.cin_rb = build_st_matching(self.chg_rb, Lin, self.w_s, self.w_t,
                                        future_boundary=None if ideal else self.w_t)
        if not ideal and R_in >= 2:
            self.cin_g = build_st_matching(self.chg_g, R_in - 1, self.w_s, self.w_t, past_boundary=True,
                                           future_boundary=self.w_t, future_ids=True)
        else:
            self.cin_g = None
        # single-layer spatial matchers (perfect syndromes)
        self.sp_flux_rb = build_st_matching(self.flux_rb, 1, 1.0, 1.0)
        self.sp_flux_g = build_st_matching(self.flux_g, 1, 1.0, 1.0)
        self.sp_chg_rb = build_st_matching(self.chg_rb, 1, 1.0, 1.0)
        self.sp_chg_g = build_st_matching(self.chg_g, 1, 1.0, 1.0)
        # switch-out (R_out noisy rounds + 1 perfect round)
        self.out_flux = build_st_matching(self.flux_rb, R_out + 1, self.w_s, self.w_t)
        self.out_chg = build_st_matching(self.chg_rb, R_out + 1, self.w_s, self.w_t)
        # green generators: solve  sum lambda_v supp_v = m'  on green qubits
        S = code.gen_supp_mat[code.xg_g][:, code.g_qubits]  # (ng, nqg)
        if gf2_rank(S) != S.shape[0]:
            raise AssertionError("green generators are not independent")
        self.green_left = gf2_left_inverse(S.T.copy())   # lambda = P m'
        self.green_S = S

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _events(rec: np.ndarray, ref0: bool = True) -> np.ndarray:
        """rec (layers, nn) -> detection events (layers*nn,), layer 0 compared with 0 if ref0."""
        ev = rec.copy()
        ev[1:] ^= rec[:-1]
        if not ref0:
            ev = ev[1:]
        return ev.reshape(-1)

    def pair_spatial(self, m, gr: Graph, syn: np.ndarray) -> np.ndarray:
        ids = decode_ids(m, syn.astype(np.uint8), gr.nq)
        return gr.qubits[np.nonzero(ids)[0]]

    def decode_st(self, m, gr: Graph, det: np.ndarray, n_extra: int = 0):
        ids = decode_ids(m, det, gr.nq + n_extra)
        return gr.qubits[np.nonzero(ids[:gr.nq])[0]], ids[gr.nq:]

    def green_lambda(self, mprime_g: np.ndarray) -> np.ndarray:
        lam = (self.green_left.astype(np.int64) @ mprime_g.astype(np.int64)) % 2
        if not np.array_equal((self.green_S.T.astype(np.int64) @ lam) % 2, mprime_g):
            raise AssertionError("green outcome pattern is not a coboundary after decoding")
        return np.asarray(self.code.xg_g)[np.nonzero(lam)[0]]



# %% [Cell 7] Protocol (one shot): switch-in (JIT) -> [ideal U] -> switch-out -> surface-code rounds -> logical class
STATE_CLASSES = ("I", "X", "Y", "Z", "S", "collapsed", "flux_D4")
STATE_INFIDELITY_T = {"I": 0.0, "Z": 1.0, "X": 0.5, "Y": 0.5, "S": 0.5, "collapsed": 0.5}
STATE_INFIDELITY_PLUS = {"I": 0.0, "Z": 1.0, "X": 0.0, "Y": 1.0, "S": 0.5, "collapsed": 0.5}


class ShotRunner:
    """One shot of the protocol.  Identical sequence of operations and decoders as v1; only the backend changed.

    Backend interface: reset(seed), apply_x, apply_z, zchecks, measure_gens(gen_idx, bare, rbits) -> handle,
    result(handle), measure_green_switch_out(gq, rbits), apply_gen_unitaries, apply_U, x_frame,
    evaluate_class(basis).  rbits (random outcome bits) are drawn only for lock-step validation backends."""

    def __init__(self, code: TriangleCode, p: float, cfg: dict):
        self.code = code
        self.p = float(p)
        self.cfg = cfg
        d = code.d
        self.R_in = int(cfg["rounds_in"]) if cfg.get("rounds_in") is not None else d
        self.R_out = int(cfg["rounds_out"]) if cfg.get("rounds_out") is not None else d
        self.dec = Decoders(code, p, self.R_in, self.R_out, cfg)
        order = str(cfg.get("x_check_order", "rgb"))
        self.x_order = np.array(sum([code.order_by_color[c] for c in order], []), np.int64)
        self.sc_order = np.array(sorted(code.xg_rb), np.int64)
        self.all_q = np.arange(code.nq, dtype=np.int64)
        self.zc_all = np.arange(len(code.zchecks), dtype=np.int64)
        self.zc_rb = np.array(code.zc_rb, np.int64)
        self.pos_x = {int(g): i for i, g in enumerate(self.x_order)}
        self.rb_pos = np.array([self.pos_x[g] for g in code.xg_rb], np.int64)
        self.g_pos = np.array([self.pos_x[g] for g in code.xg_g], np.int64)
        self.sc_pos = {int(g): i for i, g in enumerate(self.sc_order)}
        self.sc_rb_pos = np.array([self.sc_pos[g] for g in code.xg_rb], np.int64)
        zpos = {int(c): i for i, c in enumerate(self.zc_all)}
        self.zrb_pos = np.array([zpos[c] for c in code.zc_rb], np.int64)
        self.zg_pos = np.array([zpos[c] for c in code.zc_g], np.int64)

    # ---- noise
    def _depolarize(self, be, qubits: np.ndarray, rng):
        p = self.p
        r = rng.random(qubits.shape[0])
        typ = rng.integers(0, 3, qubits.shape[0])
        hit = r < p
        be.apply_x(qubits[hit & (typ != 2)])
        be.apply_z(qubits[hit & (typ != 0)])

    def _flip(self, out: np.ndarray, rng) -> np.ndarray:
        f = (rng.random(out.shape[0]) < self.p).astype(np.uint8)
        return out ^ f

    @staticmethod
    def _rbits(be, k: int, rng):
        return rng.integers(0, 2, k).astype(np.uint8) if getattr(be, "needs_rbits", False) else None

    # ---- one shot
    def run(self, be, case: str, rng) -> str:
        code, dec = self.code, self.dec
        be.reset(int(rng.integers(0, 2 ** 63 - 1)))
        R = self.R_in
        dec.jit_rb.new_shot()
        dec.jit_g.new_shot()
        nx = self.x_order.shape[0]
        Lin = R + (1 if dec.ideal else 0)
        handles = []
        # ===== switch-in: R noisy rounds (data noise, Z checks + JIT, dressed X checks r->g->b) =====
        for t in range(R):
            self._depolarize(be, self.all_q, rng)
            z = self._flip(be.zchecks(self.zc_all), rng)
            fr = dec.jit_rb.step(z[self.zrb_pos], t)
            fg = dec.jit_g.step(z[self.zg_pos], t)
            be.apply_x(np.concatenate([fr, fg]))
            handles.append(be.measure_gens(self.x_order, False, self._rbits(be, nx, rng)))
        # ===== end of switch-in =====
        if dec.ideal:
            z = be.zchecks(self.zc_all)
            cr = dec.pair_spatial(dec.sp_flux_rb, dec.flux_rb, z[self.zrb_pos])
            cg = dec.pair_spatial(dec.sp_flux_g, dec.flux_g, z[self.zg_pos])
            be.apply_x(np.concatenate([cr, cg]))
            handles.append(be.measure_gens(self.x_order, False, self._rbits(be, nx, rng)))
        xrec = np.zeros((Lin, nx), np.uint8)
        for t in range(Lin):
            m = be.result(handles[t])
            xrec[t] = m if t >= R else self._flip(m, rng)
        # charge clean-up (r,b: space-time MWPM -> net Z correction; g: estimate final syndrome, pair it)
        rec_rb = xrec[:, self.rb_pos]
        zc, _ = dec.decode_st(dec.cin_rb, dec.chg_rb, Decoders._events(rec_rb))
        rec_g = xrec[:, self.g_pos]
        if dec.ideal:
            sig_g = rec_g[-1]
        elif dec.cin_g is not None:
            ev = Decoders._events(rec_g, ref0=False)
            _, fut = dec.decode_st(dec.cin_g, dec.chg_g, ev, n_extra=dec.chg_g.nn)
            sig_g = rec_g[-1] ^ fut[:dec.chg_g.nn]
        else:
            sig_g = rec_g[-1]
        zg = dec.pair_spatial(dec.sp_chg_g, dec.chg_g, sig_g)
        be.apply_z(np.concatenate([zc, zg]))

        if case == "in_only":
            return self._evaluate_D4(be, rng)
        if case == "in_T_out":
            be.apply_U()
        elif case != "in_out":
            raise ValueError(case)
        # ===== switch-out: noise, single-shot Z^g measurement, green correction S^g_X(lambda) =====
        self._depolarize(be, self.all_q, rng)
        gq = code.g_qubits.astype(np.int64)
        mg = be.measure_green_switch_out(gq, self._rbits(be, gq.shape[0], rng))
        mg = self._flip(mg, rng)
        syn = dec.flux_g.syndrome_of(mg).astype(np.uint8)
        corr = dec.pair_spatial(dec.sp_flux_g, dec.flux_g, syn)
        mprime = mg.copy()
        gpos = {int(q): i for i, q in enumerate(gq)}
        for q in corr:
            mprime[gpos[int(q)]] ^= 1
        lam = dec.green_lambda(mprime)
        be.apply_gen_unitaries(lam)
        # ===== surface code (r, b): R_out noisy rounds + 1 perfect round, space-time MWPM =====
        Ro = self.R_out
        ns = self.sc_order.shape[0]
        zrec = np.zeros((Ro + 1, self.zc_rb.shape[0]), np.uint8)
        shandles = []
        noise_q = self.all_q if self.cfg.get("green_noise_after_switch_out", False) else code.rb_qubits.astype(np.int64)
        for t in range(Ro + 1):
            perfect = t == Ro
            if not perfect:
                self._depolarize(be, noise_q, rng)
            z = be.zchecks(self.zc_rb)
            zrec[t] = z if perfect else self._flip(z, rng)
            shandles.append(be.measure_gens(self.sc_order, True, self._rbits(be, ns, rng)))
        srec = np.zeros((Ro + 1, ns), np.uint8)
        for t in range(Ro + 1):
            m = be.result(shandles[t])
            srec[t] = m if t == Ro else self._flip(m, rng)
        xc, _ = dec.decode_st(dec.out_flux, dec.flux_rb, Decoders._events(zrec))
        be.apply_x(xc)
        zc2, _ = dec.decode_st(dec.out_chg, dec.chg_rb, Decoders._events(srec[:, self.sc_rb_pos]))
        be.apply_z(zc2)
        return be.evaluate_class("SC")

    def _evaluate_D4(self, be, rng) -> str:
        """Idealised read-out of the D4 code (same as v1): perfect Z checks, flux closing, removal of the X frame by
        dressed generators, perfect dressed X round, charge pairing, fictitious logical measurement."""
        code, dec = self.code, self.dec
        z = be.zchecks(self.zc_all)
        cr = dec.pair_spatial(dec.sp_flux_rb, dec.flux_rb, z[self.zrb_pos])
        cg = dec.pair_spatial(dec.sp_flux_g, dec.flux_g, z[self.zg_pos])
        be.apply_x(np.concatenate([cr, cg]))
        x = be.x_frame()
        u = (code.M0plus.astype(np.int64) @ x) % 2
        if not np.array_equal((code.M0.astype(np.int64) @ u) % 2, x):
            raise AssertionError("X frame not closed after flux closing")
        if u[code.kh]:
            return "flux_D4"
        gens = [code.basis_gens[c] for c in np.nonzero(u[:code.kh])[0]]
        be.apply_gen_unitaries(gens)
        if np.any(be.x_frame()):
            raise AssertionError("X frame not removed")
        nx = self.x_order.shape[0]
        m = be.result(be.measure_gens(self.x_order, False, self._rbits(be, nx, rng)))
        zr = dec.pair_spatial(dec.sp_chg_rb, dec.chg_rb, m[self.rb_pos])
        zg = dec.pair_spatial(dec.sp_chg_g, dec.chg_g, m[self.g_pos])
        be.apply_z(np.concatenate([zr, zg]))
        return be.evaluate_class("D4")


# %% [Cell 8] Sweep, statistics, plots, self-checks
def resolve_case(c: str) -> str:
    if c not in CASE_ALIASES:
        raise ValueError(f"unknown case {c}")
    return CASE_ALIASES[c]


def agresti_coull(n: int, k: int, alpha: float = 0.05) -> Tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    z = _norm.ppf(1 - alpha / 2) if _SCIPY else 1.96
    nt = n + z * z
    pt = (k + z * z / 2) / nt
    se = math.sqrt(pt * (1 - pt) / nt)
    return pt, max(0.0, pt - z * se), min(1.0, pt + z * se)


_GEOM_CACHE: Dict[int, Tuple[TriangleCode, EngineTables]] = {}


def get_code(d: int) -> Tuple[TriangleCode, EngineTables]:
    if d not in _GEOM_CACHE:
        code = TriangleCode(d)
        _GEOM_CACHE[d] = (code, EngineTables(code))
    return _GEOM_CACHE[d]


def run_point(case: str, d: int, p: float, shots: int, cfg: dict = None) -> Dict[str, object]:
    cfg = CFG if cfg is None else cfg
    case = resolve_case(case)
    code, tab = get_code(d)
    runner = ShotRunner(code, p, cfg)
    be = ProxyBackend(tab)
    seed0 = int(np.random.SeedSequence([int(cfg.get("seed", 0)), int(d), int(round(p * 1e12)),
                                        len(case) * 7 + ord(case[-1])]).generate_state(1)[0])
    rng = np.random.default_rng(seed0)
    counts = {c: 0 for c in STATE_CLASSES}
    t0 = time.time()
    for _ in range(shots):
        counts[runner.run(be, case, rng)] += 1
    dt = time.time() - t0
    n = shots
    alpha = 1.0 - float(cfg.get("confidence_level", 0.95))
    nI = counts["I"]
    tab_inf = STATE_INFIDELITY_T if case == "in_T_out" else STATE_INFIDELITY_PLUS
    n_eval = n - counts["flux_D4"]
    infid = sum(tab_inf[c] * counts[c] for c in tab_inf) / max(1, n_eval)
    out = dict(case=case, d=d, rounds_in=runner.R_in, rounds_out=runner.R_out, p=p, shots=n,
               switch_in_boundary=cfg.get("switch_in_boundary", "ideal"), confidence_level=1.0 - alpha,
               conditions=condition_key(cfg))
    k_fail = n - nI
    _, lo, hi = agresti_coull(n, k_fail, alpha)
    out.update(P_L=k_fail / n, P_L_lower=lo, P_L_upper=hi, P_I=nI / n)
    for c in STATE_CLASSES[1:]:
        out[f"P_{c}"] = counts[c] / n
    out["mean_infidelity_excl_fluxD4"] = infid  # conditional: over shots whose logical state was evaluated
    out["P_L_excl_fluxD4"] = (n_eval - nI) / max(1, n_eval)
    for c in STATE_CLASSES:
        out[f"n_{c}"] = counts[c]
    out.update(sec_per_shot=dt / max(1, n), kappa_U=int(tab.kappa), seed=seed0, engine="pauli_proxy_stim",
               code_version=CODE_VERSION)
    return out


def condition_key(cfg: dict) -> str:
    """Canonical string of everything that defines the computed quantity (used to guard resume)."""
    key = {k: cfg.get(k) for k in RUN_CONDITION_KEYS}
    key["code_version"] = CODE_VERSION
    return json.dumps(key, sort_keys=True, default=str)


def estimate_crossings(df: pd.DataFrame, col: str = "P_L", n_boot: int = 300, seed: int = 0) -> List[Dict]:
    """Pairwise crossings of neighbouring d (log-log interpolation; Jeffreys resampling). Descriptive only."""
    rng = np.random.default_rng(seed)
    rows = []
    ds = sorted(df["d"].unique())
    for a, b in zip(ds[:-1], ds[1:]):
        u = df[df["d"] == a].sort_values("p")
        v = df[df["d"] == b].sort_values("p")
        if len(u) != len(v) or not np.allclose(u["p"].to_numpy(), v["p"].to_numpy()):
            continue
        p = u["p"].to_numpy()

        def roots(ya, yb):
            y = np.log(np.maximum(ya, 1e-12)) - np.log(np.maximum(yb, 1e-12))
            out = []
            for i in range(len(p) - 1):
                if y[i] * y[i + 1] < 0:
                    lp = np.log(p[i]) - y[i] * (np.log(p[i + 1]) - np.log(p[i])) / (y[i + 1] - y[i])
                    out.append(float(np.exp(lp)))
            return out

        est = roots(u[col].to_numpy(), v[col].to_numpy())
        draws = []
        na, nb = u["shots"].to_numpy(), v["shots"].to_numpy()
        ka, kb = np.round(u[col].to_numpy() * na), np.round(v[col].to_numpy() * nb)
        for _ in range(n_boot):
            r = roots(rng.beta(ka + 0.5, na - ka + 0.5), rng.beta(kb + 0.5, nb - kb + 0.5))
            if len(r) == 1:
                draws.append(r[0])
        ci = [float(q) for q in np.quantile(draws, [0.025, 0.975])] if draws else None
        rows.append(dict(d_a=int(a), d_b=int(b), crossings=est, ci95=ci, unique_draws=len(draws)))
    return rows


CASE_TITLES = {"in_only": "switch-in only (D4)", "in_out": "switch-in → switch-out",
               "in_T_out": r"switch-in → ideal $\overline{T}^\dagger$ → switch-out"}


def plot_case(df: pd.DataFrame, case: str, path_png: str, save_pdf: bool = True, col: str = "P_L"):
    """P_L(p) per d with its Agresti-Coull interval (other columns are drawn without error bars)."""
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    pal = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    for i, d in enumerate(sorted(df["d"].unique())):
        sub = df[df["d"] == d].sort_values("p")
        y = sub[col].to_numpy().astype(float)
        y = np.where(y > 0, y, np.nan)  # zero-failure points are not drawn (see CSV)
        if col == "P_L":
            yerr = [np.nan_to_num(np.maximum(0, y - sub["P_L_lower"].to_numpy())),
                    np.nan_to_num(np.maximum(0, sub["P_L_upper"].to_numpy() - y))]
        else:
            yerr = None
        ax.errorbar(sub["p"], y, yerr=yerr, fmt="-o", capsize=3, color=pal[i % len(pal)], label=f"d={int(d)}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"physical error rate $p$ (phenomenological)", fontsize=13)
    lab = {"P_L": r"$P_L = 1-P_I$"}
    ax.set_ylabel(lab.get(col, col), fontsize=13)
    ax.set_title(CASE_TITLES.get(case, case) + f"  [{df['switch_in_boundary'].iloc[0]}]", fontsize=13)
    ax.grid(True, which="major", alpha=0.6)
    ax.grid(True, which="minor", ls=":", alpha=0.3)
    ax.legend()
    shots = sorted(set(int(v) for v in df["shots"]))
    lvl = float(df["confidence_level"].iloc[0]) if "confidence_level" in df else 0.95
    ax.text(0.98, 0.02, f"{'/'.join(map(str, shots))} shots/pt, {100 * lvl:.0f}% AC CI, "
                        f"boundary={df['switch_in_boundary'].iloc[0]}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9, alpha=0.6)
    fig.tight_layout()
    fig.savefig(path_png, dpi=180)
    if save_pdf:
        fig.savefig(os.path.splitext(path_png)[0] + ".pdf")
    plt.close(fig)


def plot_frames(df: pd.DataFrame, case: str, path_png: str):
    """Composition of the final logical state (X, Y, Z, S-class, collapsed, flux_D4) per d."""
    ds = sorted(df["d"].unique())
    fig, axes = plt.subplots(1, len(ds), figsize=(5.2 * len(ds), 4.2), squeeze=False)
    cols = [("P_X", "X"), ("P_Y", "Y"), ("P_Z", "Z"), ("P_S", "S-type (|+-i>)"),
            ("P_collapsed", "collapsed"), ("P_flux_D4", "flux (D4, not evaluated)")]
    for ax, d in zip(axes[0], ds):
        sub = df[df["d"] == d].sort_values("p")
        for c, lab in cols:
            y = sub[c].to_numpy()
            if np.any(y > 0):
                ax.plot(sub["p"], np.where(y > 0, y, np.nan), "-o", ms=3, label=lab)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"d={int(d)}")
        ax.set_xlabel("p")
        ax.grid(True, alpha=0.5)
    axes[0][0].set_ylabel("probability of final logical state class")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(CASE_TITLES.get(case, case))
    fig.tight_layout()
    fig.savefig(path_png, dpi=150)
    plt.close(fig)


def plot_all_cases(df: pd.DataFrame, path: str):
    cases = list(dict.fromkeys(df["case"]))
    fig, axes = plt.subplots(1, len(cases), figsize=(5.6 * len(cases), 4.6), sharey=True, squeeze=False)
    pal = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for ax, case in zip(axes[0], cases):
        sub0 = df[df["case"] == case]
        for i, d in enumerate(sorted(sub0["d"].unique())):
            sub = sub0[sub0["d"] == d].sort_values("p")
            y = sub["P_L"].to_numpy().astype(float)
            y = np.where(y > 0, y, np.nan)
            ax.errorbar(sub["p"], y, yerr=[np.nan_to_num(np.maximum(0, y - sub["P_L_lower"])),
                                           np.nan_to_num(np.maximum(0, sub["P_L_upper"] - y))],
                        fmt="-o", capsize=3, color=pal[i % len(pal)], label=f"d={int(d)}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(CASE_TITLES.get(case, case) + f"  [{sub0['switch_in_boundary'].iloc[0]}]")
        ax.set_xlabel("p")
        ax.grid(True, which="major", alpha=0.6)
    axes[0][0].set_ylabel(r"$P_L=1-P_I$")
    axes[0][0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_manifest(path: str, cfg: dict, extra: Optional[dict] = None):
    import datetime
    import importlib.metadata as md
    import platform
    vers = {}
    for pkg in ("numpy", "scipy", "pymatching", "stim", "pandas", "matplotlib"):
        try:
            vers[pkg] = md.version(pkg)
        except Exception:  # noqa: BLE001
            vers[pkg] = None
    rec = dict(code_version=CODE_VERSION, engine="pauli_proxy_stim", datetime=datetime.datetime.now().isoformat(),
               python=platform.python_version(), platform=platform.platform(), versions=vers,
               CFG={k: v for k, v in cfg.items()})
    if extra:
        rec.update(extra)
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=2, ensure_ascii=False, default=str)


def run_self_checks(cfg: dict = None) -> Dict[str, Dict[str, object]]:
    """For EVERY d of cfg["d_values"]: geometry, flat commutation, U = logical T^dagger, exactness conditions C1-C3
    of the Pauli-proxy reduction (random flat configurations: an implementation check, not a proof -- README_v3 Sec. 4),
    p=0 runs of all cases, final state in the code space.  Raises AssertionError on the first failure."""
    cfg = CFG if cfg is None else cfg
    shots = int(cfg.get("self_check_shots", 100))
    res = {}
    for d in sorted(int(v) for v in cfg["d_values"]):
        out = {}
        code, tab = get_code(d)
        op_nc, flat_nc = check_flat_commutation(code)
        out["noncommuting_pairs(operator, flat)"] = (op_nc, flat_nc)
        if flat_nc:
            raise AssertionError(f"d={d}: dressed generators must commute on the flat subspace")
        th = check_theorem_pointwise(code)
        out["U_logical"] = th["logical_gate"]
        if th["violations"] or th["kappa"] != 7:
            raise AssertionError(f"d={d}: U is not the logical T^dagger on this lattice: {th}")
        ic = check_intertwiner_conditions(code)
        out["C1,C2,C3 violations"] = (ic["C1_violations"], ic["C2_violations"], ic["C3_violations"])
        if ic["C1_violations"] or ic["C2_violations"] or ic["C3_violations"]:
            raise AssertionError(f"d={d}: Pauli-proxy exactness conditions violated: {ic}")
        zero = {}
        for case in ("in_only", "in_out", "in_T_out"):
            r = run_point(case, d, 0.0, shots, cfg)
            zero[case] = r["P_L"]
            if r["P_L"] != 0:
                raise AssertionError(f"d={d}: p=0 must give no logical error ({case})")
        out["p=0 P_L"] = zero
        ok = True
        rng = np.random.default_rng(5)
        for case in ("in_out", "in_T_out"):
            runner = ShotRunner(code, 0.02, cfg)
            be = ProxyBackend(tab)
            for _ in range(10):
                runner.run(be, case, rng)
                ok &= be.code_space_check()
        out["final state in code space"] = bool(ok)
        if not ok:
            raise AssertionError(f"d={d}: final state not in the code space")
        res[f"d={d}"] = out
    return res


def _load_csv(csv: str) -> pd.DataFrame:
    if os.path.exists(csv):
        return pd.read_csv(csv)
    return pd.DataFrame()


def _check_resume(old: pd.DataFrame, cond: str, csv: str):
    """A CSV is only extended if every row was computed under exactly the same conditions."""
    if not len(old):
        return
    if "conditions" not in old or not (old["conditions"] == cond).all():
        raise ValueError(f"{csv} contains results computed under other conditions (rounds, boundary, seed, decoder "
                         f"settings or code version).  Use another out_dir, or resume=False to overwrite it.\n"
                         f"  now : {cond}\n  file: {sorted(set(old.get('conditions', pd.Series(['(none)'])).astype(str)))}")


def run_experiments(cfg: dict = None) -> pd.DataFrame:
    cfg = CFG if cfg is None else cfg
    if pymatching is None:
        raise RuntimeError("pymatching is required")
    cases = [resolve_case(c) for c in (cfg.get("cases_to_run") or [cfg["case"]])]
    boundaries = list(cfg.get("boundaries_to_run") or [cfg.get("switch_in_boundary", "ideal")])
    for b in boundaries:
        if b not in ("ideal", "noisy"):
            raise ValueError(f"unknown switch_in_boundary {b}")
    ds = [int(d) for d in cfg["d_values"]]
    rec = RECOMMENDED_P_RANGE[cfg.get("noise_model", "phenomenological")]
    p_min = float(rec[0] if cfg.get("p_min") is None else cfg["p_min"])
    p_max = float(rec[1] if cfg.get("p_max") is None else cfg["p_max"])
    p_n = int(cfg["p_n"])
    ps = [p_min] if p_n == 1 else list(np.exp(np.linspace(np.log(p_min), np.log(p_max), p_n)))
    shots = int(cfg["shots"])
    out_dir = str(cfg.get("out_dir") or ".")
    os.makedirs(out_dir, exist_ok=True)
    if cfg.get("run_self_checks", True):
        print("[self-checks] (every d)")
        for dk, v in run_self_checks(cfg).items():
            print(f"  {dk}: {v}")
        print()
    for d in ds:
        code, tab = get_code(d)
        s = code.summary()
        print(f"[geometry d={d}] qubits={s['qubits']} Zchecks={s['z_checks']} Xgens={s['x_generators']} "
              f"{s['generators_by_kind']}  U -> omega^{tab.kappa} on |1> (7 = T^dagger)")
    frames = []
    for boundary in boundaries:
        bcfg = dict(cfg)
        bcfg["switch_in_boundary"] = boundary
        cond = condition_key(bcfg)
        for case in cases:
            prefix = os.path.join(out_dir, f"chsc_v3_{case}_{boundary}")
            csv = prefix + ".csv"
            old = _load_csv(csv)
            if cfg.get("resume", True):
                _check_resume(old, cond, csv)       # raises before anything is written
            else:
                if len(old):
                    print(f"  resume=False: {csv} will be overwritten")
                old = pd.DataFrame()
            if cfg.get("save_manifest", True):
                save_manifest(prefix + "_manifest.json", bcfg, dict(case=case, p_values=ps, conditions=cond))
            rows = old.to_dict("records") if len(old) else []
            print(f"\n=== case={case}  boundary={boundary}  shots={shots}  -> {csv} ===")
            t0 = time.time()
            for d in ds:
                for p in ps:
                    if len(old):
                        same = old[(old["d"] == d) & np.isclose(old["p"], p)]
                        if len(same) and (same["shots"] == shots).all():
                            print(f"  d={d} p={p:.4e}  (already in CSV under the same conditions, skipped)")
                            continue
                        if len(same):
                            raise ValueError(f"{csv}: d={d} p={p:.4e} exists with shots={sorted(set(same['shots']))}, "
                                             f"now shots={shots}.  The same seed would re-draw the same samples, so they "
                                             f"cannot be added; use another out_dir (and another seed) or resume=False.")
                    r = run_point(case, d, float(p), shots, bcfg)
                    rows.append(r)
                    pd.DataFrame(rows).sort_values(["d", "p"]).to_csv(csv, index=False)
                    print(f"  d={d} p={p:.4e}  P_L={r['P_L']:.5f} [{r['P_L_lower']:.5f},{r['P_L_upper']:.5f}]  "
                          f"X={r['P_X']:.4f} Y={r['P_Y']:.4f} Z={r['P_Z']:.4f} S={r['P_S']:.4f} "
                          f"col={r['P_collapsed']:.4f} fluxD4={r['P_flux_D4']:.4f}  ({r['sec_per_shot'] * 1e3:.1f} ms/shot)")
            # everything in the CSV (also d values of earlier calls) has the same conditions -> plot all of it
            df = pd.DataFrame(rows).sort_values(["d", "p"]).reset_index(drop=True)
            df.to_csv(csv, index=False)
            print(f"  done in {time.time() - t0:.1f} s ; saved {csv}  (d in file: {sorted(df['d'].unique())})")
            plot_case(df, case, prefix + ".png", bool(cfg.get("save_pdf", True)))
            plot_frames(df, case, prefix + "_frames.png")
            cr = estimate_crossings(df, "P_L", seed=int(cfg.get("seed", 0)))
            pd.DataFrame(cr).to_csv(prefix + "_crossings.csv", index=False)
            for row in cr:
                print(f"  crossing d={row['d_a']}-{row['d_b']}: {['%.4e' % c for c in row['crossings']]}  95%: {row['ci95']}")
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    for boundary in boundaries:
        sub = out[out["switch_in_boundary"] == boundary]
        if sub["case"].nunique() > 1:
            plot_all_cases(sub, os.path.join(out_dir, f"chsc_v3_all_cases_{boundary}.png"))
    if len(boundaries) > 1:
        plot_boundaries(out, os.path.join(out_dir, "chsc_v3_ideal_vs_noisy.png"))
    return out


def plot_boundaries(df: pd.DataFrame, path: str):
    """ideal vs noisy switch-in termination, one panel per case (solid: ideal, dashed: noisy)."""
    cases = list(dict.fromkeys(df["case"]))
    fig, axes = plt.subplots(1, len(cases), figsize=(5.6 * len(cases), 4.6), sharey=True, squeeze=False)
    pal = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for ax, case in zip(axes[0], cases):
        sub0 = df[df["case"] == case]
        for i, d in enumerate(sorted(sub0["d"].unique())):
            for b, ls in (("ideal", "-o"), ("noisy", "--s")):
                sub = sub0[(sub0["d"] == d) & (sub0["switch_in_boundary"] == b)].sort_values("p")
                if len(sub):
                    ax.plot(sub["p"], sub["P_L"].where(sub["P_L"] > 0), ls, ms=4, color=pal[i % len(pal)],
                            label=f"d={int(d)} {b}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(CASE_TITLES.get(case, case))
        ax.set_xlabel("p")
        ax.grid(True, which="major", alpha=0.6)
    axes[0][0].set_ylabel(r"$P_L=1-P_I$")
    axes[0][0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    results = run_experiments()
