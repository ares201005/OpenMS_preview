#
# @ 2023. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001
# for Los Alamos National Laboratory (LANL), which is operated by Triad
# National Security, LLC for the U.S. Department of Energy/National Nuclear
# Security Administration. All rights in the program are reserved by Triad
# National Security, LLC, and the U.S. Department of Energy/National Nuclear
# Security Administration. The Government is granted for itself and others acting
# on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this
# material to reproduce, prepare derivative works, distribute copies to the
# public, perform publicly and display publicly, and to permit others to do so.
#
# Author: Yu Zhang <zhy@lanl.gov>
#

r"""
Bosonic mixed Green's function with Fock-basis superposition trial & walker
==========================================================================

We use the (mixed) one-body Green's function for bosons

.. math::
    G_{ij} = \frac{\bra{\Psi_T}  b_i^\dagger b_j \ket{\phi_w}}{\bra{\Psi_T} \phi_w\rangle}.

When both trial and walker are expanded in the bosonic Fock basis:

.. math::
    |\Psi_T> = & \sum_n c_n \ket{n} \\
    |\phi_w> = & \sum_m a_m \ket{m}

Here :math:`n` and :math:`m` are occupation vectors :math:`(n_1,\cdots, n_M)` for :math:`M` boson modes/sites.

The Denominator :math:`D = \bra{\Psi_T} \phi_m\rangle` is:

.. math::
    D = \bra{\Psi_T}\phi\rangle = \sum_{nm} c_n^* a_m \bra{n} m\rangle = \sum_m c_m^* a_m.

i.e., it is an inner product of coefficient vectors.

Action of :math:`b_i^\dagger b_j` on a number state

.. math::
    b_j \ket{..., m_j, ...}  & = \sqrt{m_j} \ket{..., m_j - 1, ...} \\
    b_i^\dagger \ket{..., m_i, ...} & = \sqrt{m_i + 1} \ket{..., m_i + 1, ...}.

Hence:

.. math::
    b_j\ket{m} & = \sqrt{m_j} \ket{m - e_j}, \\
    b_i^\dagger \ket{m - e_j} & = \sqrt{m_i + 1} \ket{m - e_j + e_i}.

Therefore:

.. math::
    b_i^\dagger b_j \ket{m} = \sqrt{(m_i + 1)  m_j} \ket{m - e_j + e_i}


Compute the numerator :math:`N_{ij} = \bra{\Psi_T} b_i^\dagger b_j\ket{\phi_m}

.. math::
    N_{ij} &= \bra{\Psi_T} b_i^\dagger b_j \ket{\phi_m} = \sum_{mn} c_n^* a_m \bra{n} b_i^\dagger b_j \ket{m} \\
           &= \sum_{mn} c_n^* a_m \sqrt{(m_i + 1) m_j} \delta(n, m - e_j + e_i).

Thus the double sum collapses to :math:`N_{ij} = \sum_{m : m_j>0} a_m c_{m - e_j + e_i}^*  \sqrt{(m_i + 1) m_j}`.

Finally, the mixed estimator is :math:`G_{ij} = N_{ij} / D`.


Implementation notes
--------------------
- Store sparse states as dict: {occupation_tuple: complex_amplitude}.
- Complexity: O(nnz_walker * M^2) in the naive loops below.
  we can reduce it by looping only over j with m_j>0 and by using sparsity of c.
- If :math:`|D|` becomes tiny, the estimator is unstable (common AFQMC issue); handle with
  constraints / importance sampling or guard with eps.

Bosonic mixed Green's function with coherent-state for trial & walker
=====================================================================

**Single coherent-state trial and walker""
Let the boson sector trial and walker be multimode coherent states:

.. math::
    \ket{\Psi_T} = \ket{\beta} \\
    \ket{\phi_m} = \ket{\alpha}


where :math:`\alpha, \beta` are complex vectors of length M (M modes/sites), and

.. math::
    b_j \ket{\alpha} &= \alpha_j \ket{\alpha}\\
    \bra{\beta} b_i^\dagger &= \beta_i^* \bra{\beta}.

Then the mixed one-body Green's function is:

.. math::
    G_{ij} = \frac{\bra{\beta} b_i^\dagger b_j \ket{\alpha}}{\bra{\beta}\alpha\rangle}
           = \frac{\beta_i^* \alpha_j \bra{\beta}\alpha\rangle}{\bra{\beta}\alpha\rangle} = \beta_i^* \alpha_j

Note, the overlap in the denominator is
:math:`\bra{\beta}\alpha\rangle = \exp\left[-\frac{1}{2}(|\beta|^2 + |\alpha|^2 - 2\beta^* \alpha) \right]`.
where :math:`\beta^* \alpha = \sum_k \beta_k^* \alpha_k`.

So for coherent-state trial+walker, the mixed one-body GF is just the outer product: :math:`G = \beta^* \otimes \alpha`.


"""

from __future__ import annotations
import numpy as np
from math import sqrt
from typing import Dict, Tuple, Optional
from typing import Sequence

Occ = Tuple[int, ...]  # (n1,...,nM)

# using fock states

def mixed_boson_green_fock_superpositions(
    a_walker: Dict[Occ, complex],
    c_trial: Dict[Occ, complex],
    M: Optional[int] = None,
    eps: float = 0.0,
) -> np.ndarray:
    """
    Compute mixed bosonic green's function with both the trial and walker are Fock-basis superpositions:

    Parameters
    ----------
    a_walker : dict[Occ, complex]
        Sparse walker amplitudes a_m.
    c_trial : dict[Occ, complex]
        Sparse trial amplitudes c_n.
    M : int, optional
        Number of modes. If None, inferred from any key in a_walker or c_trial.
    eps : float
        Threshold for small denominator.

    Returns
    -------
    G : (M,M) complex ndarray
        Mixed one-body Green's function estimator.
    """

    if M is None:
        # infer M from any available key
        key = next(iter(a_walker.keys() or c_trial.keys()))
        M = len(key)

    # Denominator D = sum_m c_m^* a_m
    denom = 0.0 + 0.0j
    for m, a_m in a_walker.items():
        c_m = c_trial.get(m)
        if c_m is not None:
            denom += np.conjugate(c_m) * a_m

    if abs(denom) <= eps:
        raise ZeroDivisionError(
            f"<Psi_T|phi> too small: |denom|={abs(denom):.3e}. "
            "Mixed estimator unstable/undefined."
        )

    G = np.zeros((M, M), dtype=np.complex128)

    # Numerator:
    # N_ij = sum_{m : m_j>0} a_m * sqrt((m_i+1)*m_j) * c_{m - e_j + e_i}^*
    for m, a_m in a_walker.items():
        if a_m == 0:
            continue

        # only j where m_j>0 contribute
        for j in range(M):
            mj = m[j]
            if mj <= 0:
                continue

            # loop i for creation
            for i in range(M):
                # build n = m - e_j + e_i
                n_list = list(m)
                n_list[j] -= 1
                n_list[i] += 1
                n = tuple(n_list)

                c_n = c_trial.get(n)
                if c_n is None or c_n == 0:
                    continue

                mat_elem = sqrt((m[i] + 1) * mj)
                G[i, j] += a_m * mat_elem * np.conjugate(c_n)

    return G / denom


#=================================
# using coherent states
#=================================

def coherent_overlap(beta: np.ndarray, alpha: np.ndarray) -> complex:
    r"""
    Compute overlap for multimode coherent states.
    :math:`\bra{\beta}\alpha\rangle = exp( -1/2|\beta|^2 - 1/2|\alpha|^2 + \beta^* \alpha)`.
    """
    beta = np.asarray(beta, dtype=np.complex128)
    alpha = np.asarray(alpha, dtype=np.complex128)
    if beta.shape != alpha.shape:
        raise ValueError("beta and alpha must have the same shape (M,)")

    beta2 = np.vdot(beta, beta)   # sum beta^* beta = |beta|^2
    alpha2 = np.vdot(alpha, alpha)
    dot = np.vdot(beta, alpha)    # beta^* · alpha
    return np.exp(-0.5 * beta2 - 0.5 * alpha2 + dot)


def mixed_green_coherent(beta: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    r"""
    Compute :math:`G_{ij} = \bra{\beta} b_i^\dagger b_j \ket{\alpha} / \bra{\beta}\alpha\rangle = \beta_i^* \alpha_j`.

    Returns
    -------
    G : (M,M) complex ndarray
    """
    beta = np.asarray(beta, dtype=np.complex128)
    alpha = np.asarray(alpha, dtype=np.complex128)
    if beta.shape != alpha.shape:
        raise ValueError("beta and alpha must have the same shape (M,)")

    # Outer product: (beta^*)_i * alpha_j
    return np.outer(np.conjugate(beta), alpha)


def mixed_green_coherent_superpositions(
    betas: Sequence[np.ndarray],
    w: np.ndarray,
    alphas: Sequence[np.ndarray],
    v: np.ndarray,
    eps: float = 0.0,
) -> np.ndarray:
    r"""
    Optional: coherent-state superpositions

    .. math::
        \bra{\Psi_T} &= \sum_k w_k^* \bra{\beta_k} \\
        \ket{\phi_w} &= \sum_l v_l \ket{\alpha_l}

    Compute: :math:`G_{ij} = N_{ij} / D`

    where

    .. math::
        D &= \sum_{k,l} w_k^* v_l \bra{\beta_k}\alpha_l\rangle \\
        N_{ij} &= \sum_{k,l} w_k^* v_l \beta_{k,i}^* \alpha_{l,j} \bra{\beta_k}\alpha_l\rangle

    Parameters
    ----------
    betas : list of (M,) arrays
    w     : (K,) complex amplitudes for trial kets |beta_k> (bra uses w_k^*)
    alphas: list of (M,) arrays
    v     : (L,) complex amplitudes for walker kets |alpha_l>
    eps   : denominator threshold

    Returns
    -------
    G : (M,M) complex ndarray
    """
    w = np.asarray(w, dtype=np.complex128)
    v = np.asarray(v, dtype=np.complex128)
    K = len(betas)
    L = len(alphas)
    if w.shape != (K,) or v.shape != (L,):
        raise ValueError("w must be shape (K,) and v must be shape (L,)")

    M = np.asarray(betas[0]).shape[0]
    for b in betas:
        if np.asarray(b).shape != (M,):
            raise ValueError("All betas must have shape (M,)")
    for a in alphas:
        if np.asarray(a).shape != (M,):
            raise ValueError("All alphas must have shape (M,)")

    D = 0.0 + 0.0j
    N = np.zeros((M, M), dtype=np.complex128)

    for k in range(K):
        beta_k = np.asarray(betas[k], dtype=np.complex128)
        wk_conj = np.conjugate(w[k])
        for l in range(L):
            alpha_l = np.asarray(alphas[l], dtype=np.complex128)
            overlap = coherent_overlap(beta_k, alpha_l)
            weight = wk_conj * v[l] * overlap

            D += weight
            # beta_k^* alpha_l
            N += weight * np.outer(np.conjugate(beta_k), alpha_l)

    if abs(D) <= eps:
        raise ZeroDivisionError(f"<Psi_T|phi> too small: |D|={abs(D):.3e}")

    return N / D


# --- example usage ---
if __name__ == "__main__":
    # 1) 2-mode fock state example: occupations are (n1,n2)
    c_trial = {
        (1, 2): 1.0 + 0.0j,
        (2, 1): 0.2 - 0.1j,
        (0, 3): -0.3 + 0.05j,
    }

    a_walker = {
        (1, 2): 0.7 + 0.0j,
        (1, 1): 0.1 + 0.2j,
        (0, 3): -0.05 + 0.0j,
    }

    G = mixed_boson_green_fock_superpositions(a_walker, c_trial)
    print("G =\n", G)

    # 2) Single coherent-state trial and walker
    beta = np.array([0.3 + 0.1j, -0.2 + 0.0j], dtype=np.complex128)
    alpha = np.array([0.5 - 0.4j,  0.1 + 0.2j], dtype=np.complex128)

    S = coherent_overlap(beta, alpha)
    G = mixed_green_coherent(beta, alpha)

    print("<beta|alpha> =", S)
    print("G =\n", G)

    # Superpositions:
    betas = [beta, np.array([0.0 + 0.2j, 0.1 + 0.0j])]
    w = np.array([1.0 + 0.0j, 0.3 - 0.1j])
    alphas = [alpha, np.array([0.2 + 0.0j, -0.1 + 0.1j])]
    v = np.array([0.7 + 0.0j, 0.2 + 0.4j])

    Gmix = mixed_green_coherent_superpositions(betas, w, alphas, v, eps=1e-14)
    print("G (superpositions) =\n", Gmix)
