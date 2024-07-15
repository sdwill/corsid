"""
Forward modeling components for building differentiable models using jax.

TODO: Implement something clever with importlib to swap between numpy/jax on the fly
"""
from typing import List

import jax.numpy as np
from numpy.typing import NDArray


def l2sqr(x: NDArray):
    """
    Squared L2 norm. vdot() handles complex numbers correctly whereas dot() and inner() do not.
    """
    return np.vdot(x, x).real


def weighted_mean(arrs: List[NDArray], weights: List[np.float64]) -> NDArray:
    """ (sum_k weights[k] * arrs[k]) / sum(weights) """
    return np.einsum('i...,i->...', np.stack(arrs), np.asarray(weights)) / np.asarray(weights).sum()


def make_state_from_field(E_field):
    L = 2  # Length of state vector
    num_pix = np.size(E_field)

    # Gu[k, 0] = Re{control_field[k]}
    # Gu[k, 1] = Im{control_field[k]}
    # E.shape = (num_pix, 1); adding extra dimension allows hstack to (num_pix, 2)
    E = E_field[:, np.newaxis]
    return np.hstack([E.real, E.imag])  # shape = (num_pix, 2)


def batch_mmip(A, B):
    """ Matrix-matrix inner product, efficiently looped over pixels in axis 0 """
    return np.einsum('...ij,...jk->...ik', A, B)


def batch_mmop(A, B):
    """
    Matrix-matrix "outer" product A @ B.T, efficiently looped over pixels in axis 0.
    The term "outer product" is used quite loosely here; the idea is generalizing the outer product
    of two vectors, a @ b.T, to matrices, A @ B.T, i.e., we just transpose the second argument and
    perform matrix multiplicatin.
    """
    return np.einsum('...ij,...kj->...ik', A, B)


def batch_tr_mmip(A, B):
    return np.einsum('...ij,...jk->...', A, B)


def batch_mvip(A, x):
    """ Matrix-vector inner product, efficiently looped over pixels in axis 0 """
    return np.einsum('...ij,...j->...i', A, x)


def batch_vvip(x, y):
    """ Vector-vector inner product, efficiently looped over pixels in axis 0 """
    return np.einsum('...i,...i->...', x, y)


def batch_vvop(x, y):
    """ Vector-vector outer product, efficiently looped over pixels in axis 0 """
    return np.einsum('...i,...j->...ij', x, y)


def batch_mt(x):
    """ Matrix transpose, efficiently looped over pixels in axis 0 """
    return np.einsum('...ij->...ji', x)


def batch_ABAT(A, B):
    """ Matrix product of form A @ B @ A.T, looped over pixels in axis 0"""
    return np.einsum('...ij,...jk,...lk->...il', A, B, A)


def batch_tr(A):
    """ Matrix trace, looped over pixels in axis 0 """
    return np.einsum('...ii->...', A)
