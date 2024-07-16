"""
A collection of linear algebra operations that operate on the last N-1 operations of N-dimensional
numpy arrays. See corsid.differentiable for the jax-based versions of these.
"""
import numpy as np


def batch_mmip(A, B):
    """ Matrix-matrix inner product A @ B, looped over pixels in axis 0 """
    return np.einsum('...ij,...jk->...ik', A, B)


def batch_mmop(A, B):
    """
    Matrix-matrix "outer" product A @ B.T, looped over pixels in axis 0.

    The term "outer product" is used quite loosely here; the idea is generalizing the outer product
    of two vectors, a @ b.T, to matrices, A @ B.T, i.e., we just transpose the second argument and
    perform matrix multiplication.
    """
    return np.einsum('...ij,...kj->...ik', A, B)


def batch_tr_mmip(A, B):
    """ trace(A @ B), looped over pixels in axis 0 """
    return np.einsum('...ij,...jk->...', A, B)


def batch_mvip(A, x):
    """ Matrix-vector inner product A @ x, looped over pixels in axis 0 """
    return np.einsum('...ij,...j->...i', A, x)


def batch_vvip(x, y):
    """ Vector-vector inner product x @ y, looped over pixels in axis 0 """
    return np.einsum('...i,...i->...', x, y)


def batch_vvop(x, y):
    """ Vector-vector outer product x @ y.T, looped over pixels in axis 0 """
    return np.einsum('...i,...j->...ij', x, y)


def batch_mt(A):
    """ Matrix transpose A.T, looped over pixels in axis 0 """
    return np.einsum('...ij->...ji', A)


def batch_ABAT(A, B):
    """ Matrix product of form A @ B @ A.T, looped over pixels in axis 0"""
    return np.einsum('...ij,...jk,...lk->...il', A, B, A)


def eye(num_pix, N):
    """ Convenience function for generating a stack of identity matrices """
    return np.stack([np.eye(N) for _ in range(num_pix)])

