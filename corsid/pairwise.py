import logging
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from typing import Dict
from scipy.optimize import minimize
from tqdm import tqdm

from corsid import util, batch_linalg as bl, differentiable as d
from corsid.adam import AdamOptimizer
from corsid.TrainingData import TrainingData

log = logging.getLogger(__name__)
jax.config.update('jax_enable_x64', True)


def eval_dx_error(G, xs, bs):
    """
    Evaluate the error between 2*G*b[k] and x[2*j] - x[2*j] in each time step.
    This measures the consistency between the Jacobian, G, and the estimated state history.
    """
    num_input = len(list(bs.keys()))
    num_pix = xs[0].shape[0]

    dx_err = np.zeros(num_pix)
    for j in range(num_input):
        dx_err = dx_err + util.compare(2*bl.batch_mvip(G, bs[j]), xs[2*j] - xs[2*j+1])

    return dx_err.mean()

def eval_dz_error(G, H, us, zs):
    """
    Evaluatate the error between H*G*u[k] and z[k] - z[k-1] in each time step; i.e., the change
    in pairwise probe data predicted by the Jacobian.

    Derivation from state-space model:
        x[k] = x[k-1] + G*u[k] + noise
        z[k] = H*x[k] + noise
     => z[k] = H*x[k-1] + H*G*u[k] + noise
             = z[k-1] + H*G*u[k] + noise
     => z[k] - z[k-1] = H*G*u[k] + noise
    """
    num_iter = len(list(zs.keys()))
    dz_err_per_iter = np.zeros(num_iter - 1)

    for k in range(1, num_iter):
        dz = zs[k] - zs[k-1]
        dz_pred = bl.batch_mvip(H, bl.batch_mvip(G, us[k]))
        dz_err_per_iter[k-1] = util.compare(dz_pred, dz)

    return dz_err_per_iter.mean()


@jax.jit
def least_squares_pairwise_cost(G: np.ndarray, Psi: np.ndarray, bs: dict, zs: dict):
    """
    Compute the error between the predicted changes in observations, 2*H*G*b[j], and the actual
    changes in observations, z[2*j] - z[2*j+1].

    :param G: (num_pix, 2, num_act) real-valued Jacobian.
    :param Psi: (num_act, num_probe) probe matrix. Each column is a pairwise probe command.
    :param bs: dict that maps integers, j, to (num_act,) input commands.
    :param zs: dict that maps integers, k, to (num_pix, num_probe) pairwise probe data vectors.
               z[2*j] corresponds to +b[j], and z[2*j+1] corresponds to -b[j].
    """
    G = G.astype(jnp.float64)

    # FIXME: probably faster to evaluate
    #                8 * Psi.T @ (G.T @ (G @ b)))
    #  instead of    2 * (4 * (G @ Psi).T) @ (G @ b)
    H = 4 * d.batch_mt(d.batch_mmip(G, Psi))
    num_input = len(bs.keys())  # Number of input excitiations
    num_pix = zs[0].shape[0]  # Number of dark-zone pixels

    # Motivation of dztot: scale the cost function so that it's invariant to the contrast level
    dztot = 0.
    err = jnp.zeros(num_pix)

    for j in range(num_input):
        dz = zs[2 * j] - zs[2 * j + 1]
        dztot = dztot + d.batch_vvip(dz, dz)
        b = bs[j].astype(jnp.float64)  # Cast to float to avoid Jax overflow bug
        dz_pred = 2 * d.batch_mvip(H, d.batch_mvip(G, b))
        dzerr = dz_pred - dz
        err = err + d.batch_vvip(dzerr, dzerr)

    J = err.mean() / dztot.mean()
    return J

