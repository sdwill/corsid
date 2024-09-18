import logging
from dataclasses import dataclass
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from corsid import util, batch_linalg as bl, differentiable as d, TrainingData
from corsid.adam import AdamOptimizer
from corsid.least_squares import estimate_states, eval_z_error, eval_dz_error, eval_dx_error

log = logging.getLogger(__name__)
jax.config.update('jax_enable_x64', True)


@jax.jit
def cost(G: np.ndarray, Psi: np.ndarray, A: np.ndarray):
    """
    Cost function that measures the fit between the Jacobian G, and the iEFC matrix, A:

    ||A - H*G||^2

    where H = 4(G*Psi).T.
    """
    G = G.astype(jnp.float64)
    H = 4 * d.batch_mt(d.batch_mmip(G, Psi))

    A_pred = d.batch_mmip(H, G)
    A_err = A - A_pred
    return d.l2sqr(A_err.ravel()) / d.l2sqr(A.ravel())

@dataclass
class FitToIEFCResult:
    G: ArrayLike
    costs: ArrayLike
    dz_errors: ArrayLike
    dx_errors: ArrayLike
    z_errors: ArrayLike
    val_errors: ArrayLike = None


def fit_iefc_matrix(zs, us):
    """
    Calculate the iEFC matrix, given a time series of pairwise-probe data vectors (zs) and their
    associated DM updates (us).
    """
    num_iter = len(zs.keys())

    # Each z vector is shape (num_pix, num_probe).
    # Stack into an array with shape (num_pix, num_probe, num_iter), interpreted as a stack of
    # (num_probe, num_iter) matrices.
    dZ = np.stack([zs[k] - zs[k-1] for k in range(1, num_iter)], axis=-1)

    # Stack the u vectors into a (num_act, num_iter) matrix
    U = np.column_stack([us[k] for k in range(1, num_iter)])

    # dZ = A*U, so A_hat = dZ * pinv(U), with dZ and A looped over pixels
    return bl.batch_mmip(dZ, np.linalg.pinv(U))

def list_to_masked(lst):
    """ Converts a list consisting of floats and None to a masked array with None masked out """
    return np.ma.masked_invalid(np.array(lst, dtype=np.float64))

def fit_jacobian_to_iefc_matrix(
        data: TrainingData,
        validation_data: TrainingData,
        G0: ArrayLike,
        method='L-BFGS-B',
        options: Dict = None,
        tol=1e-3,
        eval_metrics_every=10,
):
    """
    Two-step algorithm for estimating Jacobian matrix: calculate the iEFC matrix in closed form,
    then fit the Jacobian to the iEFC matrix via numerical optimization.
    """
    if options is None:
        options = {'disp': False, 'maxcor': 10, 'maxls': 100}

    costs = []
    z_errors = []
    dx_errors = []
    dz_errors = []
    val_errors = []  # Same as dz_errors, but on validation data

    # Obtain iEFC matrix in closed form
    A = fit_iefc_matrix(data.dzs, data.us)

    # Starting guess for optimizer, containing only the values that we are optimizing
    starting_guess = {'G': G0.astype(np.float64)}
    unpack = util.make_unpacker(starting_guess)

    calls = 0  # Incremented every time cost function is evaluated

    def cost_for_optimizer(x):
        sol = unpack(x)
        G = sol['G']
        J, grads = jax.value_and_grad(cost, argnums=(0,))(
            G, data.Psi, A)
        gradient = np.concatenate([np.array(grad).ravel() for grad in grads])
        costs.append(J)
        log.info(f'J: {J:0.3e}\t ||∂g|| = {np.linalg.norm(gradient):0.3e}')

        # Evaluate quality-of-fit metrics
        # Only do this after every 10 cost function evaluations to save time
        nonlocal calls

        if calls % eval_metrics_every == 0:
            H = 4 * bl.batch_mt(bl.batch_mmip(G, data.Psi))
            xs = estimate_states(H, data.dzs)

            z_errors.append(eval_z_error(H, xs, data.dzs))
            dx_errors.append(eval_dx_error(G, xs, data.us))
            dz_errors.append(eval_dz_error(G, H, data.us, data.dzs))
            val_errors.append(eval_dz_error(G, H, validation_data.us, validation_data.dzs))
        else:
            z_errors.append(None)
            dx_errors.append(None)
            dz_errors.append(None)
            val_errors.append(None)

        # Increment call counter AFTER checking whether to evaluate goodness-of-fit metrics, so
        # that we get an eval at calls = 0 (at the start of the script)
        calls += 1
        return J, gradient

    if method == 'adam':
        adam = AdamOptimizer(cost_for_optimizer, x0=util.pack(starting_guess),
                             num_iter=options['num_iter'],
                             alpha=options['alpha'],
                             beta1=options['beta1'],
                             beta2=options['beta2'],
                             eps=options['eps'])
        Js, solution = adam.optimize()

    else:
        res = minimize(cost_for_optimizer, x0=util.pack(starting_guess), jac=True,
                       method=method, options=options, tol=tol)
        solution = res.x

    # Evaluate metrics once more at the end to get their final values
    G = unpack(solution)['G']
    H = 4 * bl.batch_mt(bl.batch_mmip(G, data.Psi))
    xs = estimate_states(H, data.dzs)

    z_errors.append(eval_z_error(H, xs, data.dzs))
    dx_errors.append(eval_dx_error(G, xs, data.dus))
    dz_errors.append(eval_dz_error(G, H, data.us, data.dzs))
    val_errors.append(eval_dz_error(G, H, validation_data.us, validation_data.dzs))

    result = FitToIEFCResult(
        G,
        costs=np.array(costs),
        dz_errors=list_to_masked(dz_errors),
        dx_errors=list_to_masked(dx_errors),
        z_errors=list_to_masked(z_errors),
        val_errors=list_to_masked(val_errors)
    )

    return result
