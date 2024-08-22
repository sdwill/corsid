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

def fit_jacobian_to_iefc_matrix(
        data: TrainingData,
        G0: ArrayLike,
        method='L-BFGS-B',
        options: Dict = None,
        tol=1e-3
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

    # Obtain iEFC matrix in closed form
    A = fit_iefc_matrix(data.zs, data.us)

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
        calls += 1

        if calls % 10 == 0:
            H = 4 * bl.batch_mt(bl.batch_mmip(G, data.Psi))
            xs = estimate_states(H, data.zs)

            z_errors.append(eval_z_error(H, xs, data.zs))
            dx_errors.append(eval_dx_error(G, xs, data.us))
            dz_errors.append(eval_dz_error(G, H, data.us, data.zs))

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

    result = FitToIEFCResult(
        unpack(solution)['G'],
        costs=np.array(costs),
        dz_errors=np.array(dz_errors),
        dx_errors=np.array(dx_errors),
        z_errors=np.array(z_errors),
    )

    return result
