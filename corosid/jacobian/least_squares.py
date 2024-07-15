import logging
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from corosid.common import batch_linalg as bl, differentiable as d
from corosid.common.util import compare
from corosid.common.util import l1_pairwise_probe_estimator
from corosid.jacobian.MStep import MStep
from corosid.jacobian.config import USE_ADAM, \
    ADAM_NUM_ITER, ADAM_ALPHA, ADAM_BETA1, USE_SCIPY, SCIPY_METHOD, SCIPY_OPT, SCIPY_TOL
from corosid.common.optutil import make_unpacker, pack

log = logging.getLogger(__name__)
jax.config.update('jax_enable_x64', True)


class EstimationStep:  # Sort of like an E-step, but using a pairwise probe estimator
    def __init__(self, us, zs, num_iter, num_pix, len_x, len_z):
        self.us = us
        self.zs = zs
        self.num_iter = num_iter
        self.num_pix = num_pix
        self.len_x = len_x
        self.len_z = len_z
        self.xs = {}  # State estimate, (num_iter, num_pix, len_x); set by run()

        self.G = None
        self.H = None

    def run(self):
        K = self.num_iter
        from tqdm import tqdm

        for k in tqdm(range(K), desc='Running pairwise estimator', leave=False):
            self.xs[k] = l1_pairwise_probe_estimator(self.H, self.zs[k])

    def eval_z_err(self, zs):
        z_err_per_iter = np.zeros(self.num_iter)
        for k in range(self.num_iter):
            z_err_per_iter[k] = compare(bl.batch_mvip(self.H, self.xs[k]), zs[k])

        return z_err_per_iter.mean()

    def eval_x_err(self):
        x_err_per_iter = np.zeros(self.num_iter - 1)
        for k in range(1, self.num_iter):
            x_err_per_iter[k-1] = compare(bl.batch_mvip(self.G, self.us[k]),
                                          self.xs[k] - self.xs[k - 1])

        return x_err_per_iter.mean()

    def eval_error(self, zs):
        """
        Validation error metric that compares predictions to purely observable values (differences
        of pairwise data vectors).

            x[k] = x[k-1] + G*u[k] + noise
            z[k] = H*x[k] + noise
         => z[k] = H*x[k-1] + H*G*u[k] + noise
                 = z[k-1] + H*G*u[k] + noise
         => z[k] - z[k-1] = H*G*u[k] + noise
        """
        err_per_iter = np.zeros(self.num_iter - 1)

        for k in range(1, self.num_iter):
            dz = zs[k] - zs[k-1]
            dz_pred = bl.batch_mvip(self.H, bl.batch_mvip(self.G, self.us[k]))
            err_per_iter[k-1] += compare(dz_pred, dz)

        return err_per_iter.mean()


@dataclass
class SystemIDResult:
    G: ArrayLike
    costs: ArrayLike
    error: ArrayLike
    x_error: ArrayLike
    z_error: ArrayLike
    estep: EstimationStep


@jax.jit
def least_squares_state_cost(G: np.ndarray, Psi: np.ndarray, us: dict, xs: dict, zs: dict):
    """
    Compute the error between the predicted state changes, G*u[k], and actual state changes,
    x[k] - x[k-1], summed over all states.

    The cost from each iteration has an extra factor 1/(u[k] @ u[k]) that mimics the effect of
    modeling the uncertainty in the state change as being from Jacobian errors alone. With a Kalman
    filter, we would model this as the process noise variable Q[k] = Q * u[k] @ u[k], resulting in
    the 1/(u[k] @ u[k]) factor appearing in the cost.
    """
    G = G.astype(jnp.float64)
    num_iter = len(zs.keys())
    num_pix = xs[0].shape[0]

    err = jnp.zeros((num_pix, num_iter - 1))
    # dxtot = 0.

    for k in range(1, num_iter):
        dx = xs[k] - xs[k-1]
        # dxtot += bl.batch_vvip(dx, dx)
        Gu = d.batch_mvip(G, us[k].astype(jnp.float64))  # Cast u to float to avoid Jax overflow bug
        uu = d.batch_vvip(us[k], us[k])
        xerr = Gu - dx
        err = err.at[:, k-1].set(d.batch_vvip(xerr, xerr) / uu)

    # dxtot = (dxtot / (num_iter - 1)).mean()
    J = err.mean()
    return J


@jax.jit
def least_squares_cost(G: np.ndarray, Psi: np.ndarray, us: dict, xs: dict, zs: dict):
    """
    Compute the error between the predicted changes in observations, H*G*u[k], and the actual
    changes in observations, z[k] - z[k-1]. This is similar to transition error, but mapping all
    quantities into the observable domain.
    """
    G = G.astype(jnp.float64)
    H = 4 * d.batch_mt(d.batch_mmip(G, Psi))
    num_iter = len(zs.keys())
    num_pix = xs[0].shape[0]

    err = jnp.zeros((num_pix, num_iter - 1))
    dztot = 0.

    for k in range(1, num_iter):
        dz = zs[k] - zs[k-1]
        dztot = dztot + d.batch_vvip(dz, dz)
        # uu = d.batch_vvip(us[k], us[k])

        Gu = d.batch_mvip(G, us[k].astype(jnp.float64))  # Cast u to float to avoid Jax overflow bug
        dz_pred = d.batch_mvip(H, Gu)
        dzerr = dz_pred - dz
        err = err.at[:, k-1].set(d.batch_vvip(dzerr, dzerr))

    # Motivation of dztot: scale the cost function so that it's invariant to the contrast level
    dztot = (dztot / (num_iter - 1)).mean()
    J = err.mean() / dztot
    return J


def run_least_squares_identification(
        data,
        G0,
        tol=SCIPY_TOL
):
    costs = []
    error = []
    x_error = []
    z_error = []

    # Starting guess for optimizer, containing only the values that we are optimizing
    starting_guess = {'G': G0.astype(np.float64)}
    unpack = make_unpacker(starting_guess)

    def cost_for_optimizer(x):
        sol = unpack(x)

        G = sol['G']
        H = 4 * bl.batch_mt(bl.batch_mmip(G, data.Psi))

        estimator = EstimationStep(data.us, data.zs, data.num_iter, data.num_pix, data.len_x,
                                   data.len_z)
        estimator.G = G
        estimator.H = H

        # For least-squares minimization (HGu - dz), this is only necessary for evaluating the
        # performance metrics error, x_error, and z_error
        estimator.run()

        # J, grads = jax.value_and_grad(least_squares_state_cost, argnums=(0,))(
        #     G, data.Psi, data.us, estimator.xs, data.zs)
        J, grads = jax.value_and_grad(least_squares_cost, argnums=(0,))(G, data.Psi, data.us,
                                                                        estimator.xs, data.zs)
        gradient = np.concatenate([np.array(grad).ravel() for grad in grads])

        costs.append(J)
        error.append(np.mean(estimator.eval_error(data.zs)))
        x_error.append(estimator.eval_x_err())
        z_error.append(estimator.eval_z_err(data.zs))

        log.info(f'J: {J:0.3e}\t ||∂g|| = {np.linalg.norm(gradient):0.3e}')
        return J, gradient

    if USE_ADAM:
        from corosid.common import AdamOptimizer
        num_iter = ADAM_NUM_ITER
        adam = AdamOptimizer(cost_for_optimizer, x0=pack(starting_guess), num_iter=num_iter,
                             alpha=ADAM_ALPHA, beta1=ADAM_BETA1, beta2=0.999, eps=1e-8)
        Js, x = adam.optimize()

        class Result:
            def __init__(self, x):
                self.x = x

        res = Result(unpack(x))

    if USE_SCIPY:
        res = minimize(cost_for_optimizer, x0=pack(starting_guess), jac=True,
                       method=SCIPY_METHOD, options=SCIPY_OPT, tol=tol)
        res_unpacked = unpack(res.x)
        res.x = {key: np.asarray(res_unpacked[key]) for key in res_unpacked}

    # For each variable: if it is in the list of targets, take the optimized value, otherwise take
    # the initial value.
    final_values = {'G': res.x['G']}

    # Pack results into data structure. Several of these are dummy values, such as the Kalman
    # filter parameters Q, R, x0 and P0, because this algorithm doesn't use a Kalman filter.
    result = SystemIDResult(final_values['G'],
                            costs=np.array(costs),
                            error=np.array(error),
                            x_error=np.array(x_error),
                            z_error=np.array(z_error),
                            estep=EstimationStep)

    return result
