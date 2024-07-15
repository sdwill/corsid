import logging
from dataclasses import dataclass

import jax
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from corosid.common.batch_linalg import batch_mt, batch_mmip, eye
from corosid.jacobian.EStep import EStep
from corosid.jacobian.MStep import MStep, opt_to_cov, cov_to_opt
from corosid.jacobian.config import USE_ADAM, \
    ADAM_NUM_ITER, ADAM_ALPHA, ADAM_BETA1, USE_SCIPY, SCIPY_METHOD, SCIPY_OPT, SCIPY_TOL
from corosid.common.optutil import make_unpacker, pack

log = logging.getLogger(__name__)
jax.config.update('jax_enable_x64', True)

OPT_KEYS = {
    'G': 'G',
    'Q': 'qinv',
    'R': 'rinv',
    'x0': 'x0',
    'P0': 'p0inv',
}

EXT_KEYS = {
    'G': 'G',
    'qinv': 'Q',
    'rinv': 'R',
    'x0': 'x0',
    'p0inv': 'P0',
}


ARGNUMS = {
    'x0': 2,
    'p0inv': 3,
    'G': 4,
    'qinv': 6,
    'rinv': 7
}


@dataclass
class EstimationResult:
    G: ArrayLike
    Q: np.float64
    R: np.float64
    x0: ArrayLike
    P0: np.float64
    likelihoods: ArrayLike
    error: ArrayLike
    x_error: ArrayLike
    z_error: ArrayLike
    estep: EStep
    mstep: MStep


def run_maximum_likelihood(data,
                           G0, Q0, R0, x00, P00,
                           targets=('x0', 'P0', 'G', 'Q', 'R'),
                           tol=SCIPY_TOL
                           ):
    likelihoods = []
    error = []
    x_error = []
    z_error = []

    # Initialize E-step
    estep = EStep(data.us, data.zs, data.num_iter, data.num_pix, data.len_x, data.len_z)
    estep.G = G0.astype(np.float64)
    estep.H = 4 * batch_mt(batch_mmip(G0, data.Psi))
    estep.Q = Q0 * eye(data.num_pix, data.len_x)
    estep.R = R0 * eye(data.num_pix, data.len_z)
    estep.x0 = x00.astype(np.float64)
    estep.P0 = P00 * eye(data.num_pix, data.len_x)

    # Initial values for all parameters
    # Noise covariances are transformed into the internal representation for optimization.
    initial_values = {
        'G': G0.astype(np.float64),
        'qinv': cov_to_opt(Q0),
        'rinv': cov_to_opt(R0),
        'x0': x00.astype(np.float64),
        'p0inv': cov_to_opt(P00)
    }

    # Starting guess for optimizer, containing only the values that we are optimizing
    starting_guess = {opt_key: val for opt_key, val in initial_values.items()
                      if EXT_KEYS[opt_key] in targets}
    unpack = make_unpacker(starting_guess)
    mstep_outer = None

    def cost_for_optimizer(x):
        """
        The cost function that interfaces directly with scipy.optimize.minimize() or Adam.
        To conform to the scipy interface, this accepts a vector and returns the value of the
         negative log-likelihood and its gradient at the value of x.

        Internally:
        - Splits the parameter vector into its constituent variables
        - Runs an E-step
        - Computes the value of the log-likelihood
        - Evaluates the EM Q-function and its gradients with respect to the constituent variables
          (using the M-step class)
        - Returns the value of the negative log-likelihood and its gradient with respect to x
          (by concatenating the gradient vectors for the constituent variables)

        Per Fisher's identity, at the start of an EM iteration, the gradient of the log-likelihood
        and the gradient of the EM Q-function are the same.
        """
        sol = unpack(x)

        # Update the values of the optimized variables using the current iterate (x). Any variable
        # not being optimized is held static using the starting value.
        current_values = {opt_key: sol[opt_key] if EXT_KEYS[opt_key] in targets
                          else initial_values[opt_key]
                          for opt_key in initial_values}

        # Update E-step variables.
        # The E-step expects ordinary noise covariances, so convert the inverse-variance parameters
        # first. Only need to update the variables that are actively changing during optimization.
        if 'x0' in targets:
            estep.x0 = current_values['x0']

        if 'P0' in targets:
            estep.P0 = opt_to_cov(current_values['p0inv']) * eye(data.num_pix, data.len_x)

        if 'G' in targets:
            estep.G = current_values['G']
            estep.H = 4 * batch_mt(batch_mmip(current_values['G'], data.Psi))

        if 'Q' in targets:
            estep.Q = opt_to_cov(current_values['qinv']) * eye(data.num_pix, data.len_x)

        if 'R' in targets:
            estep.R = opt_to_cov(current_values['rinv']) * eye(data.num_pix, data.len_z)

        # Evaluate the log-likelihood
        estep.run()
        J = estep.eval_likelihood()

        # Compute the gradient of the log-likelihood, which is the same as the gradient of the EM
        # Q-function when evaluated at the current parameters (this is Fisher's identity).
        mstep = MStep(estep.xs, estep.zs, estep.us, estep.Ps, estep.Ls)

        # Grab the M-step and store it in the outer scope so that it can be returned with other
        # internal stuff
        nonlocal mstep_outer
        mstep_outer = mstep

        # To get the gradients with respect to the parameters we're optimizing, we need to
        # dynamically construct a list of argument indices based on what we are trying to estimate.
        argnums = [ARGNUMS[OPT_KEYS[ext_key]] for ext_key in targets]

        x0s = estep.xs[-1]  # Smoothed estimate of initial state
        P0s = estep.Ps[-1]  # Smoothed estimate of initial state covariance

        _, gradients = jax.value_and_grad(mstep.cost, argnums)(
            x0s, P0s, current_values['x0'], current_values['p0inv'], current_values['G'], data.Psi,
            current_values['qinv'], current_values['rinv'],
            mstep.Sigma_Q, mstep.Sigma_R, mstep.Phi, mstep.B, mstep.C, mstep.D, mstep.U, mstep.V,
            num_iter=mstep.num_iter)
        gradient = np.concatenate([np.array(grad).ravel() for grad in gradients])

        likelihoods.append(estep.eval_likelihood())
        error.append(np.mean(estep.eval_error(estep.zs)))
        x_error.append(np.mean(estep.eval_x_err()))
        z_error.append(np.mean(estep.eval_z_err(estep.zs)))

        log.info(f'ℓ: {J:0.3e}\t ||∂g|| = {np.linalg.norm(gradient):0.3e}')
        return -J, gradient  # Minimize NEGATIVE log-likelihood

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
    final_values = {opt_key: res.x[opt_key] if EXT_KEYS[opt_key] in targets
                    else initial_values[opt_key]
                    for opt_key in initial_values}

    # Pack results into data structure. Convert all inverse-variance parameters back to variances.
    result = EstimationResult(final_values['G'],
                              opt_to_cov(final_values['qinv']),
                              opt_to_cov(final_values['rinv']),
                              final_values['x0'],
                              opt_to_cov(final_values['p0inv']),
                              np.array(likelihoods),
                              np.array(error),
                              np.array(x_error),
                              np.array(z_error),
                              estep, mstep_outer)

    return result
