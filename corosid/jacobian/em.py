import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from corosid.jacobian.EStep import EStep
from corosid.jacobian.MStep import MStep, cov_to_opt, opt_to_cov
from corosid.jacobian.config import EM_NUM_ITER
from corosid.jacobian.config import SCIPY_TOL
from corosid.common.batch_linalg import batch_mt, batch_mmip, eye

log = logging.getLogger(__name__)


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


def run_expectation_maximization(data,
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

    num_iter = EM_NUM_ITER

    initial_values = current_values = {
        'G': G0,
        'qinv': cov_to_opt(Q0),
        'rinv': cov_to_opt(R0),
        'x0': x00,
        'p0inv': cov_to_opt(P00)
    }

    # FIXME 20230927: should run E-step once more after the last M-step to get the likelihood with
    #  the final parameter values
    for n in range(num_iter):
        log.info(f'EM iteration {n+1} of {num_iter+1}')
        estep.run()

        likelihoods.append(estep.eval_likelihood())
        error.append(np.mean(estep.eval_error(estep.zs)))
        x_error.append(np.mean(estep.eval_x_err()))
        z_error.append(np.mean(estep.eval_z_err(estep.zs)))
        log.info(f'ℓ: {likelihoods[-1]:0.3e}')

        mstep = MStep(estep.xs, estep.zs, estep.us, estep.Ps, estep.Ls)

        res = mstep.estimate(x0s=estep.xs[-1],
                             P0s=estep.Ps[-1],
                             x00=current_values['x0'],
                             p0inv0=current_values['p0inv'],
                             G0=current_values['G'],
                             Psi=data.Psi,
                             qinv0=current_values['qinv'],
                             rinv0=current_values['rinv'],
                             targets=targets,
                             tol=tol)

        current_values = {opt_key: res.x[opt_key] if EXT_KEYS[opt_key] in targets
                          else initial_values[opt_key] for opt_key in initial_values}

        if 'x0' in targets:
            estep.x0 = current_values['x0']

        if 'P0' in targets:
            estep.P0 = opt_to_cov(current_values['p0inv']) * eye(data.num_pix, data.len_x)

        if 'G' in targets:
            estep.G = current_values['G']
            estep.H = 4 * batch_mt(batch_mmip(estep.G, data.Psi))

        if 'Q' in targets:
            estep.Q = opt_to_cov(current_values['qinv']) * eye(data.num_pix, data.len_x)

        if 'R' in targets:
            estep.R = opt_to_cov(res.x['rinv']) * eye(data.num_pix, data.len_z)

    # For each variable: if it is in the list of targets, take the optimized value, otherwise take
    # the initial value.
    result = EstimationResult(current_values['G'],
                              opt_to_cov(current_values['qinv']),
                              opt_to_cov(current_values['rinv']),
                              current_values['x0'],
                              opt_to_cov(current_values['p0inv']),
                              np.array(likelihoods),
                              np.array(error),
                              np.array(x_error),
                              np.array(z_error),
                              estep, mstep)

    return result
