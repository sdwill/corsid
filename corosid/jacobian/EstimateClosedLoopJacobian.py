import logging

import numpy as np

from corosid.common import BaseEstimateParameters
from corosid.common.batch_linalg import batch_mt, batch_mmip, eye
from corosid.jacobian.MStep import MStep
from corosid.jacobian.expectation_maximization import run_expectation_maximization
from corosid.jacobian.maximum_likelihood import run_maximum_likelihood
from corosid.jacobian.least_squares import run_prediction_error_minimization

log = logging.getLogger(__name__)


class EstimateClosedLoopJacobian(BaseEstimateParameters):
    def __init__(self, params, wr, target_dir, output_dir):
        super().__init__(params, wr, target_dir, output_dir)

    def estimate(self, targets, tol, algorithm=''):
        """
        Identify the Jacobian matrix and noise statistics numerically via joint optimization.
        """
        log.info(f'Estimating {targets}...')

        if algorithm == 'ml':
            solution = run_maximum_likelihood(
                self.training,
                self.tree['G'], self.tree['Q'], self.tree['R'], self.tree['x0'], self.tree['P0'],
                targets=targets,
                tol=tol
            )
        elif algorithm == 'em':
            solution = run_expectation_maximization(
                self.training,
                self.tree['G'], self.tree['Q'], self.tree['R'], self.tree['x0'], self.tree['P0'],
                targets=targets,
                tol=tol
            )
        elif algorithm == 'pem':
            solution = run_prediction_error_minimization(
                self.training,
                self.tree['G'],
                tol=tol
            )
        else:
            valid_values = ['ml', 'em', 'pem']
            raise ValueError(f'Valid values for algorithm: {[alg for alg in valid_values]}')

        self.solution = solution
        self.tree['G'] = solution.G
        self.tree['Q'] = solution.Q
        self.tree['R'] = solution.R
        self.tree['x0'] = solution.x0
        self.tree['P0'] = solution.P0

        self.likelihoods = np.append(self.likelihoods, self.solution.likelihoods)
        self.error = np.append(self.error, self.solution.error)
        self.x_error = np.append(self.x_error, self.solution.x_error)
        self.z_error = np.append(self.z_error, self.solution.z_error)
        self.history.append((targets, len(self.solution.error)))

        return self.tree

    def estimate_noise(self, num_em_iter, targets=('Q', 'R')):
        """
        Identify noise covariances analytically
        """
        from corosid.jacobian.EStep import EStep
        data = self.training
        estep = EStep(data.us, data.zs, data.num_iter, data.num_pix, data.len_x, data.len_z)

        estep.G = self.tree['G'].astype(np.float64)
        estep.H = 4 * batch_mt(batch_mmip(self.tree['G'], data.Psi))
        estep.Q = self.tree['Q'] * eye(data.num_pix, data.len_x)
        estep.R = self.tree['R'] * eye(data.num_pix, data.len_z)
        estep.x0 = self.tree['x0'].astype(np.float64)
        estep.P0 = self.tree['P0'] * eye(data.num_pix, data.len_x)

        log.info('Estimating Q and R analytically...')
        for n in range(num_em_iter):
            estep.run()
            mstep = MStep(estep.xs, estep.zs, estep.us,
                          estep.Ps,
                          estep.Ls)

            if 'x0' in targets:
                # FIXME: does this index for xs and Ps need to update if I run EFC in continue mode?
                x0_est = mstep.solve_x0(estep.xs[-1])
                estep.x0 = x0_est
                self.tree['x0'] = x0_est
                loglik = estep.eval_likelihood()
                log.info(f'ℓ: {loglik:0.3e}')

            if 'P0' in targets:
                # FIXME: does this index for xs and Ps need to update if I run EFC in continue mode?
                P0_est = mstep.solve_P0(estep.xs[-1], estep.Ps[-1], self.tree['x0'])
                estep.P0 = P0_est * eye(self.training.num_pix, self.training.len_x)
                self.tree['P0'] = P0_est
                loglik = estep.eval_likelihood()
                log.info(f'P0: {P0_est:0.3e}')
                log.info(f'ℓ: {loglik:0.3e}')

            if 'Q' in targets:
                Q_est = mstep.solve_Q(estep.G, mstep.Sigma_Q, mstep.Phi, mstep.C, mstep.U, mstep.V)
                estep.Q = Q_est * eye(self.training.num_pix, self.training.len_x)
                self.tree['Q'] = Q_est
                loglik = estep.eval_likelihood()
                log.info(f'Q: {Q_est:0.3e}')
                log.info(f'ℓ: {loglik:0.3e}')

            if 'R' in targets:
                R_est = mstep.solve_R(estep.G, self.training.Psi, mstep.Sigma_R, mstep.B, mstep.D)
                estep.R = R_est * eye(self.training.num_pix, self.training.len_z)
                self.tree['R'] = R_est
                loglik = estep.eval_likelihood()
                log.info(f'R: {R_est:0.3e}')
                log.info(f'ℓ: {loglik:0.3e}')