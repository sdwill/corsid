import jax
import numpy as np
from jax import numpy as jnp
from scipy.optimize import minimize

from corosid.common.batch_linalg import batch_vvop, batch_mmop
from corosid.jacobian.config import USE_ADAM, USE_SCIPY, SCIPY_METHOD, SCIPY_TOL, SCIPY_OPT, \
    ADAM_ALPHA, ADAM_BETA1
from corosid.jax import differentiable as d, make_unpacker, pack

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

FACTOR = 1


def cov_to_opt(C):
    """
    The transformation that maps a noise covariance into the internal representation used for the
    optimizer.

    Example: we numerically optimize log-inverse covariance, but interface with the external world
    via ordinary covariance.
    """
    # return np.log(1. / C)
    return np.log(C) / FACTOR


def opt_to_cov(c):
    """
    The transformation that maps internal optimizer representation to an ordinary noise covariance.
    This is the inverse operation of cov_to_opt().
    """
    # return 1. / np.exp(c)
    return np.exp(c * FACTOR)


def opt_to_icov(c):
    """
    The transformation that maps internal optimizer representation to an ordinary
    inverse-covariance.

    MUST be implemented with Jax, so that the M-step cost is differentiable.
    """
    # return jnp.exp(c)
    return 1./jnp.exp(c * FACTOR)


class MStep:
    """
    FIXME 9/19/23: implement everything below in jax to speed up construction of M-step quantities
     over numpy
    """
    def __init__(self, xs, zs, us, Ps, Ls):
        self.xs = xs
        self.zs = zs
        self.us = us

        _, self.len_x = xs[-1].shape
        self.num_pix, self.len_z = zs[0].shape
        self.num_iter = len(zs.keys())   # Number of control iterations is defined by number of zs
        self.len_u = us[0].size

        K, num_pix, len_x, len_z, len_u = self.num_iter, self.num_pix, self.len_x, self.len_z, \
            self.len_u

        self.Sigma_Q = np.zeros((num_pix, len_x, len_x))
        self.Sigma_R = np.zeros((num_pix, len_x, len_x))
        self.Phi = np.zeros((num_pix, len_x, len_x))
        self.B = np.zeros((num_pix, len_z, len_x))
        self.C = np.zeros((num_pix, len_x, len_x))
        self.D = np.zeros((num_pix, len_z, len_z))

        self.U = np.zeros((len_u, len_u))
        self.V = np.zeros((num_pix, len_x, len_u))

        for k in range(K):  # Control index from 1 to K, inclusive, following Sarkka notation
            # Reminder: state-space model for data generation is
            #           x[k] = x[k-1] + G*u[k] + w[k]
            #           z[k] = H*x[k] + r[k]
            z = zs[k]
            u = us[k]
            x = xs[k]
            P = Ps[k]
            uu = u @ u
            # uu = 1.

            sigma = batch_vvop(x, x) + P
            self.Sigma_R += sigma
            self.Sigma_Q += sigma / uu
            self.B += batch_vvop(z, x)
            self.D += batch_vvop(z, z)

            xm = xs[k-1]  # Mnemonic: "x minus one" (i.e., (k-1)th iteration)
            Pm = Ps[k-1]

            # FIXME: is this the correct index?
            L = Ls[k-1]  # Smoother gain. "G" in Sarkka notation, but this clashes with Jacobian

            # FIXME: is this supposed to be uu or umum?
            self.Phi += (batch_vvop(xm, xm) + Pm) / uu
            self.C += (batch_vvop(x, xm) + batch_mmop(P, L)) / uu

            self.U += (np.outer(u, u)) / uu
            self.V += batch_vvop(x - xm, u) / uu

        self.Sigma_R /= K
        self.Sigma_Q /= K
        self.B /= K
        self.D /= K
        self.Phi /= K
        self.C /= K
        self.U /= K
        self.V /= K

    @staticmethod
    def cost(x0s, P0s, x0, p0inv, G, Psi, qinv, rinv,
             Sigma_Q, Sigma_R, Phi, B, C, D, U, V,
             num_iter):
        """
        Cost function

        x0s: (num_pix, len_x) smoothed state estimate for training data iteration 0
        P0s: (num_pix, len_x, len_x) smoothed state covariance estimate for
                                     training data iteration 0
        x0: (num_pix, len_x) mean of state distribution in training data iteration 0
        p0inv: float, inverse-variance parameter (see Notes) for state distribution
               in training data iteration 0
        G: (num_pix, len_x, len_u) control Jacobian
        Psi: (num_pix, len_z, len_u) probe matrix
        qinv: float, inverse-variance parameter for process noise (see Notes)
        rinv: float, inverse-variance parameter for measurement noise (see Notes)
        Sigma_Q, Sigma_R, Phi, B, C, D, U, V: EM cost function parameter matrices; see class
                                              attributes for sizes
        num_iter: int, number of iterations of training data

        Notes
        -----
        - When fitting inverse-variances numerically, it may be preferential to optimize
          transformed parameters, for example to enforce nonnegativity and/or improve the condition
          number. These transformed representations are denoted by lowercase letters, whereas the
          inverse variances themselves are denoted with capitals. For example, if optimizing
          log-inverse-covariance, we have P0inv = exp(p0inv) and likewise for Qinv and Rinv.

        """

        # Any transformations that map the variables (p0inv, qinv, rinv) -> (P0inv, Qinv, Rinv) are
        # defined here
        P0inv = opt_to_icov(p0inv)
        Qinv = opt_to_icov(qinv)
        Rinv = opt_to_icov(rinv)

        x0_diff = x0s - x0
        K = jnp.float64(num_iter)

        # K = 100  # FIXME! using the value of num_iter makes jax return a gradient of zero
        P_outer = P0s + d.batch_vvop(x0_diff, x0_diff)
        P_term = d.batch_tr(P0inv * P_outer)

        # Tr{Q_inv @ (Sigma - C A.T - A C.T + A Phi A.T -
        #             G V.T - V G.T + A W G.T + G W.T A.T + G U G.T)}, with A = I
        GV = d.batch_mmop(G, V)
        Q_inner = Sigma_Q - C - d.batch_mt(C) + Phi - GV - d.batch_mt(GV) + d.batch_ABAT(G, U)
        Q_term = K * d.batch_tr(Qinv * Q_inner)

        # Tr{R_inv @ (D - B H.T - H B.T + H Sigma H.T)
        H = 4*d.batch_mt(d.batch_mmip(G, Psi))
        BH = d.batch_mmop(B, H)  # B @ H.T
        R_inner = D - BH - d.batch_mt(BH) + d.batch_ABAT(H, Sigma_R)
        R_term = K * d.batch_tr(Rinv * R_inner)

        # log(det(2*pi*I/P0)) = log((2*pi/P0) ** 2)
        #                     = 2 * log(2*pi / P0)
        #                     = 2 * (log(2*pi) - log(P0))
        #                     = 2 * (log(2*pi) - p0)

        slogPinv = 2*jnp.log(2*np.pi*P0inv)
        slogQinv = 2*K*jnp.log(2*np.pi*Qinv)
        slogRinv = 2*K*jnp.log(2*np.pi*Rinv)

        # If we consider P0inv, Qinv, and Rinv as the variables, then the sign of the determinant
        # term flips, because log |A| = -log |inv(A)|.  Optimizing with respect to those variables
        # numerically is more stable than optimizing with respect to P0, Q, R and differentiating
        # through the inverse operation.
        log_likelihood = 0.5 * (slogPinv + slogQinv + slogRinv - P_term - Q_term - R_term)

        return -log_likelihood.sum()  # Minimize the negative log-likelihood, summed over all pixels

    def estimate(self, x0s, P0s, x00, p0inv0, G0, Psi, qinv0, rinv0,
                 targets=('x0', 'P0', 'G', 'Q', 'R'),
                 tol=SCIPY_TOL):

        # Initial values for all parameters
        initial_values = {
            'x0': x00.astype(np.float64),
            'p0inv': p0inv0.astype(np.float64),
            'G': G0.astype(np.float64),
            'qinv': qinv0.astype(np.float64),
            'rinv': rinv0.astype(np.float64)
        }

        # Starting guess for optimizer, containing only the values that we are optimizing
        starting_guess = {opt_key: val for opt_key, val in initial_values.items()
                          if EXT_KEYS[opt_key] in targets}
        unpack = make_unpacker(starting_guess)

        # To get the gradients with respect to the parameters we're optimizing, we need to
        # dynamically construct a list of argument indices based on what we are trying to estimate.
        argnums = [ARGNUMS[OPT_KEYS[ext_key]] for ext_key in targets]

        def cost_for_optimizer(x):
            """
            The cost function that interfaces directly with scipy.optimize.minimize() or Adam. To
            conform to the scipy interface, this accepts a vector and returns the value of the EM
            Q-function and its gradient at the value of x.

            Internally:
            - Splits the parameter vector into its constituent variables
            - Computes the value of the cost and its gradients with respect to constituent
              variables, using Jax [1]
            - Returns the value of the cost and the gradient with respect to x (by concatenating
              the gradient vectors for the constituent variables)

            [1] Jax cannot differentiate through the step that splits the parameter vector, so we
                have to wrap this interface around the Jax code that differentiates with respect to
                the constituent variables
            """

            sol = unpack(x)

            # Update the values of the optimized variables using the current iterate (x). Any
            # variable not being optimized is held static using the starting value.
            current_values = {opt_key: sol[opt_key] if EXT_KEYS[opt_key] in targets
                              else initial_values[opt_key]
                              for opt_key in initial_values}

            J, grads = jax.value_and_grad(self.cost, argnums)(
                x0s, P0s, current_values['x0'], current_values['p0inv'], current_values['G'], Psi,
                current_values['qinv'], current_values['rinv'],
                self.Sigma_Q, self.Sigma_R, self.Phi, self.B, self.C, self.D, self.U, self.V,
                self.num_iter)
            gradient = np.concatenate([np.array(grad).ravel() for grad in grads])

            print(f'J: {J:0.3e}\t ||∂g|| = {np.linalg.norm(gradient):0.3e}')
            return J, gradient

        if USE_ADAM:
            from corosid.common import AdamOptimizer
            adam = AdamOptimizer(cost_for_optimizer,
                                 x0=pack(starting_guess), num_iter=1,
                                 alpha=ADAM_ALPHA, beta1=ADAM_BETA1, beta2=0.999, eps=1e-8)
            Js, x = adam.optimize()

            class Result:
                def __init__(self, x):
                    self.x = x

            res = Result(unpack(x))
            return res

        if USE_SCIPY:
            res = minimize(cost_for_optimizer,
                           x0=pack(starting_guess), jac=True,
                           method=SCIPY_METHOD, options=SCIPY_OPT, tol=tol)
            sol = unpack(res.x)
            res.x = {key: np.asarray(sol[key]) for key in sol}
            return res

    @staticmethod
    def solve_Q(G, Sigma_Q, Phi, C, U, V):
        """
        Closed-form estimator for process noise covariance. Take the mean of the diagonal elements,
        c, and assume that Q = cI for all pixels.
        """
        GV = d.batch_mmop(G, V)

        # Shape: (num_pix, len_x, len_x)
        Q_est = Sigma_Q - C - d.batch_mt(C) + Phi - GV - d.batch_mt(GV) + d.batch_ABAT(G, U)
        _, len_x, _ = Q_est.shape

        # Compute average value of diagonals for each pixel. Shape: (num_pix,)
        trace = d.batch_tr(Q_est) / len_x
        return np.float64(trace.mean())  # Average over all pixels

    @staticmethod
    def solve_R(G, Psi, Sigma_R, B, D):
        H = 4*d.batch_mt(d.batch_mmip(G, Psi))
        BH = d.batch_mmop(B, H)  # B @ H.T
        R_est = D - BH - d.batch_mt(BH) + d.batch_ABAT(H, Sigma_R)  # (num_pix, len_z, len_z)
        _, len_z, _ = R_est.shape

        # Compute average value of diagonals for each pixel. Shape: (num_pix,)
        trace = d.batch_tr(R_est) / len_z
        return np.float64(trace.mean())

    @staticmethod
    def solve_P0(x0s, P0s, x0):
        x0_diff = x0s - x0
        P_est = P0s + d.batch_vvop(x0_diff, x0_diff)
        _, len_x, _ = P_est.shape
        trace = d.batch_tr(P_est) / len_x
        return np.float64(trace.mean())

    @staticmethod
    def solve_x0(x0s):
        return x0s
