import numpy as np
from tqdm import tqdm

from corosid.common.batch_linalg import batch_mmip, batch_mvip, batch_ABAT
from corosid.common.util import l1_kalman_predict, l1_kalman_update, compare


def dict_rts_smoother(x_posts, x_priors, P_posts, P_priors):
    """
    RTS smoother implementation that stores internal variables as a dictionary, to simplify index calculations.
    The version in corosid.common.util uses numpy arrays.
    """
    K = max(list(x_posts.keys()))

    xs = {}
    Ps = {}
    Ls = {}

    xs[K] = x_posts[K]
    Ps[K] = P_posts[K]

    for k in tqdm(reversed(range(-1, K)), desc='Running RTS smoother', leave=False):  # k is control index (1-based)
        Ls[k] = batch_mmip(P_posts[k], np.linalg.inv(P_priors[k+1]))
        xs[k] = x_posts[k] + batch_mvip(Ls[k], xs[k+1] - x_priors[k+1])
        Ps[k] = P_posts[k] + batch_ABAT(Ls[k], Ps[k+1] - P_priors[k+1])

        # Equivalent to the non-vectorized loop below
        # for n in range(num_pix):
        #     As[k][n] = P_posts[k][n] @ np.linalg.inv(P_priors[k + 1][n])
        #     xs[k][n] = x_posts[k][n] + As[k][n] @ (xs[k + 1][n] - x_priors[k + 1][n])
        #     Ps[k][n] = P_posts[k][n] + As[k][n] @ (Ps[k + 1][n] - P_priors[k + 1][n]) @ As[k][n].T

    return xs, Ps, Ls


class EStep:
    def __init__(self, us, zs, num_iter, num_pix, len_x, len_z):
        self.us = us
        self.zs = zs
        self.num_iter = num_iter
        self.num_pix = num_pix
        self.len_x = len_x
        self.len_z = len_z

        self.x_priors = {}
        self.x_posts = {}
        self.xs = None  # Smoothed state, (num_iter, num_pix, len_x); set by smoother

        # These are populated by the Kalman filter/smoother, so it's fine if they are initialized
        # as zero
        self.P_priors = {}
        self.P_posts = {}
        self.Ps = None  # Smoothed state covariance, (num_iter, num_pix, len_x, len_x)
        self.Ls = None  # Smoother gain, (num_iter, num_pix, len_x, len_x)
        self.vs = {}
        self.Ss = {}

        # These parameters are updated from M-step values
        self.Q = None
        self.R = None
        self.G = None
        self.H = None
        self.x0 = None
        self.P0 = None

    def run(self):
        K = self.num_iter
        self.x_priors[-1] = self.x0
        self.x_posts[-1] = self.x0
        self.P_priors[-1] = self.P0
        self.P_posts[-1] = self.P0

        for k in tqdm(range(K), desc='Running Kalman filter', leave=False):
            uu = self.us[k] @ self.us[k]
            # uu = 1

            # Run the Kalman filter and RTS smoother
            self.x_priors[k], self.P_priors[k] = l1_kalman_predict(
                self.x_posts[k-1],
                batch_mvip(self.G, self.us[k]),
                self.P_posts[k-1], self.Q*uu)

            self.x_posts[k], self.P_posts[k], self.vs[k], self.Ss[k] = l1_kalman_update(
                self.x_priors[k], self.zs[k], self.P_priors[k], self.R, self.H)

        self.xs, self.Ps, self.Ls = dict_rts_smoother(self.x_posts, self.x_priors,
                                                      self.P_posts, self.P_priors)

    def eval_likelihood(self):
        """ Evaluate the marginal likelihood p(z|theta) """
        ell = 0.

        K = self.num_iter
        for k in range(K):
            # Quadratic form of innovations
            qf = np.einsum('...j,...ij,...i', self.vs[k], np.linalg.inv(self.Ss[k]), self.vs[k])
            ld = np.linalg.slogdet(2*np.pi*self.Ss[k])[1]  # [0] is sign, [1] is log-determinant
            ell += -0.5 * (qf + ld).sum()  # Sum over all pixels

        return ell.sum()

    def eval_z_err(self, zs):
        z_err_per_iter = np.zeros(self.num_iter)
        for k in range(self.num_iter):
            z_err_per_iter[k] = compare(batch_mvip(self.H, self.xs[k]), zs[k])

        return z_err_per_iter.mean()

    def eval_x_err(self):
        x_err_per_iter = np.zeros(self.num_iter)
        for k in range(self.num_iter):
            x_err_per_iter[k] = compare(batch_mvip(self.G, self.us[k]), self.xs[k] - self.xs[k - 1])

        return x_err_per_iter.mean()

    def eval_error(self, zs):
        """
        Validation error metric that compares predictions to purely observable values
        (differences of pairwise data vectors).

            x[k] = x[k-1] + G*u[k] + noise
            z[k] = H*x[k] + noise
         => z[k] = H*x[k-1] + H*G*u[k] + noise
                 = z[k-1] + H*G*u[k] + noise
         => z[k] - z[k-1] = H*G*u[k] + noise
        """
        err = 0.

        for k in range(1, self.num_iter):
            dz = zs[k] - zs[k-1]
            dz_pred = batch_mvip(self.H, batch_mvip(self.G, self.us[k]))
            err += compare(dz_pred, dz) / self.num_iter

        return err
