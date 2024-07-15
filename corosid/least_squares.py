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

from corosid import util, batch_linalg as bl, differentiable as d
from corosid.adam import AdamOptimizer
from corosid.TrainingData import TrainingData

log = logging.getLogger(__name__)
jax.config.update('jax_enable_x64', True)


def estimate_states(H, zs):
    """ Run pairwise probe estimation to estimate the state in each time step """
    num_iter = len(list(zs.keys()))
    xs = {}
    for k in tqdm(range(num_iter), desc='Estimating states', leave=False):
        xs[k] = util.l1_pairwise_probe_estimator(H, zs[k])

    return xs

def eval_z_error(H, xs, zs):
    """ Evaluate the error between H*x[k] and z[k] in each time step """
    num_iter = len(list(zs.keys()))
    z_err_per_iter = np.zeros(num_iter)
    for k in range(num_iter):
        z_err_per_iter[k] = util.compare(bl.batch_mvip(H, xs[k]), zs[k])

    return z_err_per_iter.mean()

def eval_dx_error(G, xs, us):
    """
    Evaluate the error between G*u[k] and x[k] - x[k-1] in each time step.
    This measures the consistency between the Jacobian, G, and the estimated state history.
    """
    num_iter = len(list(xs.keys()))
    dx_err_per_iter = np.zeros(num_iter - 1)
    for k in range(1, num_iter):
        dx_err_per_iter[k-1] = util.compare(bl.batch_mvip(G, us[k]), xs[k] - xs[k - 1])

    return dx_err_per_iter.mean()

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
        dz_err_per_iter[k-1] += util.compare(dz_pred, dz)

    return dz_err_per_iter.mean()

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
def least_squares_cost(G: np.ndarray, Psi: np.ndarray, us: dict, zs: dict):
    """
    Compute the error between the predicted changes in observations, H*G*u[k], and the actual
    changes in observations, z[k] - z[k-1]. This is similar to transition error, but mapping all
    quantities into the observable domain.
    """
    G = G.astype(jnp.float64)
    H = 4 * d.batch_mt(d.batch_mmip(G, Psi))
    num_iter = len(zs.keys())
    num_pix = zs[0].shape[0]

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

@dataclass
class LeastSquaresIDResult:
    G: ArrayLike
    costs: ArrayLike
    dz_errors: ArrayLike
    dx_errors: ArrayLike
    z_errors: ArrayLike

def run_batch_least_squares_id(
        data: TrainingData,
        G0: ArrayLike,
        method='L-BFGS-B',
        options: Dict = None,
        tol=1e-3
):
    if options is None:
        options = {'disp': False, 'maxcor': 10, 'maxls': 100}

    costs = []
    dz_errors = []
    dx_errors = []
    z_errors = []

    # Starting guess for optimizer, containing only the values that we are optimizing
    starting_guess = {'G': G0.astype(np.float64)}
    unpack = util.make_unpacker(starting_guess)

    def cost_for_optimizer(x):
        sol = unpack(x)

        G = sol['G']
        H = 4 * bl.batch_mt(bl.batch_mmip(G, data.Psi))

        # For least-squares minimization (HGu - dz), this is only necessary for evaluating the
        # performance metrics error, x_error, and z_error
        xs = estimate_states(H, data.zs)

        # J, grads = jax.value_and_grad(least_squares_state_cost, argnums=(0,))(
        #     G, data.Psi, data.us, estimator.xs, data.zs)
        J, grads = jax.value_and_grad(least_squares_cost, argnums=(0,))(
            G, data.Psi, data.us, data.zs)
        gradient = np.concatenate([np.array(grad).ravel() for grad in grads])

        costs.append(J)
        z_errors.append(eval_z_error(H, xs, data.zs))
        dx_errors.append(eval_dx_error(G, xs, data.us))
        dz_errors.append(eval_dz_error(G, H, data.us, data.zs))

        log.info(f'J: {J:0.3e}\t ||∂g|| = {np.linalg.norm(gradient):0.3e}')
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

    result = LeastSquaresIDResult(
        unpack(solution)['G'],
        costs=np.array(costs),
        dz_errors=np.array(dz_errors),
        dx_errors=np.array(dx_errors),
        z_errors=np.array(z_errors),
    )

    return result

class StochasticLeastSquaresID:
    """
    Estimate Jacobian using a least-squares cost function and stochastic optimization (with Adam).
    """
    def __init__(self, target_dir, output_dir, G0):
        self.costs = []
        self.dz_errors = []
        self.dx_errors = []
        self.z_errors = []
        self.target_dir = target_dir
        self.data = None  # Set by load_data during run()

        # Starting guess for optimizer, containing only the values that we are optimizing
        self.starting_guess = {'G': G0.astype(np.float64)}
        self.unpack = util.make_unpacker(self.starting_guess)
        self.adam = None

        self.output_dir = output_dir / f'{util.today()}_{util.now()}_systemid'
        self.output_dir.mkdir(exist_ok=True, parents=True)
        logging.getLogger().addHandler(
            logging.FileHandler(self.output_dir / 'output.log')
        )
        self.status_dir = self.output_dir / 'status'
        self.status_dir.mkdir(exist_ok=True)

    def estimate_states(self, sol: dict):
        G = sol['G']
        H = 4 * bl.batch_mt(bl.batch_mmip(G, self.data.Psi))

        # This is only necessary for evaluating the performance metrics (error, x_error, z_error)
        xs = estimate_states(H, self.data.zs)
        self.dz_errors.append(eval_dz_error(G, H, self.data.us, self.data.zs))
        self.dx_errors.append(eval_dx_error(G, xs, self.data.us))
        self.z_errors.append(eval_z_error(H, xs, self.data.zs))

    def cost_for_optimizer(self, x: np.ndarray):
        """
        The actual cost function passed to the optimizer. Accepts a single real-valued vector, x,
        and returns the value of the cost function, J, as well as its derivative with respect to x,
        dJ/dx.
        """
        sol = self.unpack(x)
        self.estimate_states(sol)
        forward_and_gradient = jax.value_and_grad(least_squares_cost, argnums=(0,))
        J, grads = forward_and_gradient(sol['G'],
                                        self.data.Psi,
                                        self.data.us,
                                        self.data.zs)
        gradient = np.concatenate([np.array(grad).ravel() for grad in grads])
        log.info(f'J: {J:0.3e}\t ||∂g|| = {np.linalg.norm(gradient):0.3e}\t '
                 f'dz_err = {self.dz_errors[-1]:0.3e}')
        return J, gradient

    def make_status_plot(self, epoch):
        dz_errors = np.array(self.dz_errors)
        dx_errors = np.array(self.dx_errors)
        z_errors = np.array(self.z_errors)

        fig, axs = plt.subplots(dpi=150, layout='constrained', ncols=2, figsize=(7, 3))
        ax = axs[0]
        ax.plot(np.arange(dz_errors.size), 100 * dz_errors, '-', label='Data transition error')
        ax.plot(np.arange(dx_errors.size), 100 * dx_errors, '-', label='State transition error')
        ax.plot(np.arange(z_errors.size), 100 * z_errors, '-', label='Data error')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Error [%]')

        # Scale the y-axis limits to the largest starting error metric, so that the limits don't
        # change when L-BFGS has a hiccup iteration that causes a temporarily huge error value.
        ax.set_ylim([-5, 105 * np.max([dz_errors[0], dx_errors[0], z_errors[0]])])
        ax.set_title('Convergence', weight='bold')
        ax.grid(True, linewidth=0.5, linestyle='dotted')
        ax.legend(loc='best')

        ax = axs[1]
        ax.plot(np.arange(len(self.costs)), self.costs, 'o-')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Cost')

        ax.set_title(r'Cost vs. training epoch')
        ax.grid(True, linewidth=0.5, linestyle='dotted')

        fig.savefig(self.status_dir / f'epoch_{epoch:03d}')
        plt.close(fig)

    def run(self, load_data, num_training, batch_size, training_iter_start, num_epochs,
            adam_alpha, adam_beta1, adam_beta2, adam_eps):
        batch_starts = np.arange(training_iter_start, num_training - batch_size, batch_size)
        x = util.pack(self.starting_guess)

        for epoch in range(num_epochs):
            log.info(f'Epoch {epoch}')
            np.random.shuffle(batch_starts)
            self.adam = AdamOptimizer(
                self.cost_for_optimizer, x, num_iter=len(batch_starts),
                alpha=adam_alpha,
                beta1=adam_beta1, beta2=adam_beta2, eps=adam_eps)

            for batch_index, batch_start in enumerate(batch_starts):
                log.info(f'Batch {batch_index} of {batch_starts.size}')
                iters_to_load = range(batch_start, batch_start + batch_size)
                self.data: TrainingData = load_data(iters_to_load)
                self.adam.x = x
                J, x = self.adam.iterate(batch_index)

            mean_cost_epoch = np.mean(self.adam.Js)
            self.costs.append(mean_cost_epoch)
            self.make_status_plot(epoch)

        result = LeastSquaresIDResult(self.unpack(x)['G'],
                                      costs=np.array(self.costs),
                                      dz_errors=np.array(self.dz_errors),
                                      dx_errors=np.array(self.dx_errors),
                                      z_errors=np.array(self.z_errors))

        return result

def run_stochastic_least_squares_id(
        target_dir,
        output_dir,
        G0,
        load_data,
        num_training,
        training_iter_start,
        batch_size,
        num_epochs,
        adam_alpha,
        adam_beta1,
        adam_beta2,
        adam_eps
):
    system_id = StochasticLeastSquaresID(target_dir, output_dir, G0)
    result = system_id.run(load_data, num_training, batch_size, training_iter_start, num_epochs,
                           adam_alpha, adam_beta1, adam_beta2, adam_eps)
    return result
