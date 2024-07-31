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


class ThreeImageTrainingData:
    """ Structure for the training data used for system identification """
    def __init__(
            self,
            dIs: dict = None,       # Change in intensity in each time step relative to previous
            us: dict = None,        # Change in DM command in each time step relative to previous
    ):
        if dIs is None:
            dIs = {}
        self.dIs = dIs

        if us is None:
            us = {}
        self.us = us

def eval_dI_error(G, us, dIs):
    """ Evaluate the error between |G*u[k]|^2 and I+[k] + I-[k] - 2I[k-1] in each time step """
    num_iter = len(list(dIs.keys()))
    dI_err = 0.

    for k in dIs:
        dI = dIs[k]
        u = us[k]
        dI_pred = 2 * np.abs(G @ u) ** 2
        dI_err += util.compare(dI_pred, dI)

    return dI_err / num_iter

@jax.jit
def least_squares_cost(G: np.ndarray, us: dict, dIs: dict):
    """
    Compute the error between the predicted and measured changes in intensity images
    over all time steps.
    """
    G = G.astype(jnp.complex128)

    err = 0.
    dItot = 0.  # To scale the cost function to be invariant to contrast level

    for k in dIs:
        dI = dIs[k]
        u = us[k].astype(jnp.float64)  # Cast to float to avoid Jax overflow bug
        dI_pred = 2 * jnp.abs(G @ u) ** 2
        dIerr = dI_pred - dI
        err = err + d.l2sqr(dIerr)
        dItot = dItot + d.l2sqr(dI)

    J = err / dItot
    return J


@dataclass
class ThreeImageIDResult:
    G: ArrayLike
    costs: ArrayLike
    dI_errors: ArrayLike


def run_batch_least_squares_id(
        data: ThreeImageTrainingData,
        G0: ArrayLike,
        method='L-BFGS-B',
        options: Dict = None,
        tol=1e-3
):
    if options is None:
        options = {'disp': False, 'maxcor': 10, 'maxls': 100}

    costs = []
    dI_errors = []

    # Starting guess for optimizer, containing only the values that we are optimizing
    starting_guess = {'G': G0.astype(np.complex128)}
    unpack = util.make_unpacker(starting_guess)

    def cost_for_optimizer(x):
        sol = unpack(x)
        G = sol['G']
        J, grads = jax.value_and_grad(least_squares_cost, argnums=(0,))(G, data.us, data.dIs)

        gradient = grads[0].ravel()
        gradient = np.concatenate([gradient.real, -gradient.imag])

        # Evaluate quality-of-fit metrics
        costs.append(J)
        dI_errors.append(eval_dI_error(G, data.us, data.dIs))

        log.info(f'J: {J:0.3e}\t ||∂g|| = {np.linalg.norm(gradient):0.3e}')
        return J, gradient

    res = minimize(cost_for_optimizer, x0=util.pack(starting_guess), jac=True,
                   method=method, options=options, tol=tol)
    solution = res.x

    result = ThreeImageIDResult(
        unpack(solution)['G'],
        costs=np.array(costs),
        dI_errors=np.array(dI_errors),
    )

    return result


# class StochasticLeastSquaresID:
#     """
#     Estimate Jacobian using a least-squares cost function and stochastic optimization (with Adam).
#     """
#     def __init__(self, G0, output_dir=None):
#         self.costs = []
#         self.dz_errors = []
#         self.val_errors = []  # Same as dz_errors, but on validation data
#         self.dx_errors = []
#         self.z_errors = []
#         self.data = None  # Training data; set by load_data during run()
#         self.val_data = None  # Validation data; set by load_data during run()
#
#         # Starting guess for optimizer, containing only the values that we are optimizing
#         self.starting_guess = {'G': G0.astype(np.float64)}
#         self.unpack = util.make_unpacker(self.starting_guess)
#         self.adam = None
#
#         # Directory for storing status plots (and to save resulting Jacobian if desired)
#         self.output_dir = None
#         self.status_dir = None
#
#         if output_dir is not None:
#             self.output_dir = output_dir / f'{util.today()}_{util.now()}_systemid'
#             self.output_dir.mkdir(exist_ok=True, parents=True)
#             logging.getLogger().addHandler(
#                 logging.FileHandler(self.output_dir / 'output.log')
#             )
#             self.status_dir = self.output_dir / 'status'
#             self.status_dir.mkdir(exist_ok=True)
#
#     def estimate_states(self, sol: dict):
#         G = sol['G']
#         H = 4 * bl.batch_mt(bl.batch_mmip(G, self.data.Psi))
#
#         # This is only necessary for evaluating the performance metrics (error, x_error, z_error)
#         dxs = estimate_states(H, self.data.dzs)
#         self.dz_errors.append(eval_dz_error(G, H, self.data.us, self.data.dzs))
#         self.dx_errors.append(eval_dx_error(G, dxs, self.data.us))
#         self.z_errors.append(eval_z_error(H, dxs, self.data.dzs))
#
#     def cost_for_optimizer(self, x: np.ndarray):
#         """
#         The actual cost function passed to the optimizer. Accepts a single real-valued vector, x,
#         and returns the value of the cost function, J, as well as its derivative with respect to x,
#         dJ/dx.
#         """
#         sol = self.unpack(x)
#         self.estimate_states(sol)
#         forward_and_gradient = jax.value_and_grad(least_squares_cost, argnums=(0,))
#         J, grads = forward_and_gradient(sol['G'],
#                                         self.data.Psi,
#                                         self.data.us,
#                                         self.data.dzs)
#         gradient = np.concatenate([np.array(grad).ravel() for grad in grads])
#         log.info(f'J: {J:0.3e}\t ||∂g|| = {np.linalg.norm(gradient):0.3e}\t '
#                  f'dz_err = {self.dz_errors[-1]:0.3e}')
#         return J, gradient
#
#     def make_status_plot(self, epoch):
#         batch_size = len(self.dz_errors) // (epoch+1)
#
#         # Average over all of the minibatches to get a mean value for this epoch
#         dz_errors = np.array(self.dz_errors).reshape(-1, batch_size).mean(axis=1)
#         val_errors = np.array(self.val_errors)
#         dx_errors = np.array(self.dx_errors).reshape(-1, batch_size).mean(axis=1)
#         z_errors = np.array(self.z_errors).reshape(-1, batch_size).mean(axis=1)
#
#         epoch_axis = np.arange(epoch+1)
#
#         fig, axs = plt.subplots(dpi=150, layout='constrained', ncols=2, figsize=(7, 3))
#         ax = axs[0]
#         ax.plot(epoch_axis, 100 * val_errors, '-', label='Validation error')
#         ax.plot(epoch_axis, 100 * dz_errors, '-', label='Data transition error')
#         ax.plot(epoch_axis, 100 * dx_errors, '-', label='State transition error')
#         ax.plot(epoch_axis, 100 * z_errors, '-', label='Data error')
#         ax.set_xlabel('Epoch')
#         ax.set_ylabel('Error [%]')
#
#         # Scale the y-axis limits to the largest starting error metric, so that the limits don't
#         # change when L-BFGS has a hiccup iteration that causes a temporarily huge error value.
#         ax.set_ylim([-5, 105 * np.max([dz_errors[0], dx_errors[0], z_errors[0]])])
#         ax.set_title('Convergence', weight='bold')
#         ax.grid(True, linewidth=0.5, linestyle='dotted')
#         ax.legend(loc='best')
#
#         ax = axs[1]
#         ax.plot(np.arange(len(self.costs)), self.costs, 'o-')
#         ax.set_xlabel('Epoch')
#         ax.set_ylabel('Cost')
#
#         ax.set_title(r'Cost vs. training epoch')
#         ax.grid(True, linewidth=0.5, linestyle='dotted')
#
#         if self.status_dir is not None:
#             fig.savefig(self.status_dir / f'epoch_{epoch:03d}')
#             plt.close(fig)
#
#     def run(self, load_data, num_training, batch_size, training_iter_start, num_epochs,
#             validation_iters, adam_alpha, adam_beta1, adam_beta2, adam_eps):
#         batch_starts = np.arange(training_iter_start,
#                                  training_iter_start + num_training - batch_size,
#                                  batch_size)
#         x = util.pack(self.starting_guess)
#         self.val_data = load_data(validation_iters)
#
#         for epoch in range(num_epochs):
#             log.info(f'Epoch {epoch}')
#             np.random.shuffle(batch_starts)
#             self.adam = AdamOptimizer(
#                 self.cost_for_optimizer, x, num_iter=len(batch_starts),
#                 alpha=adam_alpha,
#                 beta1=adam_beta1, beta2=adam_beta2, eps=adam_eps)
#
#             for batch_index, batch_start in enumerate(batch_starts):
#                 log.info(f'Batch {batch_index} of {batch_starts.size}')
#                 iters_to_load = range(batch_start, batch_start + batch_size)
#                 self.data: TrainingData = load_data(iters_to_load)
#                 self.adam.x = x
#                 J, x = self.adam.iterate(batch_index)
#
#             # After each epoch, evaluate error on validation data
#             G = self.unpack(x)['G']
#             H = 4 * bl.batch_mt(bl.batch_mmip(G, self.val_data.Psi))
#             self.val_errors.append(eval_dz_error(G, H, self.val_data.us, self.val_data.dzs))
#
#             mean_cost_epoch = np.mean(self.adam.Js)
#             self.costs.append(mean_cost_epoch)
#             self.make_status_plot(epoch)
#
#         result = LeastSquaresIDResult(self.unpack(x)['G'],
#                                       costs=np.array(self.costs),
#                                       dz_errors=np.array(self.dz_errors),
#                                       dx_errors=np.array(self.dx_errors),
#                                       z_errors=np.array(self.z_errors))
#
#         return result
#
# def run_stochastic_least_squares_id(
#         G0,
#         load_data,
#         num_training,
#         training_iter_start,
#         batch_size,
#         num_epochs,
#         validation_iters,
#         adam_alpha,
#         adam_beta1,
#         adam_beta2,
#         adam_eps,
#         output_dir=None
# ):
#     system_id = StochasticLeastSquaresID(G0, output_dir=output_dir)
#     result = system_id.run(load_data, num_training, batch_size, training_iter_start, num_epochs,
#                            validation_iters, adam_alpha, adam_beta1, adam_beta2, adam_eps)
#     return result
