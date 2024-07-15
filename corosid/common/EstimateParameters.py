import logging
from types import SimpleNamespace

import asdf
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from tqdm.autonotebook import tqdm

from corosid.common.util import make_z_from_probe_images, today, now, compare, embed

log = logging.getLogger(__name__)


class BaseEstimateParameters:
    def __init__(self, params, wr, target_dir, output_dir):
        self.params = params
        self.wr = wr
        self.target_dir = target_dir
        self.output_dir = output_dir

        self.error = np.array([])
        self.x_error = np.array([])
        self.z_error = np.array([])
        self.likelihoods = np.array([])
        self.history = []
        self.solution = None
        self.training = None  # Training data
        self.tree = None  # Parameter values

        self.output_dir = output_dir / f'{today()}_{now()}_systemid'
        self.output_dir.mkdir(exist_ok=True, parents=True)
        logging.getLogger().addHandler(
            logging.FileHandler(self.output_dir / 'output.log')
        )

    def load_data(self, iterations):
        """
        Load training data from disk and package it into a simple data structure.
        """
        log.info('Loading training data from disk...')

        # FIXME 9/19/23: num_iterator and len_z are constants; should be discoverable from data or
        #  settable by user
        training = SimpleNamespace()
        training.num_pix = self.params.dark_zone.sum()
        training.num_act = self.params.dm_num_act
        training.len_x = 2  # length of state vector
        training.len_z = 2  # number of probe pairs
        training.len_u = training.num_act  # length of control vector
        training.num_iter = len(iterations)  # Number of iterations of training data
        training.zs = {}
        training.us = {}

        # Construct probe matrix with shape (num_act, len_z)
        probe_commands = fits.getdata(self.target_dir / 'probe_commands.fits')
        training.Psi = np.stack([probe_commands[n].ravel() for n in range(training.len_z)]).T
        # probe_commands_2dm = [
        #     np.concatenate([probe_commands[n].ravel(), np.zeros_like(probe_commands[n].ravel())])
        #     for n in range(len_z)]  # 2-element list, each element is an (num_act,) array
        # training.Psi = np.stack(probe_commands_2dm).T  # (num_act, len_z)
        start = iterations.start

        # Use iterations 1 to num_iter+1 for training
        for k in tqdm(iterations, desc='Loading data'):
            # State-space model assumed by ML:
            #           xs[k] = xs[k-1] + G*us[k]
            #           zs[k] = H*xs[k]
            # When we actually run EFC we estimate, THEN update, which is equivalent to
            #       zs_efc[k] = H*xs_efc[k]
            #     xs_efc[k+1] = xs_efc[k] + G*us_efc[k]
            # Or equivalently:
            #       xs_efc[k] = xs_efc[k-1] + G*us_efc[k-1]
            #       zs_efc[k] = H*xs_efc[k]
            # Comparing:
            #           zs[k] = zs_efc[k]
            #           xs[k] = xs_efc[k]
            #           us[k] = us_efc[k-1]
            # In practice, we just run EFC for K+1 iterations and use iterations [1, K] as the
            # training set, with iteration 0 used as for initialization. So we load data starting
            # at EFC iteration 1. But we want training.us and training.zs to start at index 0, so
            # shift everything one index to the left (k -> k-1) after loading.

            training.us[k - start] = fits.getdata(
                self.target_dir / f'iter{k - 1:04d}' / 'command_update.fits').ravel().astype(
                np.float64)
            probe_images = fits.getdata(
                self.target_dir / f'iter{k:04d}' / f'probe_images_wr={self.wr}.fits')
            training.zs[k - start] = make_z_from_probe_images(probe_images)

            self.training = training

    def estimate(self, targets, tol, algorithm=''):
        pass

    def estimate_noise(self, num_em_iter, targets=('Q', 'R')):
        pass

    def show_results(self):
        """
        Show convergence (likelihood and training error) and print out identified noise statistics
        """
        log.info('Displaying results...')

        # zs_id = {k: batch_mvip(self.solution.estep.H, self.solution.estep.xs[k]) for k in
        #          self.solution.estep.xs}
        #
        # fig, ax = plt.subplots(dpi=150, layout='constrained')
        # ax.plot(np.stack([self.training.zs[k][0, 0] for k in self.training.zs]),
        #         label='True', color='k', linewidth=2)
        # ax.plot(np.stack([zs_id[k][0, 0] for k in self.training.zs]),
        #         label='Identified', color='C0')
        # ax.legend(loc='best')
        # ax.grid(True, linewidth=0.5, linestyle='dotted')

        fig, axs = plt.subplots(dpi=150, layout='constrained', ncols=2, figsize=(10, 5))
        ax = axs[0]
        ax.plot(np.arange(len(self.error)), 100 * self.error, 'o-', label='Error')
        ax.plot(np.arange(len(self.x_error)), 100 * self.x_error, 'o-', label='State error')
        ax.plot(np.arange(len(self.z_error)), 100 * self.z_error, 'o-', label='Data error')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Error [%]')

        # Scale the y-axis limits to the largest starting error metric, so that the limits don't
        # change when L-BFGS has a hiccup iteration that causes a temporarily huge error value.
        ax.set_ylim([0, 105 * np.max([self.error[0], self.x_error[0], self.z_error[0]])])
        ax.set_title('Convergence', weight='bold')
        ax.grid(True, linewidth=0.5, linestyle='dotted')
        ax.legend(loc='best')

        ax = axs[1]
        ax.plot(np.arange(len(self.likelihoods)), self.likelihoods, 'o-')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Log-likelihood')

        # Scale the y-axis limits to to starting cost function value, so that the limits don't
        # change when L-BFGS has a hiccup iteration.
        ax.set_ylim([0.95 * self.likelihoods[0], None])
        ax.set_title(r'$\mathbf{log\; p(z|\theta)}$')
        ax.grid(True, linewidth=0.5, linestyle='dotted')

        for ax in axs:
            for k in np.cumsum([history[1] for history in self.history]):
                ax.axvline(k-1, ls='dashed', lw=1.5, color='red')

        log.info(f"""
Convergence for wr = {self.wr}:
        Training error (before): {self.error[0]:0.3f} 
         Training error (after): {self.error[-1]:0.3f} 
State transition error (before): {self.x_error[0]:0.3f} 
 State transition error (after): {self.x_error[-1]:0.3f} 
 Data prediction error (before): {self.z_error[0]:0.3f} 
  Data prediction error (after): {self.z_error[-1]:0.3f} 
                 Log-likelihood: {self.likelihoods[-1]:2.3e}
                              Q: {self.tree["Q"]:2.3e}
                              R: {self.tree["R"]:2.3e}
        """)

        plt.savefig(self.output_dir / f'convergence_wr={self.wr}.png')
        plt.show()

    def save_results(self):
        """
        Write identification results to disk
        """
        log.info('Saving identified parameter values to disk...')
        af = asdf.AsdfFile(self.tree)
        af.write_to(self.output_dir / f'results_wr={self.wr}.asdf')

    def check_initial_intensity(self):
        """
        Compare the dark-zone intensity in iteration 0 predicted by the estimated E-field to the
        measured intensity.

        Run this before identification to use the initial guess for the initial state, or after
        identification to use the smoothed initial state estimate.
        """
        from matplotlib import colors

        x0 = self.tree['x0']

        # E0_id = tree['x0'][:, 0] + 1j * tree['x0'][:, 1]
        E0 = x0[:, 0] + 1j * x0[:, 1]
        I0_id_dz = np.abs(embed(E0, self.params.dark_zone) ** 2)
        I0_dz = np.ma.masked_where(self.params.dark_zone == 0,
                                   fits.getdata(self.target_dir / 'psf_start.fits'))

        plt.figure()
        plt.imshow(I0_id_dz, cmap='inferno', norm=colors.LogNorm(vmin=1e-11, vmax=1e0))
        plt.figure()
        plt.imshow(I0_dz, cmap='inferno', norm=colors.LogNorm(vmin=1e-11, vmax=1e0))
        plt.show()

        print(compare(I0_id_dz[self.params.dark_zone], I0_dz[self.params.dark_zone]))

