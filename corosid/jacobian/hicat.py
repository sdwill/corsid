import logging
from dataclasses import dataclass
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib import colors
from tqdm import tqdm

from corosid.common import AdamOptimizer, TrainingData, batch_linalg as bl
from corosid.common import util
from corosid.common.util import embed, make_z_from_probe_images, today, now
from corosid.jacobian import EstimateClosedLoopJacobian
from corosid.jacobian.least_squares import (EstimationStep, SystemIDResult, least_squares_cost)
from corosid.common.optutil import make_unpacker, pack

log = logging.getLogger(__name__)


def complex_G_to_real(G):
    """
    Reshapes complex-valued (num_pix, num_act) Jacobian to real-valued (num_pix, 2, num_act)

    G0 = np.stack([G.real, G.imag]) -> (2, num_pix, num_act)
    G1 = np.moveaxis(G0, 1, 0) -> (num_pix, 2, num_act)
     """
    return np.moveaxis(np.stack([G.real, G.imag]), 1, 0)  # (num_pix, 2, num_act)


def real_G_to_complex(G):
    """
    Reshapes real-valued (num_pix, 2, num_act) Jacobian to complex-valued (num_pix, num_act)
    """
    return G[:, 0, :] + 1j * G[:, 1, :]

def write_G_to_hicat_format(G, dark_zone, output_path):
    """
    Write out the identified Jacobian to disk in the format that HiCAT uses.

    - G_id is (num_pix, num_act) complex-valued
    - Embed each column into the dark-zone pixels of a 64x64 array (focal-plane array dimensions)
    - Flatten each array into a vector
    - Stack all the vectors into rows of a (num_act, num_pix_square) complex-valued array
    """
    num_act = G.shape[1]
    G_id_stacked = np.stack([embed(G[:, i], dark_zone).ravel()
                             for i in range(num_act)])

    # Stack real and imaginary parts into a (num_act, 2*num_pix_square) real array
    G_to_disk = np.hstack([G_id_stacked.real, G_id_stacked.imag])
    hdul = fits.HDUList([fits.PrimaryHDU()])
    hdul.append(fits.ImageHDU(G_to_disk, name='BOSTON'))
    hdul.writeto(output_path, overwrite=True)


def get_dark_zone(target_dir):
    """
    Figure out the dark zone by looking at the nonzero pixels of the first E-field
    estimate. HiCAT embeds this estimate back into a 2D array.
    """
    efield = fits.getdata(target_dir / 'iteration_0000' / 'estimator0_efield_640nm.fits')
    efield = efield[0] + 1j * efield[1]
    dark_zone = np.abs(efield) > 0
    return dark_zone


def get_command(target_dir, k: int):
    cmd = fits.getdata(target_dir / f'iteration_{k:04d}' / 'total_bmc_correction.fits')
    return cmd.astype(np.float64)


def get_dither(target_dir, k: int):
    dither = fits.getdata(target_dir / f'iteration_{k:04d}' / 'controller1_controller_command.fits')
    return dither.astype(np.float64)


def get_update(target_dir, k: int):
    update = fits.getdata(
        target_dir / f'iteration_{k:04d}' / 'controller0_controller_command.fits')
    # HiCAT uses the convention that
    # total_bmc_correction[1] = total_bmc_correction[0] - controller0_controller_command[0]
    # So negate the update to agree with my sign convention
    return -update.astype(np.float64)


def get_I00(target_dir, k: int, wl: int):
    return fits.getdata(target_dir / f'iteration_{k:04d}' / f'direct_{wl}nm.fits').astype(
        np.float64).max()

def get_probe_matrix(path):
    probe_commands = fits.getdata(path)  # (len_z, len_u)
    return probe_commands.T.astype(np.float64)  # (len_u, len_z)

def get_probe_images(target_dir, k: int, wl: int):
    return fits.getdata(target_dir / f'iteration_{k:04d}' /
                        f'estimator0_probe_imgs_{wl}nm.fits').astype(np.float64)

def load_efc_dataset(target_dir, iterations, wl):
    """
    Load training data from a closed-loop EFC run.
    """
    dark_zone = get_dark_zone(target_dir)
    NUM_PIX = dark_zone.sum()
    DM_NUM_ACT = get_command(target_dir, k=0).size
    # FIXME: hardcoded to 640 nm???
    print('Using alternative load data')

    # (len_u, len_z)
    Psi = get_probe_matrix(target_dir.parent /
                           'probes_paplc_2023-02-17_singleactuators.fits')
    training = TrainingData(
        num_pix=NUM_PIX,
        len_x=2,
        len_z=Psi.shape[1],
        len_u=DM_NUM_ACT,
        num_iter=len(iterations),
        Psi=Psi
    )

    # Construct probe matrix with shape (num_act, len_z)
    start = iterations.start

    # Use iterations 1 to num_iter+1 for training
    for k in tqdm(iterations, desc='Loading data'):
        """
        State-space model assumed by ML:
                  xs[k] = xs[k-1] + G*us[k]
                  zs[k] = H*xs[k]
        When we actually run EFC we estimate, THEN update, which is equivalent to
              zs_efc[k] = H*xs_efc[k]
            xs_efc[k+1] = xs_efc[k] + G*us_efc[k]
        Or equivalently:
              xs_efc[k] = xs_efc[k-1] + G*us_efc[k-1]
              zs_efc[k] = H*xs_efc[k]
        Comparing:
                  zs[k] = zs_efc[k]
                  xs[k] = xs_efc[k]
                  us[k] = us_efc[k-1]
        In practice, we just run EFC for K+1 iterations and use iterations [1, K] as the
        training set, with iteration 0 used as for initialization. So we load data starting
        at EFC iteration 1. But we want training.us and training.zs to start at index 0, so
        shift everything one index to the left (k -> k-1) after loading.
        """

        training.us[k - start] = get_update(target_dir, k - 1)

        I00 = get_I00(target_dir, k, wl)
        probe_imgs = get_probe_images(target_dir, k, wl)
        probe_imgs_dz = probe_imgs[:, dark_zone] / I00
        z = make_z_from_probe_images(probe_imgs_dz)
        training.zs[k - start] = z

    return training

def load_dither_dataset(target_dir, iterations, wl):
    """
    Load training data from an open-loop dataset consisting of random dither commands.
    """
    dark_zone = get_dark_zone(target_dir)
    NUM_PIX = dark_zone.sum()
    DM_NUM_ACT = get_command(target_dir, k=0).size
    # DITHER_K0 = 50  # EFC ends at iteration 49, so iteration 50 is the start of dithering

    Psi = get_probe_matrix(target_dir.parent /
                           'probes_paplc_2023-02-17_singleactuators.fits')

    training = TrainingData(
        num_pix=NUM_PIX,
        len_x=2,
        len_z=Psi.shape[1],
        len_u=DM_NUM_ACT,
        num_iter=len(iterations),
        Psi=Psi
    )

    # Construct probe matrix with shape (num_act, len_z)
    start = iterations.start

    # Use iterations 1 to num_iter+1 for training
    for k in tqdm(iterations, desc='Loading data'):
        """
        State-space model assumed by ML:
                  xs[k] = xs[k-1] + G*us[k]
                  zs[k] = H*xs[k]
        When we actually run EFC we estimate, THEN update, which is equivalent to
              zs_efc[k] = H*xs_efc[k]
            xs_efc[k+1] = xs_efc[k] + G*us_efc[k]
        Or equivalently:
              xs_efc[k] = xs_efc[k-1] + G*us_efc[k-1]
              zs_efc[k] = H*xs_efc[k]
        Comparing:
                  zs[k] = zs_efc[k]
                  xs[k] = xs_efc[k]
                  us[k] = us_efc[k-1]
        In practice, we just run EFC for K+1 iterations and use iterations [1, K] as the
        training set, with iteration 0 used as for initialization. So we load data starting
        at EFC iteration 1. But we want training.us and training.zs to start at index 0, so
        shift everything one index to the left (k -> k-1) after loading.
        """

        # For the HiCAT open-loop dataset, they applied random DM commands at each iteration
        # instead of computing an update with EFC. Therefore, get_update() doesn't work:
        # to difference the commands from the current iteration and the next iteration to find
        # the update from the current iteration. This gives us[k]. We can then proceed according
        # to the state-space model above.

        ### Older code using the dither_commands.fits file
        # k_dither = k - DITHER_K0
        # u_efc_km1 = self.dither[k_dither] - self.dither[k_dither-1]

        u_efc_km1 = get_dither(target_dir, k - 1) - get_dither(target_dir, k - 2)
        training.us[k - start] = u_efc_km1

        I00 = get_I00(target_dir, k, wl)
        probe_imgs = get_probe_images(target_dir, k, wl)
        probe_imgs_dz = probe_imgs[:, dark_zone] / I00
        z = make_z_from_probe_images(probe_imgs_dz)
        training.zs[k - start] = z

    return training


def load_hicat_jacobian(path, dark_zone):
    # Jacobian in FITS format is (num_act, 2*num_pix_square)
    G = fits.getdata(path).T  # (2*num_pix_square, num_act)
    G = G.astype(np.float64)
    G_real, G_imag = np.split(G, 2, axis=0)  # Both are (num_pix_square, num_act)
    return G_real[dark_zone.ravel(), :] + 1j * G_imag[dark_zone.ravel(), :]  # (num_pix, num_act)

@dataclass
class SystemIDResult:
    G: np.ndarray
    costs: np.ndarray
    error: np.ndarray
    x_error: np.ndarray
    z_error: np.ndarray
    estep: EstimationStep

class AdamResult:
    def __init__(self, x):
        self.x = x


class EstimatedClosedLoopHicatJacobian(EstimateClosedLoopJacobian):
    def __init__(self, wl, target_dir, output_dir):
        params_dummy = None  # Don't need this for the HiCAT Jacobian estimation
        super().__init__(params_dummy, wl, target_dir, output_dir)
        self.wl = wl

    def load_data(self, iterations):
        self.training = load_efc_dataset(self.target_dir, iterations, self.wl)


class EstimateOpenLoopHicatJacobian:
    """
    Estimate a HiCAT Jacobian from an open-loop dataset using a least-squares cost function +
    stochastic optimization (with Adam).
    """
    def __init__(self, target_dir, output_dir, G0, wl):
        self.costs = []
        self.error = []
        self.x_error = []
        self.z_error = []
        self.target_dir = target_dir
        self.data = None

        # Starting guess for optimizer, containing only the values that we are optimizing
        self.starting_guess = {'G': G0.astype(np.float64)}
        self.unpack = make_unpacker(self.starting_guess)
        self.estep = None
        self.adam = None

        self.output_dir = output_dir / f'{today()}_{now()}_systemid'
        self.output_dir.mkdir(exist_ok=True, parents=True)
        logging.getLogger().addHandler(
            logging.FileHandler(self.output_dir / 'output.log')
        )
        self.status_dir = self.output_dir / 'status'
        self.status_dir.mkdir(exist_ok=True)
        # self.dither = fits.getdata(self.target_dir / 'dither_commands.fits').astype(np.float64)
        self.wl = wl

    def load_data(self, iterations):
        return load_dither_dataset(self.target_dir, iterations, self.wl)

    def run_estep(self, sol: dict):
        G = sol['G']
        H = 4 * bl.batch_mt(bl.batch_mmip(G, self.data.Psi))

        self.estep.G = G
        self.estep.H = H

        # For least-squares identification (HGu - dz), this is only necessary for evaluating the
        # performance metrics # error, x_error, and z_error
        self.estep.run()
        self.error.append(np.mean(self.estep.eval_error(self.data.zs)))
        self.x_error.append(self.estep.eval_x_err())
        self.z_error.append(self.estep.eval_z_err(self.data.zs))

    def scipy_cost(self, x: np.ndarray):
        sol = self.unpack(x)
        self.run_estep(sol)
        # forward_and_gradient = jax.value_and_grad(least_squares_state_cost, argnums=(0,))
        forward_and_gradient = jax.value_and_grad(least_squares_cost, argnums=(0,))
        J, grads = forward_and_gradient(sol['G'],
                                        self.data.Psi,
                                        self.data.us,
                                        self.estep.xs,
                                        self.data.zs)
        gradient = np.concatenate([np.array(grad).ravel() for grad in grads])
        log.info(f'J: {J:0.3e}\t ||∂g|| = {np.linalg.norm(gradient):0.3e}\t '
                 f'err = {self.error[-1]:0.3e}')
        return J, gradient

    def run(self, num_training, batch_size, training_iter_start, num_epochs,
            adam_alpha, adam_beta1, adam_beta2, adam_eps):
        batch_starts = np.arange(training_iter_start, num_training - batch_size, batch_size)
        x = pack(self.starting_guess)

        for epoch in range(num_epochs):
            log.info(f'Epoch {epoch}')
            np.random.shuffle(batch_starts)
            self.adam = AdamOptimizer(
                self.scipy_cost, x, num_iter=len(batch_starts),
                alpha=adam_alpha,
                beta1=adam_beta1, beta2=adam_beta2, eps=adam_eps)

            for batch_index, batch_start in enumerate(batch_starts):
                log.info(f'Batch {batch_index} of {batch_starts.size}')
                iters_to_load = range(batch_start, batch_start + batch_size)
                self.data = self.load_data(iters_to_load)
                self.estep = EstimationStep(
                    self.data.us,
                    self.data.zs,
                    self.data.num_iter,
                    self.data.num_pix,
                    self.data.len_x,
                    self.data.len_z)
                self.adam.x = x
                J, x = self.adam.iterate(batch_index)

            mean_cost_epoch = np.mean(self.adam.Js)
            self.costs.append(mean_cost_epoch)
            self.make_status_plot(epoch)

        x_unpacked = self.unpack(x)
        final_values = {'G': x_unpacked['G']}

        # Pack results into data structure. Several of these are dummy values, such as the Kalman
        # filter parameters Q, R, x0 and P0, because this algorithm doesn't use a Kalman filter.
        result = SystemIDResult(final_values['G'],
                                costs=np.array(self.costs),
                                error=np.array(self.error),
                                x_error=np.array(self.x_error),
                                z_error=np.array(self.z_error),
                                estep=self.estep)

        return result

    def make_status_plot(self, epoch):
        fig, axs = plt.subplots(dpi=150, layout='constrained', ncols=2, figsize=(7, 3))
        error = np.array(self.error)
        x_error = np.array(self.x_error)
        z_error = np.array(self.z_error)

        ax = axs[0]
        ax.plot(np.arange(error.size), 100 * error, '-', label='Error')
        ax.plot(np.arange(x_error.size), 100 * x_error, '-', label='State error')
        ax.plot(np.arange(z_error.size), 100 * z_error, '-', label='Data error')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Error [%]')

        # Scale the y-axis limits to the largest starting error metric, so that the limits don't
        # change when L-BFGS has a hiccup iteration that causes a temporarily huge error value.
        ax.set_ylim([-5, 105 * np.max([self.error[0], self.x_error[0], self.z_error[0]])])
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


def check_state_transition(estep, dark_zone, k):
    """
    Spot check the state predicted from iteration k-1 against the state estimated in iteration k
    """
    Gu = bl.batch_mvip(estep.G, estep.us[k])
    xm1 = estep.xs[k - 1]  # State estimate in previous iteration
    x_pred = xm1 + Gu      # Predicted state in current iteration
    x = estep.xs[k]        # Actual state estimate in current iteration

    print(100 * util.compare(Gu, x - xm1))
    print(100 * util.compare(x_pred, x))

    I_delta = np.abs(util.make_E_from_state(x) - util.make_E_from_state(xm1)) ** 2
    I_delta = embed(I_delta, dark_zone, mask_output=True)

    I_Gu = embed(np.abs(util.make_E_from_state(Gu)) ** 2, dark_zone, mask_output=True)

    fig, axs = plt.subplots(dpi=150, ncols=2, figsize=(10, 4))
    ax = axs[0]
    ax.imshow(I_Gu, cmap=util.bad_to_black('inferno'), norm=colors.LogNorm(vmin=1e-8, vmax=1e-3))
    # ax.imshow(np.abs(I - coron) / coron,
    #           cmap=bad_to_black('inferno'), norm=colors.Normalize())

    ax = axs[1]
    ax.imshow(I_delta, cmap=util.bad_to_black('inferno'), norm=colors.LogNorm(vmin=1e-8, vmax=1e-3))

    plt.show()

def check_predicted_intensity(estep: EstimationStep, dark_zone: np.ndarray,
                              target_dir: Path, k: int,
                              wl: int):
    """
    Spot check the predicted dark-zone intensity against the measured image (binned down to the
    same array size)
    """
    I00 = get_I00(target_dir, k, wl)
    E = embed(estep.xs[k][:, 0] + 1j * estep.xs[k][:, 1], dark_zone, mask_output=True)
    I = np.abs(E) ** 2
    coron = np.ma.masked_where(
        dark_zone == 0,
        util.bin(fits.getdata(target_dir / f'iteration_{k:04d}' / 'coron_640nm.fits') / I00,
                 binfac=3))

    fig, axs = plt.subplots(dpi=150, ncols=2, figsize=(10, 4))
    ax = axs[0]
    ax.imshow(I, cmap=util.bad_to_black('inferno'), norm=colors.LogNorm(vmin=1e-8, vmax=1e-3))
    # ax.imshow(np.abs(I - coron) / coron,
    #           cmap=bad_to_black('inferno'), norm=colors.Normalize())

    ax = axs[1]
    ax.imshow(coron, cmap=util.bad_to_black('inferno'), norm=colors.LogNorm(vmin=1e-8, vmax=1e-3))

    print(100 * util.compare(I[dark_zone], coron[dark_zone]))


def check_predicted_pairwise_data(estep: EstimationStep, dark_zone, k, probe_number: int):
    """
    Spot check the predicted probe difference images against the measured ones
    """
    z_pred = bl.batch_mvip(estep.H, estep.xs[k])
    z = estep.zs[k]
    p = probe_number  # Alias this to a mathematical variable

    fig, axs = plt.subplots(dpi=150, ncols=2, figsize=(10, 4))
    ax = axs[0]
    ax.imshow(embed(z_pred[:, p], dark_zone, mask_output=True),
              cmap=util.bad_to_black('RdBu'), norm=colors.CenteredNorm())

    ax = axs[1]
    ax.imshow(embed(z[:, p], dark_zone, mask_output=True),
              cmap=util.bad_to_black('RdBu'), norm=colors.CenteredNorm())

    print(100 * util.compare(z, z_pred))
