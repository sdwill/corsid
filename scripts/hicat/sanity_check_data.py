"""
1208.H.3
"""
import logging
from pathlib import Path

import matplotlib

from corosid.common.util import today
from corosid.jacobian import hicat

matplotlib.use('TkAgg')

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DATA_ROOT = Path.home() / 'Box/projects/FY23-25 APRA/2023-12 HiCAT jacobian identification'
RESULTS_DIR = DATA_ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = RESULTS_DIR / f'{today()}'
TARGET_DIR = DATA_ROOT / '2023-12-08T17-25-13_wavefront_control_experiment'

WLS = [640]
wl = WLS[0]
DARK_ZONE = hicat.get_dark_zone(TARGET_DIR)

G0 = hicat.complex_G_to_real(
    hicat.load_hicat_jacobian(DATA_ROOT / f'jacobian_2023-12-07_aplc_nonconj_{wl}nm.fits',
                              DARK_ZONE))

system_id = hicat.EstimatedClosedLoopHicatJacobian(wr=wl,
                                                   target_dir=TARGET_DIR,
                                                   output_dir=OUTPUT_DIR)

#%%
from corosid.jacobian.least_squares import EstimationStep
from corosid.common import batch_linalg as bl

k0 = 38
num_iters_to_check = 10
iters_to_load = range(k0, k0+num_iters_to_check)
system_id.load_data(iters_to_load)
data = system_id.training

estep = EstimationStep(
    data.us,
    data.zs,
    data.num_iter,
    data.num_pix,
    data.len_x,
    data.len_z)
estep.G = G0
estep.H = 4 * bl.batch_mt(bl.batch_mmip(G0, data.Psi))

estep.run()
print(estep.eval_error(data.zs))
print(estep.eval_x_err())
print(estep.eval_z_err(data.zs))

#%% Spot check state transition
k = k0+8
hicat.check_state_transition(estep, DARK_ZONE, k-k0)

#%% Spot check predicted intensity
k = k0+8
hicat.check_predicted_intensity(estep, DARK_ZONE, TARGET_DIR, k-k0, wl)

#%% Spot check pairwise data
k = k0+8
p = 0  # Probe index
hicat.check_predicted_pairwise_data(estep, DARK_ZONE, k-k0, p)