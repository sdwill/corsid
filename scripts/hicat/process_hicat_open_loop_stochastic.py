"""
Estimate the HiCAT Jacobian from open-loop training datasets using a stochastic optimizer.
"""
import logging
from pathlib import Path

import matplotlib

from corosid.common.util import today
from corosid.jacobian import hicat

matplotlib.use('agg')

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DATA_ROOT = Path.home() / 'Research/torch_projects/WFSC'
RESULTS_DIR = DATA_ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)
output_dir = RESULTS_DIR / f'{today()}'
WLS = [620]

SPEM_TARGET_DIR = DATA_ROOT / '2024-01-20T15-55-46_wavefront_control_experiment'
SPEM_NUM_EPOCHS = 30
SPEM_BATCH_SIZE = 100
ADAM_ALPHA = 1e-8
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8
NUM_TRAINING_ITER = 3500
TRAINING_ITER_START = 200  # Iteration where EFC stops and dithering begins
VALIDATION_ITERS = range(3500, 4000)

dark_zone = hicat.get_dark_zone(SPEM_TARGET_DIR)

Gs = {}
for wl in WLS:
    Gs[wl] = hicat.load_hicat_jacobian(DATA_ROOT /
                                       f'jacobians/jacobian_2024-01-16_aplc_nonconj_{wl}nm.fits',
                                       dark_zone)
    G_id = hicat.complex_G_to_real(Gs[wl])
    spem = hicat.EstimateOpenLoopHicatJacobian(
        SPEM_TARGET_DIR, output_dir, G0=G_id, wl=wl
    )
    spem.data = spem.load_data(VALIDATION_ITERS)

    sol = {
        'G': G_id
    }
    spem.run_estep(sol)
    zerr = spem.error[-1]

    #
    # system_id_result = spem.run(NUM_TRAINING_ITER,
    #                             SPEM_BATCH_SIZE,
    #                             TRAINING_ITER_START,
    #                             SPEM_NUM_EPOCHS,
    #                             ADAM_ALPHA, ADAM_BETA1, ADAM_BETA2, ADAM_EPS)
    #
    # G_final = hicat.real_G_to_complex(system_id_result.G)
    #
    # hicat.write_G_to_hicat_format(G_final, dark_zone,
    #                               output_dir / f'jacobian_identified_{today()}_{wl}nm.fits')
    #
