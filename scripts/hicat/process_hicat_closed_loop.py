"""
Estimate HiCAT Jacobian from closed-loop EFC data.
"""
import logging
from pathlib import Path

import matplotlib
import numpy as np

from corosid.common.util import today
from corosid.jacobian import hicat

matplotlib.use('TkAgg')

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DATA_ROOT = Path.home() / 'Box/projects/FY23-25 APRA/2023-12 HiCAT jacobian identification'
RESULTS_DIR = DATA_ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)
output_dir = RESULTS_DIR / f'{today()}'
WLS = [640]

target_dir = DATA_ROOT / '2023-12-07T16-32-15_wavefront_control_experiment_w_latest_flatmap_rcond04_trun15'
dark_zone = hicat.get_dark_zone(target_dir)
num_pix = dark_zone.sum()
len_x = 2
ML_ITER_START = 0
NUM_TRAINING_ITER = 11

Gs = {}
for wl in WLS:
    Gs[wl] = hicat.load_hicat_jacobian(DATA_ROOT / f'jacobian_2023-12-07_aplc_nonconj_{wl}nm.fits',
                                       dark_zone)

    tree = {wl: {
        'G': Gs[wl],
        'Q': 1e-3,
        'R': 1e-13,
        'x0': np.zeros((num_pix, len_x)),
        'P0': 1.
    } for wl in WLS}

    system_id = hicat.EstimatedClosedLoopHicatJacobian(wl=wl,
                                                       target_dir=target_dir,
                                                       output_dir=output_dir)
    system_id.load_data(range(ML_ITER_START + 1, ML_ITER_START + NUM_TRAINING_ITER))
    system_id.tree = {key: tree[wl][key] for key in tree[wl]}
    Gamma = hicat.complex_G_to_real(system_id.tree['G'])
    system_id.tree['G'] = Gamma

    system_id.estimate(['G'], tol=1e-16, algorithm='pem')
    # for _ in range(2):
    #     system_id.estimate(['Q', 'R'], tol=1e-6, algorithm='ml')
    #     # system_id.estimate(['Q', 'R'], tol=1e-6, algorithm='ml')
    #     system_id.estimate(['G'], tol=1e-6, algorithm='ml')

    system_id.show_results()
    # system_id.save_results()

    tree[wl]['G'] = hicat.real_G_to_complex(system_id.tree['G'])
    tree[wl]['Q'] = system_id.tree['Q']
    tree[wl]['R'] = system_id.tree['R']
    tree[wl]['x0'] = system_id.tree['x0']
    tree[wl]['P0'] = system_id.tree['P0']

    output_path = output_dir / f'jacobian_identified_{today()}_{wl}nm.fits'
    hicat.write_G_to_hicat_format(tree[wl]['G'], dark_zone, system_id.output_dir)

