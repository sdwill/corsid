from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TARGET_DIR = (Path.home() / 'Box' / 'Scott\'s' /
              '2024-03-01_systemID_vs_model_jacobian_convergence_runs')

apriori = np.load(TARGET_DIR / 'raw_contrast_model_jacobian_rcond10_to_04.npz')['arr_0']
# apriori = np.load(TARGET_DIR / 'raw_contrast_model_jacobian_rcond10_to_08.npz')['arr_0']
# apriori = np.load(TARGET_DIR / 'raw_contrast_model_jacobian_rcond50_to_04.npz')['arr_0']
sysID = np.load(TARGET_DIR / 'raw_contrast_systemID_jacobian_5000iter_dithers_rcond10_to_08.npz')[
    'arr_0']
# sysID = np.load(TARGET_DIR / 'raw_contrast_systemID_jacobian_5000iter_dithers_rcond50_to_07.npz')[
#     'arr_0']

fig, ax = plt.subplots(dpi=250)
ax.semilogy(apriori, 'k', label='Model Jacobian')
ax.semilogy(sysID, 'C1--', label='Identified Jacobian')
ax.grid(True, lw=0.5, ls='dotted', which='both')
ax.set_xlabel('Control iteration')
ax.set_ylabel('Dark zone contrast')
ax.legend(loc='best')
ax.set_ylim([None, 1e-6])
ax.set_title('System identification on HiCAT')

plt.savefig(Path.home() / 'Box/projects/FY23-25 APRA/2023-12 HiCAT Jacobian identification/'
                          '20240306_hicat_results.pdf')
