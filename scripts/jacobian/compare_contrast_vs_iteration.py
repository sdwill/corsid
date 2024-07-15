import re
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('TkAgg')


def parse_log(path):
    contrast_re = re.compile('Mean contrast after iteration *')

    results = {}
    with open(path, 'r') as file:
        for line in file:
            match = contrast_re.findall(line)
            if match:
                parts = line.split(' ')
                contrast = float(parts[-1])
                iteration = int(parts[4].split(':')[0])
                results[iteration] = contrast
    return results

def parse_ecsv(path):
    from astropy.io import ascii
    table = ascii.read(path, format='ecsv')
    data_dict = {}
    for key in table.keys():
        data_dict[key] = np.asarray(table[key])
    return data_dict


#%%
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# RESULTS_DIR = Path.home() / 'Box/projects/FY23-25 APRA/2023-06 jacobian-based wfsc/results'
RESULTS_DIR = Path.home() / 'Box/projects/FY23-25 APRA/results'

truth = parse_ecsv(str(RESULTS_DIR / '2024-02-26' / '2024-02-26_09-15-24_wfsc' / 'results.txt'))
prior = parse_ecsv(str(RESULTS_DIR / '2024-02-27' / '2024-02-27_16-28-11_wfsc' / 'results.txt'))
identified = parse_ecsv(RESULTS_DIR / '2024-02-27' / '2024-02-27_15-38-46_wfsc-id-continue' /
                        '2024-02-27_15-38-46_wfsc' / 'results.txt')
# # 1213.1
# truth = parse_ecsv(RESULTS_DIR / '2023-12-13' / '2023-12-13_08-28-00_wfsc' / 'results.txt')
#
# # 1212.6
# prior = parse_ecsv(RESULTS_DIR / '2023-12-12' / '2023-12-12_15-03-14_wfsc' / 'results.txt')
#
# # 1212.7
# pem_data = parse_ecsv(RESULTS_DIR / '2023-12-12' / '2023-12-12_16-25-07_wfsc-id-continue' /
#                       '2023-12-12_16-25-07_wfsc' / 'results.txt')
#
# # 1213.2
# pem_state = parse_ecsv(RESULTS_DIR / '2023-12-13' / '2023-12-13_08-41-28_wfsc-id-continue' /
#                        '2023-12-13_08-41-28_wfsc' / 'results.txt')
#
# # 1212.8
# ml = parse_ecsv(RESULTS_DIR / '2023-12-12' / '2023-12-12_16-52-44_wfsc-id-continue' /
#                 '2023-12-12_16-52-44_wfsc' / 'results.txt')
#
# # 1213.4
# em = parse_ecsv(RESULTS_DIR / '2023-12-13' / '2023-12-13_10-05-10_wfsc-id-continue' /
#                 '2023-12-13_10-05-10_wfsc' / 'results.txt')
#
fig, ax = plt.subplots(dpi=150, figsize=(6, 4))
ax.semilogy(truth['iteration'], truth['mean_contrast'],
            lw=2, label='True model', color='k')
ax.semilogy(prior['iteration'], prior['mean_contrast'],
            lw=2, ls='dashdot', label='Mismatched model')
ax.semilogy(identified['iteration'], identified['mean_contrast'],
            lw=2, ls='dashdot', label='Identified model')
# ax.semilogy(pem_data['iteration'], pem_data['mean_contrast'],
#             lw=2, ls='dashdot', label='PEM (data-domain)')
# ax.semilogy(pem_state['iteration'], pem_state['mean_contrast'],
#             lw=2, ls='dashdot', label='PEM (state-domain)')
# ax.semilogy(ml['iteration'], ml['mean_contrast'],
#             lw=2, ls='dashed', label='ML')
# ax.semilogy(em['iteration'][:20], em['mean_contrast'][:20],
#             lw=2, ls='dashed', label='EM (diverged)')
ax.spines['top'].set_linewidth(1.5)    # Top frame
ax.spines['bottom'].set_linewidth(1.5)  # Bottom frame
ax.spines['left'].set_linewidth(1.5)   # Left frame
ax.spines['right'].set_linewidth(1.5)  # Right frame

for k in [10, 20, 30]:
    ax.annotate('Jacobian update',
                xy=(k, identified['mean_contrast'][k]),
                xytext=(k + 1, identified['mean_contrast'][k] * 5),
                fontsize=8,
                color='r',
                ha='left',
                va='bottom',
                arrowprops=dict(arrowstyle='fancy', facecolor='r', edgecolor='r', relpos=(0, 0),
                                linewidth=0.1, shrinkA=0),
                bbox=dict(pad=2, facecolor='none', edgecolor='none')
                )
ax.legend(loc='best')
ax.set_xticks(np.arange(0, 41, 5))
ax.grid(True, lw=0.5, ls='dotted')
ax.set_xlabel('WFS&C iteration')
ax.set_ylabel('Dark-zone contrast')
ax.set_title('Closed-loop system identification', weight='bold', y=1.01)

today = datetime.now()
output_dir = RESULTS_DIR / today.strftime('%Y-%m-%d')
output_dir.mkdir(parents=True, exist_ok=True)
# plt.savefig(RESULTS_DIR / )
plt.savefig(output_dir /
            f'contrast_comparison_{today.strftime("%Y%m%d")}.pdf')
plt.show()

#%%
from astropy.io import ascii
ecsv_file_path = (RESULTS_DIR / '2023-12-13' / '2023-12-13_08-41-28_wfsc-id-continue' /
                  '2023-12-13_08-41-28_wfsc' / 'results.txt')
table = ascii.read(ecsv_file_path, format='ecsv')