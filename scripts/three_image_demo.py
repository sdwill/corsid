"""
NOTE: This demo script is meant to be run as a series of cells, using e.g. Spyder or PyCharm (in
scientific mode).
"""
import numpy as np
import corsid.batch_linalg as bl
import matplotlib.pyplot as plt
from corsid.util import compare
import corsid.three_image as ti

from scipy.linalg import hadamard
# np.random.seed(3794)

num_pix = 10  # Number of focal-plane pixels
len_u = 8  # Number of DM actuators
len_x = 2
num_iter = 20  # Number of time steps of simulated data
num_training_iter = 16   # Number of time steps to use for training
training_iter_start = 0  # Starting time step for training data

training_iters = range(num_training_iter)
validation_iters = range(num_training_iter, num_iter)

# Real-valued form of Jacobian: each pixel has a 2x(len_u) Jacobian for its real and imagiary parts
G = np.random.randn(num_pix, len_u) + 1j * np.random.randn(num_pix, len_u)
R = 1e-3  # Measurement noise covariance magnitude
Q = 1e-3  # Process noise covariance magnitude


#%% Simulate some data

def simulate_data():
    x0 = np.random.randn(num_pix) + 1j * np.random.randn(num_pix)
    I0 = np.abs(x0) ** 2

    # Excite the system with Hadamard modes
    cmds = hadamard(2**np.ceil(np.log2(num_iter)))
    cmds = {j: cmds[:, j][:len_u] for j in range(num_iter)}
    dIs = {}
    us = {}
    # cmds = {k: np.random.randn(len_u) for k in range(num_iter)}

    for j in range(num_iter):
        us[j] = cmds[j]

        # FIXME add process noise, instability
        I_plus = np.abs(x0 + G @ us[j]) ** 2
        I_minus = np.abs(x0 - G @ us[j]) ** 2
        dIs[j] = I_plus + I_minus - 2*I0  # FIXME Replace I0 with a new, uncommanded image at each j

    return us, dIs

us, dIs = simulate_data()

#%% Run batch optimization to identify Jacobian

training_data = ti.ThreeImageTrainingData(
    dIs={j: dIs[j] for j in training_iters},
    us={j: us[j] for j in training_iters},
)
G_err = np.random.randn(*G.shape) + 1j * np.random.randn(*G.shape)
G0 = G + 0.5 * G_err  # Simulate imperfect starting knowledge

result = ti.run_batch_least_squares_id(
    data=training_data,
    G0=G0,
    method='L-BFGS-B',
    options=dict(disp=True),
    tol=1e-6
)

#%% Evaluate results from batch optimization

def evaluate_results(G0, G_id, training_data, validation_data):
    print(f'  Starting error in dI (training): '
          f'{100*ti.eval_dI_error(G0, training_data.us, training_data.dIs):0.2f}%')
    print(f'     Final error in dI (training): '
          f'{100*ti.eval_dI_error(G_id, training_data.us, training_data.dIs):0.2f}%')
    print(f'   Minimum error in dI (training): ',
          f'{100*ti.eval_dI_error(G, training_data.us, training_data.dIs):0.2f}%')

    print(f'Starting error in dI (validation): '
          f'{100*ti.eval_dI_error(G0, validation_data.us, validation_data.dIs):0.2f}%')
    print(f'   Final error in dI (validation): '
          f'{100*ti.eval_dI_error(G_id, validation_data.us, validation_data.dIs):0.2f}%')
    print(f' Minimum error in dI (validation): '
          f'{100*ti.eval_dI_error(G, validation_data.us, validation_data.dIs):0.2f}%')

    # Compare the identified Jacobian to the true Jacobian
    print(f'     Starting error relative to G: {100*compare(G0, G):0.2f}%')
    print(f'        Final error relative to G: {100*compare(G_id, G):0.2f}%')

    print(' Starting error relative to G.T*G: {:0.2f}%'.format(
        100*compare(G0.conj().T @ G0, G.conj().T @ G)))
    print('    Final error relative to G.T*G: {:0.2f}%'.format(
        100*compare(G_id.conj().T @ G_id, G.conj().T @ G)))


validation_data = ti.ThreeImageTrainingData(
    dIs={j: dIs[j] for j in validation_iters},
    us={j: us[j] for j in validation_iters},
)

evaluate_results(G0, result.G, training_data, validation_data)


#%% Run stochastic optimization using Adam

def load_data(iters_to_load):
    """
    Interface function that loads a minibatch from "disk". Here, the simulation is small enough to
    store the entire training set in RAM and select subsets. In a real experiment, this may not be
    possible. If so, each minibatch is loaded from disk when needed.
    """
    dzs_to_load = {}
    us_to_load = {}

    iters_to_load = list(iters_to_load)
    start = iters_to_load[0]
    for k in iters_to_load:
        dzs_to_load[k-start] = zs[k]-zs[k-1]
        us_to_load[k-start] = us[k]

    return TrainingData(
        num_pix=num_pix,
        len_x=len_x,
        len_z=len_z,
        len_u=len_u,
        num_iter=len(iters_to_load),
        dzs=dzs_to_load,
        us=us_to_load,
        Psi=Psi
    )


result = ls.run_stochastic_least_squares_id(
    G0, load_data,
    num_training=num_iter,
    training_iter_start=1,
    batch_size=4,
    num_epochs=20,
    validation_iters=validation_iters,
    adam_alpha=5e-2,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_eps=1e-8,
    output_dir=None
)

#%% Evaluate the results of the stochastic optimizer
evaluate_results(G0, Psi, result.G, training_data, validation_data)
plt.show()
