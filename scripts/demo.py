"""
NOTE: This demo script is meant to be run as a series of cells, using e.g. Spyder or PyCharm (in
scientific mode).
"""
import numpy as np
import corsid.batch_linalg as bl
import matplotlib.pyplot as plt
from corsid.util import compare
from corsid import TrainingData
import corsid.least_squares as ls

from scipy.linalg import hadamard

num_pix = 10  # Number of focal-plane pixels
len_x = 2  # Length of state vector at each pixel (always 2: real/imaginary part of the E-field)
len_z = 2  # Length of data vector at each pixel (number of pairwise probe pairs)
len_u = 8  # Number of DM actuators
num_iter = 20  # Number of time steps of simulated data
num_training_iter = 16   # Number of time steps to use for training
training_iter_start = 0  # Starting time step for training data

training_iters = range(num_training_iter)
validation_iters = range(num_training_iter, num_iter)

# Real-valued form of Jacobian: each pixel has a 2x(len_u) Jacobian for its real and imagiary parts
G = np.random.randn(num_pix, len_x, len_u)
R = 1e-3  # Measurement noise covariance magnitude
Q = 1e-3  # Process noise covariance magnitude

#%% Optimize probe functions
# The performance of the system ID algorithm strongly depends on the choice of probes- bad
# probe choice creates a poorly-conditioned observation matrix, and therefore low-quality data
# Since we have a random Jacobian in this example, we have no physics-based intuition for good
# probes, so implement a simple condition-number minimization scheme to numerically find probes.
Psi0 = np.random.randn(len_u, len_z)  # Matrix of probe commands (each command is one column)
H0 = 4*bl.batch_mt(bl.batch_mmip(G, Psi0))  # Pairwise observation matrix: 4(G*Psi).T

def optimize_probes():
    from scipy.optimize import minimize
    from corsid import differentiable as d
    import jax.numpy as jnp
    import jax

    def cost_for_jax(Psi):
        H = 4*d.batch_mt(d.batch_mmip(G, Psi))
        return jnp.linalg.cond(H).mean()

    def cost_for_optimizer(x):
        Psi = x.reshape((len_u, len_z))
        J, grad = jax.value_and_grad(cost_for_jax)(Psi)
        return J, grad.ravel()

    x0 = Psi0.ravel()
    res = minimize(cost_for_optimizer, x0=x0, method='L-BFGS-B', options=dict(disp=True), tol=1e-5,
                   jac=True)
    return res.x.reshape(Psi0.shape)

Psi = optimize_probes()
H = 4*bl.batch_mt(bl.batch_mmip(G, Psi))  # Pairwise observation matrix: 4(G*Psi).T
print(f'Mean condition number before optimization: {np.linalg.cond(H0).mean():0.2f}')
print(f'Mean condition number after optimization: {np.linalg.cond(H).mean():0.2f}')

#%% Simulate some data

def simulate_data():
    x0 = np.zeros((num_pix, len_x))

    # Excite the system with Hadamard modes
    # us = hadamard(len_u)
    # us = {k: us[:, k] for k in range(len_u)}

    us = {}
    xs = {-1: x0}
    zs = {}
    for k in range(num_iter):
        us[k] = np.random.randn(len_u)
        xs[k] = xs[k-1] + bl.batch_mvip(G, us[k]) + np.sqrt(Q) * np.random.randn(num_pix, len_x)
        zs[k] = bl.batch_mvip(H, xs[k]) + np.sqrt(R) * np.random.randn(num_pix, len_z)

    xs.pop(-1)

    return xs, us, zs


xs, us, zs = simulate_data()

#%% Run batch optimization to identify Jacobian

training_data = TrainingData(
    num_pix=num_pix,
    len_x=len_x,
    len_z=len_z,
    len_u=len_u,
    num_iter=num_iter,
    zs={k: zs[k] for k in training_iters},
    us={k: us[k] for k in training_iters},
    Psi=Psi
)

G0 = G + np.random.randn(*G.shape)*0.5  # Simulate imperfect starting knowledge

result = ls.run_batch_least_squares_id(
    data=training_data,
    G0=G0,
    method='L-BFGS-B',
    options=dict(disp=True),
    tol=1e-6
)

#%% Evaluate results from batch optimization

def evaluate_results(G0, Psi, G_id, training_data, validation_data):
    H_id = 4*bl.batch_mt(bl.batch_mmip(G_id, Psi))

    # Compare the error in the predicted data changes vs. the observed data changes
    H0 = 4*bl.batch_mt(bl.batch_mmip(G0, Psi))
    print(f'  Starting error in dz (training): '
          f'{100*ls.eval_dz_error(G0, H0, training_data.us, training_data.zs):0.2f}%')
    print(f'     Final error in dz (training): '
          f'{100*ls.eval_dz_error(G_id, H_id, training_data.us, training_data.zs):0.2f}%')
    print(f'Starting error in dz (validation): '
          f'{100*ls.eval_dz_error(G0, H0, validation_data.us, validation_data.zs):0.2f}%')
    print(f'   Final error in dz (validation): '
          f'{100*ls.eval_dz_error(G_id, H_id, validation_data.us, validation_data.zs):0.2f}%')

    # Compare the identified Jacobian to the true Jacobian
    print(f'     Starting error relative to G: {100*compare(G0, G):0.2f}%')
    print(f'        Final error relative to G: {100*compare(G_id, G):0.2f}%')

    # Note that the identified Jacobian, in general, will be in a different basis than the true
    # Jacobian, because for any orthonormal transformation P, P*G is consistent with the observed
    # data. But G.T * G = (P*G).T * (P*G), so compare against that to see how close we got to the
    # truth, regardless of basis transformations
    print(' Starting error relative to G.T*G: {:0.2f}%'.format(
        100*compare(bl.batch_mmip(bl.batch_mt(G0), G0), bl.batch_mmip(bl.batch_mt(G), G))))
    print('    Final error relative to G.T*G: {:0.2f}%'.format(
        100*compare(bl.batch_mmip(bl.batch_mt(G_id), G_id), bl.batch_mmip(bl.batch_mt(G), G))))

validation_data = TrainingData(
    num_pix=num_pix,
    len_x=len_x,
    len_z=len_z,
    len_u=len_u,
    num_iter=num_iter,
    zs={k-num_training_iter: zs[k] for k in validation_iters},
    us={k-num_training_iter: us[k] for k in validation_iters},
    Psi=Psi
)
evaluate_results(G0, Psi, result.G, training_data, validation_data)

#%% Run stochastic optimization using Adam

def load_data(iters_to_load):
    """
    Interface function that loads a minibatch from "disk". Here, the simulation is small enough to
    store the entire training set in RAM and select subsets. In a real experiment, this may not be
    possible. If so, each minibatch is loaded from disk when needed.
    """
    zs_to_load = {}
    us_to_load = {}

    iters_to_load = list(iters_to_load)
    start = iters_to_load[0]
    for k in iters_to_load:
        zs_to_load[k-start] = zs[k]
        us_to_load[k-start] = us[k]

    return TrainingData(
        num_pix=num_pix,
        len_x=len_x,
        len_z=len_z,
        len_u=len_u,
        num_iter=len(iters_to_load),
        zs=zs_to_load,
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