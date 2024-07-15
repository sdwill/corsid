# Algorithm parameters
EM_NUM_ITER = 3

# For system ID, use adam or scipy.minimize.optimize?
USE_ADAM = False
USE_SCIPY = True

# Optimizer parameters
SCIPY_METHOD = 'L-BFGS-B'
SCIPY_TOL = 1e-3
SCIPY_OPT = {'disp': False, 'maxcor': 10, 'maxls': 100}

ADAM_NUM_ITER = 20
ADAM_ALPHA = 5e-5
ADAM_BETA1 = 0.1

# Which system identification algorithm to use
RUN_EM = False
RUN_ML = True

