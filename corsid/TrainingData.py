import numpy as np

class TrainingData:
    """ Structure for the training data used for system identification """
    def __init__(
            self,
            num_pix: int = None,    # Number of pixels in dark zone
            len_x: int = None,      # Length of state vector (usually 2)
            len_z: int = None,      # Length of data vector (number of probe pairs)
            len_u: int = None,      # Length of control vector (number of DM actuators)
            num_iter: int = None,   # Number of iterations of training data (time steps)
            dzs: dict = None,       # Change in data vector in time step relative to previous
            us: dict = None,        # Change in DM command in each time step relative to previous
            Psi: np.ndarray = None  # Probe command matrix (len_u, len_z). Each probe is a column.
    ):
        self.num_pix = num_pix
        self.len_x = len_x
        self.len_z = len_z
        self.len_u = len_u
        self.num_iter = num_iter

        if dzs is None:
            dzs = {}
        self.dzs = dzs

        if us is None:
            us = {}
        self.us = us

        self.Psi = Psi
