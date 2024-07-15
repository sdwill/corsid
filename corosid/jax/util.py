"""
Utility functions that use ordinary numpy, i.e., are NOT intended to be used in a differentiable forward model.
"""
from typing import Dict, AnyStr, Callable

import numpy as np
from numpy.typing import NDArray


def get_splits(variables: Dict[AnyStr, NDArray]) -> NDArray[int]:
    """
    Given a dictionary of variables, compute the indices that would be used to split the vector
    formed by concatenating the raveled variable values. See pack().
    """
    lengths = []
    for var in variables.values():
        split = np.size(var)

        # Complex-valued variables have twice as many degrees of freedom (real + imag)
        if np.iscomplexobj(var):
            split *= 2
        lengths.append(split)
    splits = np.cumsum(np.array(lengths))
    return splits


def pack(variables: Dict[AnyStr, NDArray]) -> NDArray:
    """
    Given a dictionary of variables, concatenate the raveled values into a single vector.
    """
    x0 = []
    for key in variables:
        try:
            val = variables[key].ravel()
        except AttributeError:
            val = variables[key]  # Starting guess is scalar, so no ravel() method
        x0.append(val.real)
        if np.iscomplexobj(val):
            x0.append(val.imag)

    return np.hstack(x0)  # Create a single vector-valued numpy array


def make_unpacker(variables: Dict[AnyStr, NDArray]) -> Callable:
    """
    For a given set of variables, generate a function unpack() that splits the vector into its
    constituent pieces and returns them as a dictionary with the same keys as variables.
    """
    splits = get_splits(variables)

    def unpack(x: NDArray) -> Dict[AnyStr, NDArray]:
        ret = {key: val for key, val in zip(variables.keys(), np.split(x, splits))}
        for key in ret:
            # If complex, unpack the real/imag parts and combine them into a single complex number
            if np.iscomplexobj(variables[key]):
                real, imag = np.split(ret[key], 2)
                ret[key] = real + 1j * imag

            # Check for scalar parameters
            if np.size(ret[key]) == 1:
                ret[key] = ret[key].item()

            try:
                ret[key] = np.reshape(ret[key], variables[key].shape)
            except AttributeError:  # variables[key] is scalar, so has no shape attribute
                pass  # Scalar value, so don't actually have to do anything

        return ret

    return unpack
