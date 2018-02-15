import numpy as np


def expand_shapes_to_array(x, first_dim=True):
    if not isinstance(x, np.ndarray):
        x_array = np.array(x)
    else:
        x_array = x
    is_expanded = False
    if len(x_array.shape) == 1:
        if first_dim:
            x_array = np.array([x])
        else:
            x_array = np.array([[x_c] for x_c in x])
        is_expanded = True
    return x_array, is_expanded
