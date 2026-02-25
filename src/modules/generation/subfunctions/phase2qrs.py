import numpy as np


def phase2qrs(phase: np.ndarray) -> np.ndarray:
    sy = np.sign(phase)
    cond1 = np.diff(sy) == 2
    cond2 = sy[:-1] == 0
    flag_cross = cond1 | cond2
    return np.where(flag_cross)[0]
