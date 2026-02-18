from dataclasses import dataclass

import numpy as np


@dataclass
class GaussParameters:
    alpha: np.ndarray
    beta: np.ndarray
    theta: np.ndarray

    def __iter__(self):
        return iter((self.alpha, self.beta, self.theta))
