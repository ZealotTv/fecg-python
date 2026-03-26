from dataclasses import dataclass

import numpy as np


@dataclass
class HDmodelParameters:
    H: np.ndarray
    VCG: np.ndarray
    theta: np.ndarray
    traj: np.ndarray
    stm: np.ndarray
    rax: float = 0.2
    ray: float = 0.16
    raz: float = 0.14
    rht: float = 0.05
    ntype: int = 1
