from dataclasses import dataclass, field

import numpy as np


@dataclass
class DmodelParameters:
    H: np.ndarray
    VCG: np.ndarray
    theta: float
    traj: np.ndarray
    stm: np.ndarray
    rax: float = 0.2
    ray: float = 0.16
    raz: float = 0.14
    rht: float = 0.05
    SNRfct: np.ndarray = field(default_factory=lambda: np.linspace(-np.pi, np.pi, 250))
    stm: np.ndarray = field(default_factory=lambda: np.linspace(-np.pi, np.pi, 250))
    ntype: int = 1
