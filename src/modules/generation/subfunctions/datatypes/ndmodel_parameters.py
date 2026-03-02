from dataclasses import dataclass, field

import numpy as np


@dataclass
class NDmodelParameters:
    H: np.ndarray
    VCG: np.ndarray
    ntype: int = 3
    SNRfct: np.ndarray = field(default_factory=lambda: np.linspace(-np.pi, np.pi, 250))
