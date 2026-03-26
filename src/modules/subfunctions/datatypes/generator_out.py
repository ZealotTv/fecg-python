from dataclasses import dataclass, asdict

import numpy as np

from .hdmodel_parameters import HDmodelParameters
from .simulation_parameters import SimulationParameters


@dataclass
class GeneratorOut:
    mixture: np.ndarray
    mecg: np.ndarray
    fecg: np.ndarray
    noise: np.ndarray
    m_model: HDmodelParameters
    f_model: list
    mqrs: np.ndarray
    fqrs: np.ndarray
    params: SimulationParameters

    def to_dict(self):
        return asdict(self)
