from dataclasses import dataclass

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
