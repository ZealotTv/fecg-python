from dataclasses import dataclass, asdict
from typing import Any
import numpy as np

from .hdmodel_parameters import HDmodelParameters
from .simulation_parameters import SimulationParameters


@dataclass
class GeneratorOut:
    mixture: np.ndarray[Any, np.dtype[np.float32]]
    mecg: np.ndarray[Any, np.dtype[np.float32]]
    fecg: np.ndarray[Any, np.dtype[np.float32]]
    noise: np.ndarray[Any, np.dtype[np.float32]]
    m_model: HDmodelParameters
    f_model: list
    mqrs: np.ndarray
    fqrs: np.ndarray
    params: SimulationParameters

    def to_dict(self):
        return asdict(self)
