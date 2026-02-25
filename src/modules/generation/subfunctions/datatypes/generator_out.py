from dataclasses import dataclass
from .simulation_parameters import SimulationParameters
import numpy as np

"""
out.mixture: generated ecg mixture [NB_EL x param.n matrix]
out.m_model: structure contaning dipole model for the foetus [struct]
    m_model.H: Dower-like matrix for dipole either 2D (time invariant) or 3D (variant case).
    m_model.VCG: VCG for dipole
    m_model.type: maternal (1) or foetal (2) dipole
out.f_model: structure contaning dipole model for the foetus [struct] ibid m_model
out.mecg:   mecg projected signal
out.fecg:   fecg projected signal
out.vols:   contains volume conductor information (electrodes and heart position)
out.mqrs:   maternal reference QRS
out.fqrs:   foetal reference QRS
out.param:  parameters used in the simulation [struct]
selvcgm:    selected maternal vcg [cell]
selvcgf:    selected foetal vcg [cell]
"""


@dataclass
class GeneratorOut:
    m_model_H: np.ndarray
    m_model_VCG: np.ndarray
    m_model_type: int
    f_model_H: np.ndarray
    f_model_VCG: np.ndarray
    f_model_type: int
    mecg: np.ndarray
    fecg: np.ndarray
    param: SimulationParameters
