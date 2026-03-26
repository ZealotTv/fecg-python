from dataclasses import dataclass, field

import numpy as np


@dataclass
class SimulationParameters:
    mheart: np.ndarray = field(
        default_factory=lambda: np.array([2 * np.pi / 3, 0.2, 0.4])
    )
    fheart: np.ndarray = field(
        default_factory=lambda: np.array([[-np.pi / 10, 0.35, -0.3]])
    )
    elpos: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [-np.pi / 10, 0.35, -0.3],
                [-np.pi / 10, 0.35, -0.3],
                [-np.pi / 10, 0.35, -0.3],
            ]
        )
    )
    refpose: np.ndarray = field(default_factory=lambda: np.array([np.pi, 0.5, -0.3]))
    NB_FOETUSES: int = field(init=False)
    n: int = 60000
    fs: int = 1000
    ntype: np.ndarray = field(default_factory=lambda: np.array(["MA"]))
    noise_fct: np.ndarray = field(init=False)
    SNRfm: int = -9
    SNRmn: int = 10
    mhr: int = 90
    fhr: np.ndarray = field(init=False)
    macc: int = 0
    facc: np.ndarray = field(init=False)
    mtypeacc: str = "nsr"
    maccmean: int = 0
    maccstd: int = 1
    ftypeacc: np.ndarray = field(init=False)
    faccmean: np.ndarray = field(init=False)
    faccstd: np.ndarray = field(init=False)
    ftraj: np.ndarray = field(init=False)
    mtraj: str = "none"
    fname: str = "aecg"
    mres: int = 0
    fres: np.ndarray = field(init=False)
    mvcg: np.ndarray = field(default_factory=lambda: np.random.randint(0, 8))
    fvcg: np.ndarray = field(init=False)
    evcg: np.ndarray = field(default_factory=lambda: np.random.randint(0, 3))
    posdev: int = 1
    mectb: int = 0
    fectb: int = 0

    def __post_init__(self):
        self.NB_FOETUSES = np.size(self.fheart, 0)
        self.noise_fct = np.tile(1, len(self.ntype))
        self.fhr = np.tile(150, self.NB_FOETUSES)
        self.facc = np.zeros(self.NB_FOETUSES)
        self.ftypeacc = np.tile("nsr", self.NB_FOETUSES)
        self.faccmean = np.tile(0, self.NB_FOETUSES)
        self.faccstd = np.tile(1, self.NB_FOETUSES)
        self.ftraj = np.tile("none", self.NB_FOETUSES)
        self.fres = np.zeros(self.NB_FOETUSES)
        self.fvcg = np.random.randint(0, 8, self.NB_FOETUSES)
