import pickle

import numpy as np
from resampy import resample
from scipy.signal import butter, filtfilt, zpk2tf
from sklearn.decomposition import PCA

from .aryule import aryule
from .datatypes import DmodelParameters

data_path = "src/modules/generation/subfunctions/data/noise_sources/"


def add_noisedipole(
    N: int = 60000,
    fs: int = 1000,
    ntype: str = "MA",
    epos: np.ndarray = np.array([[1, 2, 3]]),
    noisepos: np.ndarray = np.array([[4, 5, 6]]),
) -> DmodelParameters:
    AR_ORDER = 12
    FS_NSTDB = 360
    NP_NSTDB = 20 * FS_NSTDB
    LG_NSTDB = FS_NSTDB * 29 * 60 - NP_NSTDB
    NB_EL = np.size(epos, 0)
    start = round(LG_NSTDB * np.random.rand()) - 1
    stop = start + NP_NSTDB
    with open(f"{data_path}/{ntype}.pkl", "rb") as f:
        noise = pickle.load(f)[start:stop]
    if ntype in ["MA", "EM"]:
        B, A = butter(5, 1 * 2 / FS_NSTDB, "highpass")
        noise[:, 0] = filtfilt(B, A, noise[:, 0])
        noise[:, 1] = filtfilt(B, A, noise[:, 1])
        noiser = np.zeros((int(len(noise) * fs / FS_NSTDB), 2))
        noiser[:, 0] = resample(noise[:, 0], FS_NSTDB, fs)
        noiser[:, 1] = resample(noise[:, 1], FS_NSTDB, fs)
        noise = noiser
    x = np.random.randn(N + AR_ORDER, 2)
    a = np.zeros((AR_ORDER, N + AR_ORDER))
    noise_ar = np.zeros((N, 2))
    y = np.zeros(N + AR_ORDER)
    st = -0.001
    ed = 0.001
    for cc in range(0, 2):
        atemp = aryule(noise[:, cc], AR_ORDER)
        a[:, 0] = atemp[1:]
        rinit = np.roots(atemp)
        for ev in range(1, N + AR_ORDER):
            r = np.roots(atemp)
            sImg = np.imag(r)
            sRea = np.real(r)
            rdNb = st + (ed - st) * np.random.rand(AR_ORDER, 2)
            dz = np.diag(sRea) @ rdNb[:, 0] + np.diag(sImg) @ rdNb[:, 1] * 1j
            pn = r + dz
            ind = np.abs(rinit - pn) > 0.05
            pn[ind] = r[ind]
            indlim = (sImg**2 + sRea**2) >= 0.99
            pn[indlim] = r[indlim]
            _, atemp = zpk2tf([], pn, 1)
            a[:, ev] = atemp[1:]
            if ev >= AR_ORDER:
                y[ev] = x[ev, cc] - np.dot(
                    a[:, ev], y[np.arange(ev - 1, ev - AR_ORDER - 1, -1)]
                )
        noise_ar[:, cc] = y[AR_ORDER:]
    pca = PCA()
    pc = pca.fit_transform(noise_ar)
    noise_ar = np.column_stack((noise_ar, pc[:, 0] / np.std(pc[:, 0])))

    diff = epos - np.tile(noisepos, (NB_EL, 1))
    den_norm = np.diag(1 / np.sqrt(np.sum(diff**2, axis=1)) ** 3)
    H = den_norm @ diff
    dmodel = DmodelParameters(H=H, VCG=noise_ar, ntype=3)
    return dmodel
