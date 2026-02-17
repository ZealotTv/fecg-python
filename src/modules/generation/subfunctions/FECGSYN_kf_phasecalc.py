import numpy as np
from rrprocess import rrprocess


def FECGSYN_kf_phasecalc(peaks: np.ndarray, NbSamples: int) -> np.ndarray:
    phase = np.zeros(NbSamples)
    m = np.diff(peaks)
    L = peaks[0]
    if m.size == 0:
        phase = np.linspace(-2 * np.pi, 2 * np.pi, NbSamples)
    else:
        phase[0:L] = np.linspace(2 * np.pi - L * 2 * np.pi / m[0], 2 * np.pi, L)
        for i in range(0, len(peaks) - 1):
            phase[peaks[i] : peaks[i + 1]] = np.linspace(0, 2 * np.pi, m[i])
        L = len(phase) - peaks[-1]
        phase[peaks[-1] : :] = np.linspace(0, L * 2 * np.pi / m[-1], L)
    phase = np.mod(phase, 2 * np.pi)
    phase[phase > np.pi] = phase[phase > np.pi] - 2 * np.pi
    return phase


RRtemp = rrprocess()
NB_SUB = len(RRtemp)
rr = 60 / 90
fs = 1000
n = 60000
RR = np.matlib.repmat(rr, NB_SUB, 1)
csum = np.cumsum(RR)
mask = np.round(csum * fs) < n
csum = csum[mask]

print(FECGSYN_kf_phasecalc(np.round(csum * fs).astype(int), n))
