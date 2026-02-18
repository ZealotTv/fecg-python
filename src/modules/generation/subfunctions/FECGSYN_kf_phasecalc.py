import numpy as np


def FECGSYN_kf_phasecalc(peaks: np.ndarray, NbSamples: int) -> np.ndarray:
    phase = np.zeros(NbSamples)
    m = np.diff(peaks)
    L = peaks[0]
    if len(m) == 0:
        phase = np.linspace(-2 * np.pi, 2 * np.pi, NbSamples)
    else:
        phase[:L] = np.linspace(2 * np.pi - L * 2 * np.pi / m[0], 2 * np.pi, L)
        for i in range(len(peaks) - 1):
            phase[peaks[i] - 1 : peaks[i + 1]] = np.linspace(0, 2 * np.pi, m[i] + 1)
        L = len(phase) - peaks[-1]
        phase[peaks[-1] :] = np.linspace(0, L * 2 * np.pi / m[-1], L)
    phase = np.mod(phase, 2 * np.pi)
    phase[phase > np.pi] -= 2 * np.pi
    return phase
