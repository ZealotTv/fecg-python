import numpy as np
from resampy import resample
from scipy.interpolate import interp1d


def generate_breathing(fs: int = 1000, N: int = 250, fres: int = 1000):
    deltaA = 0.3
    fa = 0.1
    deltaF = 0.05
    ff = 0.1
    n = np.arange(1, N + 1)
    a = 2 / np.pi * (1 + deltaA * np.sin(2 * np.pi * fa * n / fs))
    b = 2 / (2 * np.pi) * (1 + deltaA * np.sin(2 * np.pi * fa * n / fs))
    c = 2 / (3 * np.pi) * (1 + deltaA * np.sin(2 * np.pi * fa * n / fs))

    signal = 0.5 - (
        a
        * np.sin(
            2 * np.pi * fres * n / fs + deltaF / ff * np.sin(2 * np.pi * ff * n / fs)
        )
        + b
        * np.sin(
            4 * np.pi * fres * n / fs + deltaF / ff * np.sin(2 * np.pi * ff * n / fs)
        )
        + c
        * np.sin(
            6 * np.pi * fres * n / fs + deltaF / ff * np.sin(2 * np.pi * ff * n / fs)
        )
    )
    signal_noisy = signal + 0.1 * np.random.rand(1, N)
    x2 = np.arange(1, N + 1, 100)
    signal_noisy_interp = interp1d(np.arange(1, N + 1), signal_noisy)(x2)
    bwa = resample(signal_noisy_interp, 1, 100)

    bwa /= np.abs(np.max(bwa)) + np.abs(np.min(bwa))
    bwa += np.abs(np.min(bwa)) - 0.5
    return bwa
