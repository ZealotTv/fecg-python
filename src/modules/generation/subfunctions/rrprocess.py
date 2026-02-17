import numpy as np


def rrprocess(
    n: int = 512,
    hrmean: int = 60,
    lfhfr: float = 0.5,
    hrstd: int = 1,
    sfrr: int = 1,
    flo: float = 0.1,
    fhi: float = 0.25,
    flostd: float = 0.01,
    fhistd: float = 0.01,
) -> float:
    w1 = 2 * np.pi * flo
    w2 = 2 * np.pi * fhi
    c1 = 2 * np.pi * flostd
    c2 = 2 * np.pi * fhistd
    sig2 = 1
    sig1 = lfhfr
    rrmean = 60 / hrmean
    rrstd = 60 * hrstd / (hrmean * hrmean)

    df = sfrr / n
    w = np.arange(0, n) * 2 * np.pi * df
    dw1 = w - w1
    dw2 = w - w2

    Hw1 = sig1 * np.exp(-0.5 * (dw1 / c1) ** 2) / np.sqrt(2 * np.pi * c1**2)
    Hw2 = sig2 * np.exp(-0.5 * (dw2 / c2) ** 2) / np.sqrt(2 * np.pi * c2**2)
    Hw = Hw1 + Hw2
    Hw0 = np.concatenate(
        (Hw[0 : np.floor(n / 2).astype(int) - 1], Hw[np.floor(n / 2).astype(int) :: -1])
    )
    Sw = (sfrr / 2) * np.sqrt(Hw0)
    print(np.floor(n / 2))

    ph0 = 2 * np.pi * np.ones(np.floor(n / 2).astype(int) - 1)
    ph = np.concatenate(([0], ph0.flatten(), [0], -np.flipud(ph0).flatten()))
    SwC = Sw * np.exp(1j * ph)
    x = (1 / n) * np.real(np.fft.ifft(SwC))

    xstd = np.std(x)
    ratio = rrstd / xstd
    rr = rrmean + x * ratio
    return rr


print(rrprocess())
