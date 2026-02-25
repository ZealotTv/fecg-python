import numpy as np
from .datatypes import HRVParameters
from .FECGSYN_kf_phasecalc import FECGSYN_kf_phasecalc
from .rrprocess import rrprocess
from scipy.interpolate import interp1d


def generate_hrv(
    hrv_param: HRVParameters, n: int = 60000, fs: int = 1000, theta0: float = np.pi / 3
) -> np.ndarray:
    strhrv = hrv_param
    NB_SUB = np.ceil(strhrv.hr * (n / fs) / 60).astype(int)
    RRtemp = rrprocess(
        n=NB_SUB,
        hrmean=strhrv.hr,
        lfhfr=strhrv.lfhfr,
        hrstd=strhrv.hrstd,
        flo=strhrv.flo,
        fhi=strhrv.fhi,
    )
    NB_SUB = len(RRtemp)
    HRV = 60.0 / RRtemp
    match strhrv.typeacc:
        case "none":
            rr = 60 / strhrv.hr
            RR = np.full(NB_SUB, rr)
        case "nsr":
            RR = 60.0 / HRV
        case "tanh":
            tmp = np.linspace(-20, 20, NB_SUB)
            strhrv.accmean *= 20
            tmp = ((np.tanh(tmp - strhrv.accmean) + 1) / 2) * strhrv.acc
            HRV += tmp
            RR = 60.0 / HRV
        case "mexhat":
            x = np.linspace(-3, 3, NB_SUB)
            strhrv.accmean *= 3
            c = 2 / (np.sqrt(3 * strhrv.accstd) * np.pi ** (1 / 4))
            sombrero = (
                c
                * (1 - (x - strhrv.accmean) ** 2 / strhrv.accstd**2)
                * np.exp(-((x - strhrv.accmean) ** 2) / (2 * strhrv.accstd))
            )
            sombrero *= strhrv.acc / max(sombrero)
            HRV += sombrero
            RR = 60.0 / HRV
        case "gauss":
            x = np.linspace(-3, 3, NB_SUB)
            strhrv.accmean *= 3
            gauss = (
                1
                / (np.sqrt(2 * np.pi) * strhrv.accstd)
                * np.exp(-((x - strhrv.accmean) ** 2) / (2 * strhrv.accstd**2))
            )
            gauss = strhrv.acc * gauss / max(gauss)

            HRV += gauss
            RR = 60.0 / HRV
    # я не знаю как сделать без костылей, эти 6 строчек кода прокляты
    csum = np.cumsum(RR)
    RR = RR[csum <= n / fs]
    csum = np.cumsum(RR)
    mask = np.round(csum * fs) <= n
    csum = csum[mask]
    RR = RR[: len(csum)]
    RR_rs = interp1d(np.cumsum(RR), RR)(np.arange(RR[0], sum(RR), 1 / fs))
    nbm_str = np.ceil(RR[0] * fs).astype(int)
    nbm_end = np.ceil(n - sum(RR) * fs).astype(int)
    RR_rs = np.concatenate(
        [
            np.full(nbm_str, RR_rs[0]),
            RR_rs,
            np.full(nbm_end, RR_rs[-1]),
        ],
    )
    hr = 1.0 / RR_rs[:n]
    w = 2 * np.pi * hr

    theta = FECGSYN_kf_phasecalc(np.round(csum * fs).astype(int), n)
    nshift = np.where(theta > theta0)[0][0]
    theta = FECGSYN_kf_phasecalc(np.round(csum * fs).astype(int), n + nshift)
    theta = np.delete(theta, slice(0, nshift))
    return theta, w
