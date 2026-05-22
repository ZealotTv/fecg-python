from numba import njit, objmode, prange
import numpy as np
from scipy.signal import (
    butter,
    filtfilt,
    medfilt,
    lfilter,
    resample,
    iirnotch,
)
import pickle
from sklearn.decomposition import FastICA
import time


@njit(cache=True, fastmath=True)
def norm_corr(x, y):
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)
    corr = np.correlate(x, y, mode="full")
    corr /= len(x)
    return corr


@njit(cache=True, fastmath=True)
def maxsc(v, perc=5):
    v = np.asarray(v).ravel()
    vo = np.sort(v)

    n = len(v)
    if perc < 0:
        fi = n + perc
    else:
        fi = n - int(np.floor(n * perc / 100.0))

    fi = int(fi) - 1
    return vo[fi]


@njit(cache=True, fastmath=True)
def meansc(v, perci=5, percf=None):
    v = np.asarray(v).ravel()
    if percf is None:
        percf = perci

    if perci == 50 and percf == 50:
        return np.median(v)

    if perci + percf >= 100:
        return np.empty(1, dtype=v.dtype)[0]

    vo = np.sort(v)
    n = len(v)

    ii = 1 + int(np.floor(n * perci / 100.0))
    fi = n - int(np.floor(n * percf / 100.0))

    return np.mean(vo[ii - 1 : fi])


@njit(cache=True, fastmath=True)
def mimaxsc(v, perci=5, percf=None):
    v = np.asarray(v).ravel()
    if percf is None:
        percf = perci

    vo = np.sort(v)
    n = len(v)

    if perci < 0:
        ii = 1 - perci
    else:
        ii = 1 + int(np.floor(n * perci / 100.0))

    if percf < 0:
        fi = n + percf
    else:
        fi = n - int(np.floor(n * percf / 100.0))

    segment = vo[int(ii) - 1 : int(fi)]
    return np.min(segment), np.max(segment)


@njit(cache=True, parallel=True, fastmath=True)
def np_max_1(arr):
    result = np.empty(arr.shape[0])
    for i in prange(len(result)):
        result[i] = np.max(arr[i, :])
    return result


@njit(cache=True, fastmath=True)
def meanMaxSc(v, nel, percmi=5, percma=None):
    v = np.asarray(v).ravel()
    if percma is None:
        percma = percmi

    n = len(v)

    if nel >= n:
        return np.max(v)

    # Разбиение на блоки (векторизация вместо цикла)
    n_blocks = n // nel
    v_cut = v[: n_blocks * nel]
    blocks = v_cut.reshape(n_blocks, nel)

    maxi = np_max_1(blocks)

    if percmi < 0:
        ii = 1 - percmi
    else:
        ii = 1 + np.int64(np.floor(len(maxi) * percmi / 100.0))

    if percma < 0:
        fi = len(maxi) + percma
    else:
        fi = len(maxi) - np.int64(np.floor(len(maxi) * percma / 100.0))

    omaxi = np.sort(maxi)
    return np.mean(omaxi[np.int64(ii) - 1 : np.int64(fi)])


def meanMiMaSc(v, nel, percmi=5, percma=None):
    v = np.asarray(v).ravel()
    if percma is None:
        percma = percmi

    n = len(v)
    if nel > n:
        nel = n

    # Векторизация блоков
    n_blocks = n // nel
    v_cut = v[: n_blocks * nel]
    blocks = v_cut.reshape(n_blocks, nel)

    mini = np.min(blocks, axis=1)
    maxi = np.max(blocks, axis=1)

    if percmi < 0:
        ii = 1 - percmi
    else:
        ii = 1 + int(np.floor(len(maxi) * percmi / 100.0))

    if percma < 0:
        fi = len(maxi) + percma
    else:
        fi = len(maxi) - int(np.floor(len(maxi) * percma / 100.0))

    omini = np.sort(mini)
    omaxi = np.sort(maxi)

    meaMi = np.mean(omini[int(ii) - 1 : int(fi)])
    meaMa = np.mean(omaxi[int(ii) - 1 : int(fi)])

    return meaMi, meaMa


@njit(cache=True, fastmath=True)
def mimaxscG(v, perci=5, percf=None, pmarg=0.1):
    v = np.asarray(v).ravel()
    if percf is None:
        percf = perci

    vo = np.sort(v)
    n = len(vo)

    if perci < 0:
        ii = 1 - perci
    else:
        ii = 1 + int(np.floor(n * perci / 100.0))

    if percf < 0:
        fi = n + percf
    else:
        fi = n - int(np.floor(n * percf / 100.0))

    segment = vo[int(ii) - 1 : int(fi)]

    mi = np.min(segment)
    ma = np.max(segment)

    marg = pmarg * (ma - mi)
    return np.array([mi - marg, ma + marg])


@njit(cache=True, fastmath=True)
def meanMaxScW(v, wl, wm, stepwl=1, percmi=5, percma=None):
    if percma is None:
        percma = percmi

    if percmi < 0 or percma < 0 or (percmi + percma) >= 100:
        raise ValueError('"meanMaxScW": error in input parameters')

    if stepwl < 1:
        stepwl = 1

    v = np.asarray(v)
    if v.size == 0:
        return np.array([])

    # Приведение к столбцу как в MATLAB
    if v.ndim > 1:
        nt, dummy = v.shape
        if nt < dummy:
            v = v.T
            nt = dummy
    else:
        v = v.ravel()
        nt = len(v)

    wlm = int(np.trunc(wl / wm))

    ii = 1 + int(np.floor(wlm * percmi / 100.0))
    fi = wlm - int(np.floor(wlm * percma / 100.0))

    wl = wlm * wm
    nwl = int(np.trunc((nt - wl) / stepwl)) + 1

    vme = np.zeros(nwl)

    iwl = 0  # Python index

    for is_ in range(nwl):
        segment = v[iwl : iwl + wl]

        # reshape как (wm, wlm) в MATLAB (column-major!)
        seg_reshaped = np.reshape(segment, (wm, wlm), order="F")

        vmawm = np.max(seg_reshaped, axis=0)
        omawl = np.sort(vmawm)

        vme[is_] = np.mean(omawl[ii - 1 : fi])

        iwl += stepwl

    return vme


def FecgDetrFilt(X, fs):
    X = np.asarray(X)
    if len(X.shape) != 2:
        ns = 1
    else:
        ns = X.shape[1]
    fmaxd = 5
    fmaxn = fmaxd / (fs / 2)

    b, a = butter(6, fmaxn, btype="low")
    Xb = filtfilt(b, a, X, axis=0)

    Xd = X - Xb
    RRpmean = 1.0
    ww = int(np.trunc(RRpmean * 1.5 * fs))

    for is_ in range(ns):
        x = Xd[:, is_]
        xmeaMi, xmeaMa = meanMiMaSc(x, ww, 8, 8)

        thmima = 1.3
        xthmi = xmeaMi * thmima
        xthma = xmeaMa * thmima

        iibadDetr = np.where((x < xthmi) | (x > xthma))[0]

        if len(iibadDetr) > 1.3 * fs:
            wm = int(np.trunc(0.26 * fs))
            wm = 2 * (wm // 2) + 1  # нечётное окно

            # аналог medfilt1mit → используем medfilt
            Xb[:, is_] = medfilt(X[:, is_], kernel_size=wm)

            x = X[:, is_] - Xb[:, is_]

            xmeaMi, xmeaMa = meanMiMaSc(x, ww, 8, 8)

            xthmi = xmeaMi * thmima
            xthma = xmeaMa * thmima

            x[x < xthmi] = xthmi
            x[x > xthma] = xthma

            fmaxd = 130
            fmaxn = fmaxd / (fs / 2)

            b, a = butter(6, fmaxn, btype="low")
            x = filtfilt(b, a, x)

            Xd[:, is_] = x

    return Xd


def medfilt1mit(x, m=3, nit=1):
    x = np.asarray(x)

    # Проверка ориентации (MATLAB-логика)
    if x.ndim > 1 and x.shape[1] > x.shape[0]:
        x = x.T
        colV = False
    else:
        x = x.reshape(-1, 1) if x.ndim > 1 else x.ravel()
        colV = True

    n = len(x)
    m2 = int(np.floor(m / 2))

    xi = np.median(x[: min(n, m)])
    xf = np.median(x[max(0, n - min(n, m)) :])

    # паддинг как в MATLAB
    xt = np.concatenate([np.full(m2, xi), x, np.full(m2, xf)])

    for _ in range(nit):
        xx = xt.copy()
        xt = medfilt(xx, kernel_size=m)

        # MATLAB: all(~(xt-xx)) → проверка равенства
        if np.allclose(xt, xx):
            break

    xmf = xt[m2:-m2] if m2 > 0 else xt

    if colV:
        return xmf
    else:
        return xmf.reshape(1, -1)


@njit(cache=True, fastmath=True)
def QRSdetectorM(vadx, vdx, Fs, pth=0.5, RRts=0.87, pmQT=1):
    vadx = np.asarray(vadx).ravel()
    vdx = np.asarray(vdx).ravel()
    sqRRts = np.sqrt(RRts)
    QTlen = 0.420 * sqRRts
    QRSd = int(np.round((0.05 + 0.25 * sqRRts) * Fs))
    maskQT = int(np.round((0.07 + pmQT * QTlen) * Fs))
    rthT = 2.4
    nsp = int(np.round(0.15 * sqRRts * Fs))
    nsd = int(np.round(0.078 * sqRRts * Fs))
    RRtc = Fs * RRts
    QRSref = []
    QRSonset = []
    inizio = []
    vimaxd = []
    fine = []
    QRSoffset = []
    isai = int(1.5 * Fs)
    fsai = min(len(vadx), int(np.floor(60 * RRtc)))
    w2 = int(np.trunc(2 * RRtc))
    mD2 = meanMaxSc(vadx[isai:fsai], w2, 1, 1)
    meaD = mD2
    th = pth * meaD
    thbon = 0.5 * th
    thboff = 0.5 * th
    minD, maxD = mimaxsc(vdx[isai:fsai], 1, 1)
    vsdx = vdx if (maxD > -minD * 1.1) else -vdx
    jq = 0
    QRS = 0
    maxd = 0
    imaxd = 0
    RRcm = RRtc
    iinizio = -maskQT
    ifine = iinizio
    for i in prange(len(vadx)):
        vadxi = vadx[i]
        if QRS:
            if vadxi < th:
                inizio.append(iinizio)
                vimaxd.append(imaxd)
                ifine = i
                fine.append(i)
                maxdc = min(maxd, th * 4)
                thon = min((0.5 * thbon + 0.25 * maxdc), th)
                thoff = min((0.5 * thboff + 0.25 * maxdc), th)
                start = max(iinizio - nsp, 0)
                end = max(iinizio, 0)
                segment = vadx[start:end]
                idx = np.where(segment > thon)[0]
                QRSonset.append(start + (idx[0] if len(idx) else 0))
                start = max(iinizio, 0)
                end = min(ifine, len(vsdx))
                seg = vsdx[start:end]
                imaxs = np.argmax(seg)
                QRSref.append(start + imaxs)
                start = ifine - 1
                end = min(ifine + nsd, len(vadx))
                seg = vadx[start:end]
                idx = np.where(seg > thoff)[0]
                QRSoffset.append(start + (idx[-1] if len(idx) else 0))
                meaD = 0.97 * meaD + 0.03 * maxdc
                th = pth * meaD
                thbon = 0.5 * th
                thboff = 0.5 * th
                if jq > 0:
                    RRcj = QRSref[jq] - QRSref[jq - 1]
                    RRcm = RRcm + 0.97 * np.sign(RRcj - RRcm) * min(
                        abs(RRcj - RRcm), 0.1 * RRcm
                    )
                    RRsm = RRcm / Fs
                    if RRsm < RRts * 0.4 or RRsm > RRts * 2.5:
                        RRsm = RRts
                        RRcm = RRsm * Fs
                    sqRRsm = np.sqrt(RRsm)
                    QTlen = 0.420 * sqRRsm
                    maskQT = int(np.round((0.07 + pmQT * QTlen) * Fs))
                    nsp = int(np.round(0.15 * sqRRsm * Fs))
                    nsd = int(np.round(0.078 * sqRRsm * Fs))
                QRS = 0
                jq += 1
            elif vadxi > maxd:
                imaxd = i
                maxd = vadxi
        else:
            if vadxi > th:
                if i < iinizio + QRSd or i < ifine + nsd:
                    if jq > 0:
                        jq -= 1
                    ifine = i
                    if vadxi > maxd:
                        imaxd = i
                        maxd = vadxi
                    QRS = 1
                elif vadxi > th * (rthT - (rthT - 1) * (i - ifine) / maskQT):
                    imaxd = i
                    iinizio = i
                    maxd = vadxi
                    QRS = 1
    if jq > 0:
        qrsM = np.vstack(
            (
                np.array(QRSref[:jq]),
                np.array(QRSonset[:jq]),
                np.array(inizio[:jq]),
                np.array(vimaxd[:jq]),
                np.array(fine[:jq]),
                np.array(QRSoffset[:jq]),
            )
        )
    else:
        qrsM = np.empty((6, 0), dtype=np.int64)
    return qrsM


@njit(cache=True, fastmath=True)
def FecgQRSmDet(Se, fs):
    Se = np.asarray(Se)
    ndt, ns = Se.shape

    # нормализация каналов
    X = np.zeros_like(Se, dtype=np.float64)
    for is_ in range(ns):
        X[:, is_] = (Se[:, is_] - np.mean(Se[:, is_])) / np.std(Se[:, is_])
    ecg = X
    ecgf = ecg.copy()

    # производный фильтр
    nu = int(np.ceil(0.0070 * fs))
    nz = int(np.floor(0.0090 * fs / 2) * 2 + 1)

    B = np.concatenate((np.ones(nu), np.zeros(nz), -np.ones(nu)))

    delay = int(np.floor(len(B) / 2))

    ecgf0 = ecgf[0, :].repeat(delay).reshape((-1, delay)).T
    ecgf1 = ecgf[-1, :].repeat(delay).reshape((-1, delay)).T

    ecgfx = np.vstack((ecgf0, ecgf, ecgf1))
    # ecgfx = np.empty(ecgf0.size + ecgf.size + ecgf1.size)
    # ecgfx[: ecgf0.size] = ecgf0
    # ecgfx[ecgf0.size : ecgf0.size + ecgf.size] = ecgf
    # ecgfx[ecgf0.size + ecgf.size :] = ecgf1

    with objmode(decgr="float64[:,:]"):
        decgr = lfilter(B, [1], ecgfx, axis=0)
        decgr = decgr[2 * delay :, :]
    adecg = np.abs(decgr)
    # окна
    w8 = int(np.trunc(8 * fs))
    w2 = int(np.trunc(2 * fs))
    w02 = int(np.trunc(0.2 * fs))
    mD8 = np.zeros(ns)
    mD2 = np.zeros(ns)
    mD02 = np.zeros(ns)
    for is_ in range(ns):
        mD8[is_] = meanMaxSc(adecg[:, is_], w8, 0, 5)
        mD2[is_] = meanMaxSc(adecg[:, is_], w2, 0, 5)
        mD02[is_] = meanMaxSc(adecg[:, is_], w02, 0, 1)
    qualFact = mD2 / (mD02 + mD8)

    ics = np.argsort(-qualFact)

    decgs = decgr[:, ics]
    adecgs = adecg[:, ics]

    decg1 = decgs[:, 0]
    adecg1 = adecgs[:, 0]

    # полосовой фильтр
    fmind = 5
    fmaxd = 20
    Wn = (fmind / (fs / 2), fmaxd / (fs / 2))

    with objmode(adecg1="float64[:]"):
        b, a = butter(1, Wn, btype="band")
        adecg1 = filtfilt(b, a, adecg1)

    npx = int(np.trunc(1 * fs))

    decg1x = np.concatenate((np.zeros(npx), decg1, np.zeros(npx)))
    adecg1x = np.concatenate((np.zeros(npx), adecg1, np.zeros(npx)))

    pth = 0.45
    qrsM = QRSdetectorM(adecg1x, decg1x, fs, pth, 0.85, 1)
    qrsM = qrsM - npx

    return qrsM


@njit(cache=True, fastmath=True)
def weightFun2(npp, npd, fs):
    nppc1 = int(np.trunc(0.06 * fs))
    npdc1 = int(np.trunc(0.06 * fs))
    nppc2 = int(np.trunc(0.08 * fs))
    npdc2 = min(int(np.trunc(0.2 * fs)), npd - npdc1)

    ii1 = 0
    ie1 = npp - nppc1 - nppc2

    ii2 = ie1
    ie2 = ie1 + nppc2

    ii3 = ie2
    ie3 = ie2 + nppc1 + npdc1 + 1

    ii4 = ie3
    ie4 = ie3 + npdc2

    ii5 = ie4
    ie5 = npp + npd + 1

    wwg = np.zeros(npp + npd + 1)

    # MATLAB → Python (учёт полуинтервалов)
    wwg[ii1:ie1] = 0.20
    wwg[ii2:ie2] = 0.20 + 0.8 * (np.arange(1, nppc2 + 1) / nppc2)
    wwg[ii3:ie3] = 1.0
    wwg[ii4:ie4] = 1.0 - 0.8 * (np.arange(1, npdc2 + 1) / npdc2)
    wwg[ii5:ie5] = 0.20

    return wwg.T


@njit(cache=True)
def my_tile(arr, reps):
    if reps[0] == 0 or reps[1] == 0:
        return np.empty(
            (reps[0], arr.size if reps[1] == 0 else arr.size), dtype=arr.dtype
        )
    else:
        return arr.repeat(reps[0]).reshape((-1, reps[0])).T


@njit(cache=True, fastmath=True)
def FecgQRSmCanc(X, qrsM, fs):
    X = np.asarray(X)
    ndt, ns = X.shape
    qrsR = qrsM[0, :]
    nQRS = len(qrsR)
    RRc = np.diff(qrsR)
    RRs = RRc / fs
    RRmean = meansc(RRs, 4, 4)
    npp = int(np.trunc(0.2 * fs))
    npd = int(np.trunc(min(0.5, 0.8 * (RRmean - 0.1)) * fs))
    npt = 1 + npp + npd

    Xx = X.copy()

    npqp = int(np.trunc(0.12 * fs))

    if qrsR[0] - npqp < 1:
        qi = 1
        npxp = 0
        Xx[:npqp, :] = Xx[npqp, :]
    else:
        qi = 0
        npxp = max(0, npp + 1 - qrsR[0])
        Xx = np.vstack((my_tile(X[0, :], (npxp, 1)), X))
    npqd = int(np.trunc(0.14 * fs))
    qf = nQRS

    if qrsR[-1] + npqd > X.shape[0]:
        qf -= 1
        Xx[-npqd:, :] = Xx[-npqd - 1, :]

    if qrsR[qf - 1] + 0.85 * fs * np.median(RRs[-4:]) < X.shape[0]:
        npep = int(np.trunc(max(0.1 * fs, 0.15 * fs * np.mean(RRs[-4:]))))
        Xx[-npep:, :] = Xx[-npep - 1, :]

    npxd = max(0, int(qrsR[qf - 1] + npd - ndt))
    Xx = np.vstack((Xx, my_tile(X[-1, :], (npxd, 1))))

    ndtx = Xx.shape[0]
    nqe = qf - qi - 1

    iqw = qrsR[qi:qf]

    iw = npxp + iqw - npp
    fw = npxp + iqw + npd
    A = np.zeros((npt, nqe))
    wwg = weightFun2(npp, npd, fs)

    Xc = np.zeros_like(Xx)
    for is_ in range(ns):
        for iq in range(nqe):
            iwq = int(iw[iq])
            fwq = int(fw[iq])
            A[:, iq] = Xx[iwq : fwq + 1, is_] * wwg

        # SVD
        U, S, Vt = np.linalg.svd(A, full_matrices=False)

        mt = nqe
        nds = 3
        # подавление компонент
        Sr = np.diag(S)
        for k in range(mt - nds - 1):
            Sr[mt - k - 1, mt - k - 1] = 0

        Ar = U @ Sr @ Vt

        # обратная вставка
        for iq in range(nqe):
            iwq = int(iw[iq])
            fwq = int(fw[iq])
            Xc[iwq : fwq + 1, is_] = Ar[:, iq] / wwg

        # линейная интерполяция разрывов
        for iq in range(nqe - 1):
            fwq = int(fw[iq])
            iwqs = int(iw[iq + 1])

            if iwqs > fwq:
                dv = Xc[iwqs, is_] - Xc[fwq, is_]
                pv = dv / (iwqs - fwq)

                for k in range(1, iwqs - fwq):
                    Xc[fwq + k, is_] = Xc[fwq, is_] + pv * k

    # удаление паддинга
    Xc = Xc[npxp : ndtx - npxd, :]
    Xx = Xx[npxp : ndtx - npxd, :]

    Xe = Xx - Xc

    return Xe


def filtNotchFB(x, fnotch, fs, Q=30):
    b, a = iirnotch(fnotch, Q, fs)
    return filtfilt(b, a, x)


def FecgNotchFilt(X, fs):
    from scipy.signal import welch

    X = np.asarray(X)
    Xf = np.zeros_like(X)

    for is_ in range(X.shape[1]):
        x = X[:, is_]
        Px, Fv = welch(x, fs=fs)

        # детекция пика
        def detect_peak(fmin, fmax):
            idx = np.where((Fv > fmin) & (Fv < fmax))[0]
            if len(idx) == 0:
                return False, 0, 0

            Pxw = Px[idx]
            Fw = Fv[idx]

            imax = np.argmax(Pxw)
            maxP = Pxw[imax]
            fpeak = Fw[imax]

            idx_bg = np.where(
                ((Fv > fmin - 3) & (Fv < fmin)) | ((Fv > fmax) & (Fv < fmax + 3))
            )[0]

            if len(idx_bg) == 0:
                return False, 0, 0

            mean_bg = np.mean(Px[idx_bg])
            std_bg = np.std(Px[idx_bg])

            return (maxP - mean_bg) > 5 * std_bg, (maxP - mean_bg), fpeak

        e50, d50, f50 = detect_peak(49, 51)
        e60, d60, f60 = detect_peak(59, 61)

        if e50 or e60:
            fnotch = f50 if d50 > d60 else f60

            xf = x.copy()
            for k in range(1, 5):
                xf = filtNotchFB(xf, k * fnotch, fs)

            Xf[:, is_] = xf
        else:
            Xf[:, is_] = x

    return Xf


def ImpArtElimS(x, thE=4, wm=None, pvsc=10):
    x = np.asarray(x).ravel()
    n = len(x)

    if wm is None:
        wm = min(33, n - 1)

    wm = 2 * (wm // 2) + 1  # нечётное окно

    xc = x.copy()

    xmed = medfilt1mit(xc, wm, 1)
    xad = np.abs(xc - xmed)

    xad_pos = xad[xad > 0]
    if len(xad_pos) == 0:
        return xc

    return xc


def FecgImpArtCanc(X, fs):
    X = np.asarray(X).copy()
    ndt, ns = X.shape

    npti = 10
    nptim = 3

    # начальная коррекция
    for is_ in range(ns):
        X[:npti, is_] = np.median(X[npti : npti + nptim, is_])

    thE = 4
    wm = int(np.trunc(0.06 * fs))
    pvsc = 2

    Xc = np.zeros_like(X)

    for is_ in range(ns):
        Xc[:, is_] = ImpArtElimS(X[:, is_], thE, wm, pvsc)

    return Xc


def pipeline(fs, signal_mixture):
    interp_factor = 4
    fs_new = fs * interp_factor
    filtered = FecgImpArtCanc(signal_mixture.T, fs)
    filtered = FecgNotchFilt(filtered, fs)
    filtered = FecgDetrFilt(filtered, fs)
    ICA = FastICA(3)
    component = ICA.fit_transform(filtered)
    comp_int = resample(component, np.round(len(component) * interp_factor))
    mqrs = FecgQRSmDet(comp_int, fs_new)
    x_reduced = FecgQRSmCanc(comp_int, mqrs, fs_new)
    x_reduced = ICA.inverse_transform(x_reduced)
    x_reduced = resample(x_reduced, len(component))
    return x_reduced


with open(
    f"/home/efremov/work/FECG/help/PEREPISIVANIE TOCHNO/gen/temp_with_noise_MA_SNR=12.pkl",
    "rb",
) as f:
    temp_file = pickle.load(f)
    a = pipeline(temp_file["params"]["fs"], temp_file["mixture"][:3])

with open(
    f"/home/efremov/work/FECG/help/PEREPISIVANIE TOCHNO/gen/temp_twins_with_noise_EM_SNR=6.pkl",
    "rb",
) as f:
    temp_file = pickle.load(f)
    start_time = time.perf_counter()
    a = pipeline(temp_file["params"]["fs"], temp_file["mixture"][:3])
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Overall execution time2: {execution_time:.6f} seconds")

# with open(
#     f"/home/efremov/work/FECG/help/PEREPISIVANIE TOCHNO/gen/temp_with_noise_MA_SNR=0.pkl",
#     "rb",
# ) as f:
#     temp_file = pickle.load(f)
#     start_time = time.perf_counter()
#     a = pipeline(temp_file["params"]["fs"], temp_file["mixture"][:3])
#     end_time = time.perf_counter()
#     execution_time = end_time - start_time
#     print(f"Overall execution time3: {execution_time:.6f} seconds")

# with open(
#     f"/home/efremov/work/FECG/help/PEREPISIVANIE TOCHNO/gen/temp_with_noise_EM_SNR=12.pkl",
#     "rb",
# ) as f:
#     temp_file = pickle.load(f)
#     start_time = time.perf_counter()
#     a = pipeline(temp_file["params"]["fs"], temp_file["mixture"][:3])
#     end_time = time.perf_counter()
#     execution_time = end_time - start_time
#     print(f"Overall execution time4: {execution_time:.6f} seconds")
