import numpy as np


def generate_ecg_mixture(SNRfm, SNRmn, mqrs, fqrs, fs, *sources):
    NB_EL = sources[0].H.shape[0]
    NB_SIG2MIX = len(sources)
    NB_SAMPS = sources[0].VCG.shape[1]

    NB_FOETUSES = 0
    NB_NOISE = 0
    for src in sources:
        if src.type == 2:
            NB_FOETUSES += 1
        if src.type == 3:
            NB_NOISE += 1

    MHR = 60
    FHR = 120

    cpt2 = 0
    cpt3 = 0
    signalf = np.zeros((NB_FOETUSES * NB_EL, NB_SAMPS))
    signaln = []

    # == project dipoles
    for src in sources:
        # time varying H
        if src.H.ndim == 3:
            signal = np.zeros((NB_EL, NB_SAMPS))
            for t in range(NB_SAMPS):
                signal[:, t] = src.H[:, :, t] @ src.VCG[:, t]
        else:
            signal = src.H @ src.VCG

        # route by type
        if src.type == 1:  # maternal
            mecg = signal

        elif src.type == 2:  # fetal
            cpt2 += 1
            signalf[(cpt2 - 1) * NB_EL : cpt2 * NB_EL, :] = signal

        elif src.type == 3:  # noise
            cpt3 += 1
            signal = signal * src.SNRfct  # bsxfun equivalent
            signaln.append(signal)

    mixture = mecg.copy()

    mbeats = 60 * fs * len(mqrs) / mixture.shape[1]
    Pm = np.sum(mixture**2, axis=1) * (MHR / mbeats)
    powerm = np.mean(Pm)

    fecg = []

    if NB_FOETUSES > 0:
        fbeats = np.array([len(x) for x in fqrs])
        fbeats = 60 * fs * fbeats / signalf.shape[1]

        ampf = np.sum(signalf**2, axis=1).reshape(NB_EL, -1)
        ampf = ampf @ np.diag(FHR / fbeats)
        powerf = np.mean(ampf, axis=0)

        for i in range(NB_FOETUSES):
            p = np.sqrt(powerm / powerf[i]) * 10 ** (SNRfm / 20)

            fblock = p * signalf[i * NB_EL : (i + 1) * NB_EL, :]
            mixture += fblock
            fecg.append(fblock)
    noise = []

    if NB_NOISE > 0:
        noisegain = np.array([np.mean(np.sum(n**2, axis=1)) for n in signaln])

        noisegain = noisegain / np.sum(noisegain)

        sig = np.stack(signaln, axis=2)
        sigpow = np.sum(np.sum(sig, axis=2) ** 2, axis=1)
        meannoisepow = np.mean(sigpow)

        p = np.sqrt(powerm / meannoisepow) * 10(-SNRmn / 20)

        for i, nsrc in enumerate(signaln):
            nblock = noisegain[i] * (p * nsrc)
            mixture += nblock
            noise.append(nblock)

    return mixture, mecg, fecg, noise
