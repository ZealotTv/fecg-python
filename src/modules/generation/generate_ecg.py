import numpy as np
from subfunctions import (
    GaussParameters,
    HRVParameters,
    SimulationParameters,
    GeneratorOut,
    add_cardiacdipole,
    generate_ecg_mixture,
    build_gauss_parameters,
    cart2pol,
    ecg_model,
    generate_hrv,
    phase2qrs,
    pol2cart,
    sph2cart,
    traject_generator,
    add_noisedipole,
)


def generate_ecg(params: SimulationParameters) -> GeneratorOut:
    elpos = np.vstack([params.elpos, params.refpose])
    gp_m = {"norm": build_gauss_parameters("normal", params.mvcg)}
    if params.mectb:
        gp_m["ectopic"] = build_gauss_parameters("ectopic")
        for axis in gp_m["ectopic"]:
            for param in gp_m["ectopic"][axis]:
                VCGect = ecg_model(GaussParameters(**gp_m["ectopic"][axis]))
                VCGnorm = ecg_model(GaussParameters(**gp_m["norm"][axis]))
                gp_m["ectopic"][axis][param] *= np.max(np.abs(VCGnorm)) / np.max(
                    np.abs(VCGect)
                )

    r_m = 0.05
    L_m = np.eye(3)
    R_m = np.array([0, 0, 0])
    mh_cart = params.mheart
    mh_cart[0], mh_cart[1] = pol2cart(params.mheart[1], params.mheart[0])
    if params.posdev:
        xp, yp, zp = sph2cart(
            2 * np.pi * np.random.rand() - 1,
            np.arcsin(2 * np.random.rand() - 1),
            r_m * np.random.rand(),
        )
        mh_cart += np.array([xp, yp, zp])
    params.mheart[0], params.mheart[1], params.mheart[2] = cart2pol(
        mh_cart[0], mh_cart[1], mh_cart[2]
    )
    HRV = HRVParameters(
        hr=params.mhr,
        flo=params.mres,
        acc=params.macc,
        typeacc=params.mtypeacc,
        accmean=params.maccmean,
        accstd=params.maccstd,
    )
    theta_m, w_m = generate_hrv(HRV)
    Xc, Yc = pol2cart(elpos[:, 1], elpos[:, 0])
    epos = np.array([Xc, Yc, elpos[:, 2]])
    if params.mtraj == "none":
        mtraj = np.array([mh_cart])
    else:
        xl = np.linspace(0, mh_cart[0], 101)
        yl = np.linspace(0, mh_cart[1], 101)
        zl = np.linspace(0, mh_cart[2], 101)
        idx = np.random.randint(50, 101, 3)
        mh_cart2 = np.array([xl[idx[0]], yl[idx[1]], zl[idx[2]]])
        mtraj = traject_generator(params.n, mh_cart, mh_cart2, params.mtraj)

    print("Generating maternal model...")
    m_model = add_cardiacdipole(
        params.n, params.fs, gp_m, L_m, theta_m, w_m, params.mres, R_m, epos.T, mtraj
    )
    L_f = np.eye(3)
    R_fh = 0.1
    f_model = [None] * params.NB_FOETUSES

    for fet in range(params.NB_FOETUSES):
        print(f"Generating model for fetus {fet + 1}...")
        fh_cart = pol2cart(
            params.fheart[fet][0], params.fheart[fet][1], params.fheart[fet][2]
        )

        if params.posdev:
            xp, yp, zp = sph2cart(
                2 * np.pi * np.random.rand(),
                np.arcsin(2 * np.random.rand() - 1),
                R_fh * np.random.rand(),
            )
            posf_start = np.array([xp, yp, zp]) + fh_cart
        else:
            posf_start = fh_cart

        xl = np.linspace(0, posf_start[0], 101)
        yl = np.linspace(0, posf_start[1], 101)
        zl = np.linspace(0, posf_start[2], 101)
        idx = np.random.randint(50, 101, 3)
        posf_end = np.array([xl[idx[0]], yl[idx[1]], zl[idx[2]]])
        ftraj = traject_generator(params.n, posf_start, posf_end, params.ftraj[fet])
        gp_f = {"norm": build_gauss_parameters("normal", params.fvcg[fet])}
        if params.mectb:
            gp_f["ectopic"] = build_gauss_parameters("ectopic")
            for axis in gp_f["ectopic"]:
                for param in gp_f["ectopic"][axis]:
                    VCGect = ecg_model(GaussParameters(**gp_f["ectopic"][axis]))
                    VCGnorm = ecg_model(GaussParameters(**gp_f["norm"][axis]))
                    gp_f["ectopic"][axis][param] *= np.max(np.abs(VCGnorm)) / np.max(
                        np.abs(VCGect)
                    )
        if params.posdev:
            theta0_f = (2 * np.random.rand() - 1) * np.pi
            r0 = (2 * np.random.rand(3) - 1) * np.pi
            R_f = r0
        else:
            theta0_f = -np.pi / 2
            R_f = np.array([-3 * np.pi / 4, 0, -np.pi / 2])

        HRV = HRVParameters(
            hr=params.fhr[fet],
            lfhfr=0.8,
            hrstd=3,
            flo=params.fres[fet],
            acc=params.facc[fet],
            typeacc=params.ftypeacc[fet],
            accmean=params.faccmean[fet],
            accstd=params.faccstd[fet],
        )
        theta_f, w_f = generate_hrv(HRV, params.n, params.fs, theta0_f)

        f_model[fet] = add_cardiacdipole(
            params.n,
            params.fs,
            gp_f,
            L_f,
            theta_f,
            w_f,
            params.fres[fet],
            R_f,
            epos.T,
            ftraj,
        )
        f_model[fet].ntype = 2
    n_model = [None] * params.NB_FOETUSES

    for n in range(params.NB_FOETUSES):
        print(f"Generating model for noise source {n + 1} ..")

        xn, yn = pol2cart(2 * np.pi * np.random.rand(), 0.1 * np.random.rand())
        pos_noise = np.array([xn, yn, 0.1 * np.random.rand() - (0.5 * (n % 2))])

        model = add_noisedipole(
            params.n,
            params.fs,
            params.ntype[n],
            epos.T,
            pos_noise,
        )

        model.SNRfct = params.noise_fct[n]
        n_model[n] = model
    #  # =========================
    # # == QRS
    # # =========================
    mqrs = phase2qrs(m_model.theta)
    fqrs = [phase2qrs(f.theta) for f in f_model]
    # # =========================
    # # == MIXING
    # # =========================
    print("Projecting dipoles...")
    mixture, mecg, fecg, noise = generate_ecg_mixture(
        params.SNRfm, params.SNRmn, mqrs, fqrs, params.fs, m_model, *f_model, *n_model
    )

    # # =========================
    # # == GROUND REMOVAL
    # # =========================
    ground = mixture[-1, :]
    mixture = mixture[:-1, :] - ground

    ground = mecg[-1, :]
    mecg = mecg[:-1, :] - ground

    fecg = [f[:-1, :] - f[-1, :] for f in fecg] if fecg else []

    noise = [n[:-1, :] - n[-1, :] for n in noise] if noise else []

    # # =========================
    # # == OUTPUT
    # # =========================
    # out = {
    #     "mixture": mixture,
    #     "mecg": mecg,
    #     "fecg": fecg,
    #     "noise": noise,
    #     "m_model": m_model,
    #     "f_model": f_model,
    #     "mqrs": mqrs,
    #     "fqrs": fqrs,
    #     "param": param,
    #     "selvcgm": selvcgm,
    #     "selvcgf": selvcgf
    # }

    # return out
