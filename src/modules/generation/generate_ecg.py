import numpy as np
from subfunctions import (
    GaussParameters,
    HRVParameters,
    SimulationParameters,
    add_cardiacdipole,
    build_gauss_parameters,
    cart2pol,
    ecg_model,
    generate_hrv,
    pol2cart,
    sph2cart,
    traject_generator,
)


def generate_ecg(params: SimulationParameters):
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
        xl = np.linspace(0, mh_cart[0])
        yl = np.linspace(0, mh_cart[1])
        zl = np.linspace(0, mh_cart[2])
        idx = np.random.randint(50, 100, (1, 3))[0]
        mh_cart2 = np.array([xl[idx[0]], yl[idx[1]], zl[idx[2]]])
        mtraj = traject_generator(params.n, mh_cart, mh_cart2, params.mtraj)

    m_model = add_cardiacdipole(
        params.n, params.fs, gp_m, L_m, theta_m, w_m, params.mres, R_m, epos.T, mtraj
    )
    m_model.type = 1
    L_f = np.eye(3)
    R_f = 0.1
    # f_model = []
    # selvcgf = []

    # for fet in range(NB_FOETUSES):
    #     print(f"Generating model for fetus {fet + 1} ..")

    #     fh_cart = param.fheart[fet].copy()
    #     fh_cart[0], fh_cart[1] = pol2cart(param.fheart[fet][0], param.fheart[fet][1])

    #     if param.posdev:
    #         xp, yp, zp = sph2cart(
    #             2 * np.pi * np.random.rand(),
    #             np.arcsin(2 * np.random.rand() - 1),
    #             0.1 * np.random.rand(),
    #         )
    #         posf_start = np.array([xp, yp, zp]) + fh_cart
    #     else:
    #         posf_start = fh_cart

    #     xl = np.linspace(0, posf_start[0])
    #     yl = np.linspace(0, posf_start[1])
    #     zl = np.linspace(0, posf_start[2])
    #     idx = np.random.randint(50, 101, 3)
    #     posf_end = np.array([xl[idx[0]], yl[idx[1]], zl[idx[2]]])

    #     gp_f = SimpleNamespace()
    #     gp_f.norm, sel = load_gparam(param.fvcg[fet], "normal")
    #     selvcgf.append(sel)

    #     # HRV
    #     theta0_f = (2 * np.random.rand() - 1) * np.pi if param.posdev else -np.pi / 2
    #     strhrv.hr = param.fhr[fet]
    #     strhrv.flo = param.fres[fet]
    #     strhrv.acc = param.facc[fet]

    #     theta_f, w_f = generate_hrv(strhrv, param.n, param.fs, theta0_f)

    #     traj = traject_generator(param.n, posf_start, posf_end, param.ftraj[fet])

    #     model = add_cardiacdipole(
    #         param.n,
    #         param.fs,
    #         gp_f,
    #         np.eye(3),
    #         theta_f,
    #         w_f,
    #         param.fres[fet],
    #         SimpleNamespace(x=0, y=0, z=0),
    #         epos,
    #         traj,
    #         0,
    #     )
    #     model["type"] = 2
    #     f_model.append(model)
    # # =========================
    # # == NOISE
    # # =========================
    # n_model = []

    # for n in range(NB_NOISES):
    #     print(f"Generating model for noise source {n+1} ..")

    #     xn, yn = pol2cart(2*np.pi*np.random.rand(), 0.1*np.random.rand())
    #     pos_noise = np.array([
    #         xn, yn,
    #         0.1*np.random.rand() - (0.5*(n % 2))
    #     ])

    #     model, tmp_handle, noise_misc = add_noisedipole(
    #         param.n, param.fs,
    #         param.ntype[n],
    #         epos,
    #         pos_noise,
    #         debug
    #     )

    #     model["SNRfct"] = param.noise_fct[n]
    #     model["pos"] = pos_noise
    #     model["type"] = 3
    #     n_model.append(model)
    #  # =========================
    # # == QRS
    # # =========================
    # mqrs = phase2qrs(m_model["theta"])
    # fqrs = [phase2qrs(f["theta"]) for f in f_model]

    # # =========================
    # # == MIXING
    # # =========================
    # print("Projecting dipoles...")
    # mixture, mecg, fecg, noise, ecg_f_handle = generate_ecg_mixture(
    #     debug, param.SNRfm, param.SNRmn,
    #     mqrs, fqrs, param.fs,
    #     m_model, *f_model, *n_model
    # )

    # # =========================
    # # == GROUND REMOVAL
    # # =========================
    # ground = mixture[-1, :]
    # mixture = mixture[:-1, :] - ground

    # ground = mecg[-1, :]
    # mecg = mecg[:-1, :] - ground

    # fecg = [
    #     f[:-1, :] - f[-1, :]
    #     for f in fecg
    # ] if fecg else []

    # noise = [
    #     n[:-1, :] - n[-1, :]
    #     for n in noise
    # ] if noise else []

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
