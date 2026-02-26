import numpy as np
from scipy.signal import butter, filtfilt

from .datatypes import DmodelParameters
from .generate_breathing import generate_breathing
from .rotate_xyz import rotate_xyz


def add_cardiacdipole(
    N: int,
    fs: int,
    gp_all: dict,
    L: np.ndarray,
    theta: np.ndarray,
    w: np.ndarray,
    fres: int,
    R0: np.ndarray,
    epos: np.ndarray,
    traj: np.ndarray,
) -> DmodelParameters:
    ect = False
    if "ectopic" in gp_all:
        ect = True
        gp_ecto = gp_all["ecto"]
        rn = 0.7 + 0.1 * np.random.rand()
        re = 0.2 + 0.1 * np.random.rand()
        STM = np.array([[rn, (1 - rn)], [(1 - re), re]])
        S = np.array([1, 0])
    else:
        STM = None

    gp_norm = gp_all["norm"]

    RESP_ANG_X = 2 * 0.1
    RESP_ANG_Y = 2 * 0.08
    RESP_ANG_Z = 2 * 0.07
    HEART_T_RESP = 0.05

    NB_EL = epos.shape[1]
    dt = 1 / fs

    VCG = np.zeros((3, N))

    ncy = np.where(np.diff(theta) < 0)[0]

    if fres == 0:
        brwave = np.zeros(N)
    else:
        brwave = generate_breathing(fs, N, fres, 0)
    if fres == 0 and traj.shape[0] == 1:
        diff = epos - traj
        den_norm = 1.0 / np.sqrt(np.sum(diff**2, axis=1)) ** 3
        H = den_norm[:, None] * diff
    else:
        H = np.zeros((NB_EL, 3, N))
    if fres == 0:
        dpos = traj
    else:
        dpos = traj + np.column_stack(
            [np.zeros(N), np.zeros(N), HEART_T_RESP * (brwave + 0.5)]
        )

    crst = X = Y = Z = 0
    for i in range(N):
        if ect and i in ncy:
            rd_nb = np.random.rand()
            crst = np.argmax(S)
            if rd_nb > STM[crst, 0]:
                S = np.array([0, 1])
            else:
                S = np.array([1, 0])

        gp = gp_ecto if (ect and np.argmax(S) == 1) else gp_norm

        def angle_diff(th, center):
            return np.mod(th - center + np.pi, 2 * np.pi) - np.pi

        dthetaix = angle_diff(theta[i], np.array(gp["x"]["theta"]))
        dthetaiy = angle_diff(theta[i], np.array(gp["y"]["theta"]))
        dthetaiz = angle_diff(theta[i], np.array(gp["z"]["theta"]))

        X -= dt * np.sum(
            w[i]
            * np.array(gp["x"]["alpha"])
            / (np.array(gp["x"]["beta"]) ** 2)
            * dthetaix
            * np.exp(-(dthetaix**2) / (2 * np.array(gp["x"]["beta"]) ** 2))
        )

        Y -= dt * np.sum(
            w[i]
            * np.array(gp["y"]["alpha"])
            / (np.array(gp["y"]["beta"]) ** 2)
            * dthetaiy
            * np.exp(-(dthetaiy**2) / (2 * np.array(gp["y"]["beta"]) ** 2))
        )

        Z -= dt * np.sum(
            w[i]
            * np.array(gp["z"]["alpha"])
            / (np.array(gp["z"]["beta"]) ** 2)
            * dthetaiz
            * np.exp(-(dthetaiz**2) / (2 * np.array(gp["z"]["beta"]) ** 2))
        )

        thetax = R0[0] + RESP_ANG_X * brwave[i]
        thetay = R0[1] + RESP_ANG_Y * brwave[i]
        thetaz = R0[2] + RESP_ANG_Z * brwave[i]

        R = rotate_xyz(thetax, thetay, thetaz)

        VCG[:, i] = R @ L @ np.array([X, Y, Z])

        if traj.shape[0] > 1:
            dr = dpos[i]
            diff = epos - dr
            den_norm = 1.0 / np.sqrt(np.sum(diff**2, axis=1)) ** 3
            h_1 = np.diag(den_norm) @ diff
            H[:, :, i] = h_1

    B, A = butter(5, 0.7 * 2 / fs, btype="lowpass")
    opol = 6

    for i in range(3):
        p = np.polyfit(np.arange(N), VCG[i, :], opol)
        trend = np.polyval(p, np.arange(N))
        VCG[i, :] -= trend
        VCG[i, :] -= filtfilt(B, A, VCG[i, :])
        VCG[i, :] /= np.max(np.abs(VCG[i, :]))

    dmodel = DmodelParameters(
        H, VCG, theta, traj, STM, RESP_ANG_X, RESP_ANG_Y, RESP_ANG_Z, HEART_T_RESP
    )
    if ect:
        dmodel.stm = STM

    return dmodel
