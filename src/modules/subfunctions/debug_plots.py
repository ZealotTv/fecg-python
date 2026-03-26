import matplotlib.pyplot as plt
import numpy as np
from .datatypes import GeneratorOut


def debug_plots(out: GeneratorOut, debug):

    f_handles = []

    NB_FOETUSES = len(out.f_model)

    print("Generating plots ...")

    col = np.array(
        [
            [1, 0, 0],
            [0, 0, 1],
            [0, 0.8, 0],
            [0.4, 0.4, 0],
            [0, 0.8, 0.8],
            [0.4, 0, 0.8],
            [0.8, 0.4, 1],
            [0.4, 0.4, 1],
        ]
    )

    LINE_WIDTH = 2
    FONT_SIZE = 12

    fs = out.params.fs
    n = out.params.n
    tm = np.arange(1 / fs, n / fs + 1 / fs, 1 / fs)

    NB_EL2PLOT = min(3, out.mixture.shape[0])

    fig1, ax = plt.subplots(NB_EL2PLOT, 1, figsize=(8, 6), sharex=True)
    fig1.suptitle("Some generated AECG")
    f_handles.append(fig1)

    if NB_EL2PLOT == 1:
        ax = [ax]

    for ee in range(NB_EL2PLOT):
        ax[ee].plot(tm, out.mixture[ee, :], color=col[ee], linewidth=LINE_WIDTH)
        ax[ee].set_ylabel("Amplitude [NU]")
        ax[ee].tick_params(labelsize=FONT_SIZE)

    ax[-1].set_xlabel("Time [sec]")

    # ====================================
    # == VCG plots
    # ====================================
    if debug > 1:
        fig2, ax = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
        fig2.suptitle("VCG plots")
        f_handles.append(fig2)

        for vv in range(3):
            # maternal
            ax[vv, 0].plot(
                tm,
                out.m_model.VCG[vv, :],
                color=col[2 * vv],
                linewidth=LINE_WIDTH,
            )
            ax[vv, 0].plot(tm[out.mqrs], out.m_model.VCG[vv, out.mqrs], "+k")
            ax[vv, 0].set_ylabel("Amplitude")
            ax[vv, 0].set_xlim(0, 4)
            ax[vv, 0].set_title(f"Mother VCG channel {vv + 1}")

            # fetal
            for fet in range(NB_FOETUSES):
                ax[vv, 1].plot(
                    tm,
                    out.f_model[fet].VCG[vv, :],
                    color=col[(2 * vv + fet + 1) % len(col)],
                    linewidth=LINE_WIDTH,
                )
                ax[vv, 1].plot(
                    tm[out.fqrs[fet]],
                    out.f_model[fet].VCG[vv, out.fqrs[fet]],
                    "+k",
                )
                ax[vv, 1].set_xlim(0, 4)

            ax[vv, 1].set_title(f"Foetus VCG channel {vv + 1}")

        ax[-1, 0].set_xlabel("Time [sec]")
        ax[-1, 1].set_xlabel("Time [sec]")
        plt.savefig("VCG plots.pdf")
    # ====================================
    # == MECG & FECG before mixing
    # ====================================
    if debug > 2:
        fig3, ax = plt.subplots(NB_EL2PLOT, 1, figsize=(8, 6), sharex=True)
        fig3.suptitle("Projected FECG and MECG before mixing")
        f_handles.append(fig3)

        if NB_EL2PLOT == 1:
            ax = [ax]
        for ee in range(NB_EL2PLOT):
            ax[ee].plot(tm, out.mecg[ee, :], "b", linewidth=LINE_WIDTH, label="MECG")

            for fet in range(NB_FOETUSES):
                ax[ee].plot(
                    tm,
                    out.fecg[fet][ee, :],
                    color=col[(fet + 3) % len(col)],
                    linewidth=LINE_WIDTH,
                    label=f"FECG {fet + 1}",
                )

            ax[ee].legend()
            ax[ee].set_ylabel("Amplitude")
            ax[ee].set_xlim(0, 4)

        ax[-1].set_xlabel("Time [sec]")
        plt.savefig("Projected before mixing.pdf")

    if debug > 3:
        fig3, ax = plt.subplots(NB_EL2PLOT, 1, figsize=(8, 6), sharex=True)
        fig3.suptitle("Projected FECG and MECG after mixing")
        f_handles.append(fig3)

        if NB_EL2PLOT == 1:
            ax = [ax]
        for ee in range(NB_EL2PLOT):
            ax[ee].plot(tm, out.mixture[ee, :], "b", linewidth=LINE_WIDTH, label="MECG")
            ax[ee].legend()
            ax[ee].set_ylabel("Amplitude")
            ax[ee].set_xlim(0, 3)

        ax[-1].set_xlabel("Time [sec]")
        plt.savefig("Projected after mixing.pdf")

    plt.tight_layout()
    return f_handles
