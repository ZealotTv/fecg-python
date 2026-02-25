import numpy as np

from .datatypes import GaussParameters


def ecg_model(
    ecg_parameters: GaussParameters,
    phasemn: np.linspace = np.linspace(-np.pi, np.pi, 250),
) -> np.linspace:
    alpha, beta, theta = ecg_parameters

    Z = np.zeros(np.size(phasemn))
    for j in range(len(alpha)):
        dtetai = (phasemn - theta[j] + np.pi) % (2 * np.pi) - np.pi
        Z += alpha[j] * np.exp(-(dtetai**2) / (2 * beta[j] ** 2))
    return Z
