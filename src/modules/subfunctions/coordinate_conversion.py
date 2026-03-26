# Source - https://stackoverflow.com/a/26757297
# Posted by nzh, modified by community. See post 'Timeline' for change history
# Retrieved 2026-02-25, License - CC BY-SA 3.0

import numpy as np


def cart2pol(x: np.ndarray, y: np.ndarray, z=None) -> np.ndarray:
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    if z:
        return rho, phi, z
    return rho, phi


def pol2cart(rho: np.ndarray, phi: np.ndarray, z=None) -> np.ndarray:
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    if z:
        return x, y, z
    return x, y


def cart2sph(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    return azimuth, elevation, r


def sph2cart(azimuth: np.ndarray, elevation: np.ndarray, r: np.ndarray) -> np.ndarray:
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return x, y, z
