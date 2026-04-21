import warnings
import pickle
import numpy as np
from modules.generate_ecg import generate_ecg
from modules.subfunctions import pol2cart, SimulationParameters

warnings.filterwarnings("ignore")
x_el = np.pi / 12 * np.linspace(3, 10, 20) - np.pi / 2
x_el = x_el.reshape(-1, 1)
y_el = 0.5 * np.ones((len(x_el), 1))
xy_el = np.tile(np.hstack((x_el, y_el)), (5, 1))
z_el = np.tile(np.linspace(-0.1, -0.4, 5), (len(x_el), 1))
z_el = z_el.reshape(len(xy_el), 1, order="F")
abdmleads = np.hstack((xy_el, z_el))
Xc, Yc = pol2cart(abdmleads[:, 1], abdmleads[:, 0])
epos = np.array([Xc, Yc, abdmleads[:, 2]]).T

el_idx = [[30, 55, 75, 90], [10, 55, 75, 90], [12, 89, 96, 97]]
# theta = np.linspace(-np.pi / 3, np.pi / 3, 12)
# r = np.linspace(0.25, 0.35, 8)
# z = np.linspace(-0.4, -0.2, 5)

# T, R, Z = np.meshgrid(theta, r, z, indexing="ij")

# matrix = np.stack((T.ravel(), R.ravel(), Z.ravel()), axis=1)
f_hearts = [
    [-np.pi / 3, 0.35, -0.3],
    # [-np.pi / 3, 0.35, -0.2],
    # [-np.pi / 3, 0.35, -0.1],
    # [-np.pi / 3, 0.3, -0.3],
    # [-np.pi / 3, 0.3, -0.2],
    # [-np.pi / 3, 0.3, -0.1],
    # [-np.pi / 3, 0.25, -0.3],
    # [-np.pi / 3, 0.25, -0.2],
    # [-np.pi / 3, 0.25, -0.1],
    # [-np.pi / 5, 0.35, -0.3],
    # [-np.pi / 5, 0.35, -0.2],
    # [-np.pi / 5, 0.35, -0.1],
    # [-np.pi / 5, 0.3, -0.3],
    # [-np.pi / 5, 0.3, -0.2],
    # [-np.pi / 5, 0.3, -0.1],
    # [-np.pi / 5, 0.25, -0.3],
    # [-np.pi / 5, 0.25, -0.2],
    # [-np.pi / 5, 0.25, -0.1],
    # [0, 0.35, -0.3],
    # [0, 0.35, -0.2],
    # [0, 0.35, -0.1],
    # [0, 0.3, -0.3],
    # [0, 0.3, -0.2],
    # [0, 0.3, -0.1],
    # [0, 0.25, -0.3],
    # [0, 0.25, -0.2],
    # [0, 0.25, -0.1],
    # [np.pi / 5, 0.35, -0.3],
    # [np.pi / 5, 0.35, -0.2],
    # [np.pi / 5, 0.35, -0.1],
    # [np.pi / 5, 0.3, -0.3],
    # [np.pi / 5, 0.3, -0.2],
    # [np.pi / 5, 0.3, -0.1],
    # [np.pi / 5, 0.25, -0.3],
    # [np.pi / 5, 0.25, -0.2],
    # [np.pi / 5, 0.25, -0.1],
    # [np.pi / 3, 0.35, -0.3],
    # [np.pi / 3, 0.35, -0.2],
    # [np.pi / 3, 0.35, -0.1],
    # [np.pi / 3, 0.3, -0.3],
    # [np.pi / 3, 0.3, -0.2],
    # [np.pi / 3, 0.3, -0.1],
    # [np.pi / 3, 0.25, -0.3],
    # [np.pi / 3, 0.25, -0.2],
    # [np.pi / 3, 0.25, -0.1],
]
ntype = ["", "MA", "EM", "BW"]
fs = 250
lal = SimulationParameters(
    elpos=epos[el_idx[1]],
    fheart=f_hearts,
    ntype=np.array([""]),
    SNRmn=6,
    fs=fs,
    n=fs * 60,
    ftraj=np.array(["none"]),
)
out = generate_ecg(lal)

with open("Foetus_1_Elpos_2_traj.pkl", "wb") as f:
    pickle.dump(out.to_dict(), f)

print("Done!")
