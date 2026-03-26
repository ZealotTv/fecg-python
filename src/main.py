import warnings
import pickle
import numpy as np
from modules.generate_ecg import generate_ecg
from modules.subfunctions import pol2cart, SimulationParameters

warnings.filterwarnings("ignore")
x = np.pi / 12 * np.array([3, 4, 5, 6, 7, 8, 9, 10]) - np.pi / 2
x = x.reshape(-1, 1)
y = 0.5 * np.ones((8, 1))
xy = np.tile(np.hstack((x, y)), (4, 1))
z = np.tile(np.array([-0.1, -0.2, -0.3, -0.4]), (8, 1))
z = z.reshape(32, 1, order="F")
abdmleads = np.hstack((xy, z))
refs = np.array([[-np.pi / 4, 0.5, 0.4], [(5 / 6 - 0.5) * np.pi, 0.5, 0.4]])
elpos = np.vstack((abdmleads, refs))
Xc, Yc = pol2cart(elpos[:, 1], elpos[:, 0])
epos = np.array([Xc, Yc, elpos[:, 2]]).T

new_electrodes = np.array(
    [
        (epos[3] + epos[4]) / 2,
        (epos[9] + epos[17]) / 2,
        (epos[14] + epos[23]) / 2,
        (epos[27] + epos[28]) / 2,
    ]
)

lal = SimulationParameters(
    elpos=new_electrodes,
    fheart=np.array(
        [
            [-np.pi / 5, 0.35, -0.3],
        ]
    ),
    ntype=np.array(
        [
            "BW",
        ]
    ),
    SNRmn=12,
)
out = generate_ecg(lal)

with open("temp_with_noise_BW_SNR=12.pkl", "wb") as f:
    pickle.dump(out.to_dict(), f)
