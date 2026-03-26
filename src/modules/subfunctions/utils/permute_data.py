import pickle
from pathlib import Path

import numpy as np
from scipy.io import loadmat

data_path = "CHANGE_ME"
data_out_path = "CHANGE_ME"
Path(data_out_path).mkdir(parents=True, exist_ok=True)

p = Path(data_path)
for file in p.iterdir():
    filename = file.stem
    if filename == "EctopicBeatGaussians":
        continue
    fsplit = filename.split("_")
    data = loadmat(f"{data_path}/{filename}.mat")
    alpha = data["Param"][1]
    beta = data["Param"][2]
    theta = data["Param"][0]

    with open(f"{data_out_path}/{fsplit[1]}_{fsplit[-1]}.pkl", "wb") as f:
        pickle.dump([alpha, beta, theta], f)


# old 1
old_1 = {
    "x": {
        "alpha": np.array([0.03, 0.08, -0.13, 0.85, 1.11, 0.75, 0.06, 0.1, 0.17, 0.39]),
        "beta": np.array(
            [
                0.0906,
                0.1057,
                0.0453,
                0.0378,
                0.0332,
                0.0302,
                0.0378,
                0.6040,
                0.3020,
                0.1812,
            ]
        ),
        "theta": np.array([-1.09, -0.83, -0.19, -0.07, 0, 0.06, 0.22, 1.2, 1.42, 1.68]),
    },
    "y": {
        "alpha": np.array([0.035, 0.015, -0.019, 0.32, 0.51, -0.32, 0.04, 0.08]),
        "beta": np.array([0.07, 0.07, 0.04, 0.055, 0.037, 0.0604, 0.450, 0.3]),
        "theta": np.array([-1.1, -0.9, -0.76, -0.11, -0.01, 0.065, 0.8, 1.58]),
    },
    "z": {
        "alpha": np.array(
            [-0.03, -0.14, -0.035, 0.045, -0.4, 0.46, -0.12, -0.2, -0.35]
        ),
        "beta": np.array([0.03, 0.12, 0.04, 0.4, 0.045, 0.05, 0.8, 0.4, 0.2]),
        "theta": np.array([-1.1, -0.93, -0.7, -0.4, -0.15, 0.095, 1.05, 1.25, 1.55]),
    },
}


# old2
old_2 = {
    "x": {
        "alpha": np.array([0.007, -0.011, 0.13, 0.007, 0.0275]),
        "beta": np.array([0.1, 0.03, 0.045, 0.02, 0.3]),
        "theta": np.array([-0.7, -0.17, 0, 0.18, 1.4]),
    },
    "y": {
        "alpha": np.array([0.04, 0.3, 0.45, -0.35, 0.05]),
        "beta": np.array([0.1, 0.05, 0.03, 0.04, 0.3]),
        "theta": np.array([-0.9, -0.08, 0, 0.05, 1.3]),
    },
    "z": {
        "alpha": np.array([-0.014, 0.003, -0.04, 0.046, -0.01]),
        "beta": np.array([0.1, 0.4, 0.03, 0.03, 0.3]),
        "theta": np.array([-0.8, -0.3, -0.1, 0.06, 0.35]),
    },
}


# ectopic
ectopic = {
    "x": {
        "alpha": np.array(
            [0.03, 0.08, -0.13, 0.65, 0.70, 0.01, 0.06, -0.05, -0.17, -0.39, 0.03]
        ),
        "beta": np.array(
            [0.0906, 0.1057, 0.0453, 0.3378, 0.5, 0.2, 0.5, 0.1040, 0.1020, 0.1812, 0.5]
        ),
        "theta": np.array(
            [-1.09, -0.83, -0.19, -0.07, 0, 0.06, 0.22, 1.2, 1.42, 1.68, 2.9]
        ),
    },
    "y": {
        "alpha": np.array(
            [0.035, 0.015, -0.13, 0.35, 0.55, 0.06, 0.06, -0.05, -0.17, -0.39, 0.014]
        ),
        "beta": np.array(
            [0.0906, 0.1057, 0.0453, 0.3378, 0.5, 0.2, 0.5, 0.1040, 0.1020, 0.1812, 0.5]
        ),
        "theta": np.array(
            [-1.1, -0.9, -0.76, -0.07, 0, 0.06, 0.22, 1.2, 1.42, 1.68, 2.9]
        ),
    },
    "z": {
        "alpha": np.array(
            [-0.03, -0.15, -0.013, 0.065, -0.70, 0.01, -0.06, 0.05, 0.17, 0.39, 0.03]
        ),
        "beta": np.array(
            [0.0906, 0.1057, 0.0453, 0.3378, 0.5, 0.2, 0.5, 0.1040, 0.1020, 0.1812, 0.5]
        ),
        "theta": np.array(
            [-1.09, -0.83, -0.19, -0.07, 0, 0.06, 0.22, 1.2, 1.42, 1.68, 2.9]
        ),
    },
}


for i in ["x", "y", "z"]:
    with open(f"{data_out_path}/old_1_{i}.pkl", "wb") as f:
        pickle.dump(list(old_1[i].values()), f)

    with open(f"{data_out_path}/old_2_{i}.pkl", "wb") as f:
        pickle.dump(list(old_2[i].values()), f)

    with open(f"{data_out_path}/ectopic_{i}.pkl", "wb") as f:
        pickle.dump(list(ectopic[i].values()), f)
