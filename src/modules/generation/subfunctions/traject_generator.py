import numpy as np


def traject_generator(
    N: int, pos_i: np.ndarray, pos_f: np.ndarray, type: str
) -> np.ndarray:
    NB_CIRC = 3.5
    traj = np.zeros((N, 3))
    match type:
        case "none":
            traj = pos_i
        case "step":
            traj = np.matlib.repmat(pos_i, N, 1)
            traj[(np.round(N / 2).astype(int)) :] = np.matlib.repmat(
                pos_f, np.round(N / 2).astype(int), 1
            )
        case "linear":
            trajx = np.linspace(pos_i[0], pos_f[0], N)
            trajy = np.linspace(pos_i[1], pos_f[1], N)
            trajz = np.linspace(pos_i[2], pos_f[2], N)
            traj = np.array((trajx, trajy, trajz)).T
            print(traj.shape)
        case "helix":
            traj[:, 2] = np.linspace(pos_i[2], pos_f[2], N).T
            center = (pos_i[:2] + pos_f[:2]) / 2
            r = np.sqrt(np.sum((pos_i[:2] - center) ** 2))
            w = np.linspace(0, 2 * np.pi * NB_CIRC, N).T
            print(pos_i[::1])
            phi = np.atan((pos_i[1] - center[1]) / (pos_i[0] - center[0]))
            traj[:, 0] = r * np.cos(w + phi) + center[0]
            traj[:, 1] = r * np.sin(w + phi) + center[1]
        case _:
            print("TrajectGenerator: Unknown trajectory type")

    return traj
