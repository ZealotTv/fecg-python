import numpy as np


def rotate_xyz(tetax: float, tetay: float, tetaz: float) -> np.ndarray:
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(tetax), np.sin(tetax)],
            [0, -np.sin(tetax), np.cos(tetax)],
        ]
    )
    Ry = np.array(
        [
            [np.cos(tetay), 0, -np.sin(tetay)],
            [0, 1, 0],
            [np.sin(tetay), 0, np.cos(tetay)],
        ]
    )
    Rz = np.array(
        [
            [np.cos(tetaz), np.sin(tetaz), 0],
            [-np.sin(tetaz), np.cos(tetaz), 0],
            [0, 0, 1],
        ]
    )

    return Rx @ Ry @ Rz
