import os

import warnings
import numpy as np
from modules.generate_ecg import generate_ecg
from modules.subfunctions import SimulationParameters
from tqdm import tqdm
from itertools import combinations
from extractor import norm_corr, pipeline

from functools import partial
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")


def process_comb(comb_ref_id, all_mix, all_fecg, fs):
    scores_across_foetus = []
    for mix, fecg in zip(all_mix, all_fecg):
        diff = mix[comb_ref_id]
        filtered = pipeline(fs, diff)

        corr_vals = np.array(
            [norm_corr(filtered[i], fecg[comb_ref_id][i]) for i in range(3)]
        )
        score_current = np.max(np.round(corr_vals, 3))
        scores_across_foetus.append(score_current)

    pair_score = np.round(np.mean(scores_across_foetus), 3)
    return (pair_score, comb_ref_id)


def main_parallel(fs, all_mix, all_fecg, comb_ref):
    with ThreadPoolExecutor(max_workers=(os.process_cpu_count() or 1) + 4) as executor:
        func = partial(process_comb, all_mix=all_mix, all_fecg=all_fecg, fs=fs)
        results = list(
            tqdm(
                executor.map(func, comb_ref),
                total=len(comb_ref),
                desc="Parallel processing",
            )
        )
        return results


if __name__ == "__main__":
    R_CHEST = 0.35
    R_HIPS = 0.4
    R_BELLY = 0.1
    A = 8.0
    C = 6.0
    V0 = -0.1
    W = 0.3

    # HEARTS
    N_PHI_HEARTS_RIGHT = 5
    N_PHI_HEARTS_LEFT = 3
    N_Z_HEARTS = 3
    N_R = 2
    phi_right_hearts = np.linspace(-5 * np.pi / 18, 0, N_PHI_HEARTS_RIGHT)
    phi_left_hearts = np.linspace(np.pi / 12, np.pi / 6, N_PHI_HEARTS_LEFT)
    phi_hearts = np.concatenate([phi_right_hearts, phi_left_hearts])
    z_low_hearts = np.linspace(-0.35, -0.07, N_Z_HEARTS)
    z_high_hearts = np.array([0.04])
    z_lin_hearts = np.concatenate([z_low_hearts, z_high_hearts])

    d_vals = np.linspace(0.04, 0.11, N_R)
    phi_3d, z_3d, d_3d = np.meshgrid(phi_hearts, z_lin_hearts, d_vals, indexing="ij")
    R_base_pts = R_CHEST * np.exp(-A * (z_3d - 0.3) ** 2) + R_HIPS * np.exp(
        -C * (z_3d + 0.3) ** 2
    )
    belly_profile_pts = 1 / (1 + ((z_3d - V0) / W) ** 4)
    front_pts = np.maximum(0, np.cos(phi_3d))
    belly_pts = R_BELLY * belly_profile_pts * front_pts
    R_surface = R_base_pts + belly_pts
    R_inner = R_surface - d_3d
    R_inner = np.maximum(R_inner, 0)
    x_center = 0.1 * np.exp(-10 * (z_3d - V0) ** 2)
    x_inner = x_center + R_inner * np.cos(phi_3d)
    y_inner = R_inner * np.sin(phi_3d)
    z_inner = z_3d
    phi_flat = phi_3d.flatten()
    R_flat = R_inner.flatten()
    z_flat = z_3d.flatten()
    points_inner_cyl = np.column_stack((phi_flat, R_flat, z_flat))

    # ELECTRODES
    N_PHI_PTS_RIGHT = 6
    N_PHI_PTS_LEFT = 3
    N_Z_PTS = 3
    z_low_pts = np.linspace(-0.35, -0.1, N_Z_PTS)
    z_high_pts = np.array([0.07])
    z_lin_pts = np.concatenate([z_low_pts, z_high_pts])
    phi_right_pts = np.linspace(-np.pi / 3, 0, N_PHI_PTS_RIGHT)
    phi_left_pts = np.linspace(np.pi / 12, np.pi / 4, N_PHI_PTS_LEFT)
    phi_pts = np.concatenate([phi_right_pts, phi_left_pts])
    phi_pts, z_lin_pts = np.meshgrid(phi_pts, z_lin_pts)
    R_base_pts = R_CHEST * np.exp(-A * (z_lin_pts - 0.3) ** 2) + R_HIPS * np.exp(
        -C * (z_lin_pts + 0.3) ** 2
    )
    belly_profile_pts = 1 / (1 + ((z_lin_pts - V0) / W) ** 4)
    front_pts = np.maximum(0, np.cos(phi_pts))
    belly_pts = R_BELLY * belly_profile_pts * front_pts
    R_pts = R_base_pts + belly_pts
    x_pts = R_pts * np.cos(phi_pts)
    x_pts += 0.1 * np.exp(-10 * (z_lin_pts - V0) ** 2)
    y_pts = R_pts * np.sin(phi_pts)
    z_pts = z_lin_pts
    x_center = 0.1 * np.exp(-10 * (z_lin_pts - V0) ** 2)
    R_true = np.sqrt(x_pts**2 + y_pts**2)
    phi_true = np.arctan2(y_pts, x_pts)
    phi_flat = phi_true.flatten()
    r_flat = R_true.flatten()
    z_flat = z_lin_pts.flatten()
    points_cyl = np.column_stack((phi_flat, r_flat, z_flat))

    # REFERENCE
    refpos = np.array([points_cyl[7][1], points_cyl[7][0], -0.8])
    points_cyl_ref = np.vstack([points_cyl, refpos])

    FS = 200
    TIME = 20
    combs = list(combinations(range(points_cyl.shape[0]), 4))
    top_n = 50
    all_mix = []
    all_fecg = []

    for foetus in tqdm(range(points_inner_cyl.shape[0]), desc="Foetus generation"):
        params = SimulationParameters(
            elpos=points_cyl,
            fheart=[points_inner_cyl[foetus]],
            ntype=np.array([""]),
            SNRmn=6,
            fs=FS,
            n=FS * TIME,
            ftraj=np.array(["none"]),
            ground=True,
        )
        out = generate_ecg(params).to_dict()
        all_mix.append(out["mixture"])
        all_fecg.append(out["fecg"][0])

    comb_ref = []
    for comb in combs:
        comb_ref.append([*comb])
    reults = main_parallel(fs=FS, all_mix=all_mix, all_fecg=all_fecg, comb_ref=comb_ref)

    print("Done!")
    reults.sort(key=lambda x: x[0], reverse=True)
    top_30 = reults[:top_n]

    print(f"\nTop {top_n} triples:\n")
    for rank, (score, comb_ref_id) in enumerate(top_30, 1):
        print(f"{rank:2d}. comb = {comb_ref_id} -> score = {score}")
