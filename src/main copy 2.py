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
from pprint import pprint

warnings.filterwarnings("ignore")


def process_comb(comb_ref_id, all_mix, all_fecg, fs):
    scores_across_foetus = []
    for mix, fecg in zip(all_mix, all_fecg):
        diff = mix[comb_ref_id]
        filtered = pipeline(fs, diff)

        corr_vals = np.array(
            [norm_corr(filtered[i], fecg[comb_ref_id][i]) for i in range(3)]
        )
        score_current = np.round(corr_vals.mean(), 3)
        scores_across_foetus.append(score_current)
    return scores_across_foetus


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
    R_chest = 0.35
    R_hips = 0.4
    R_belly = 0.1
    a = 8.0
    c = 6.0

    v0 = -0.1
    w = 0.3
    n_phi_hearts = 7
    n_z_hearts = 3

    phi_hearts = np.linspace(-5 * np.pi / 18, np.pi / 6, n_phi_hearts)
    z_lin_hearts = np.linspace(-0.35, 0.0, n_z_hearts)
    n_r = 3
    d_vals = np.linspace(0.04, 0.11, n_r)  # расстояние от поверхности

    phi_3d, z_3d, d_3d = np.meshgrid(phi_hearts, z_lin_hearts, d_vals, indexing="ij")

    R_base_pts = R_chest * np.exp(-a * (z_3d - 0.3) ** 2) + R_hips * np.exp(
        -c * (z_3d + 0.3) ** 2
    )

    belly_profile_pts = 1 / (1 + ((z_3d - v0) / w) ** 4)
    front_pts = np.maximum(0, np.cos(phi_3d))
    belly_pts = R_belly * belly_profile_pts * front_pts

    R_surface = R_base_pts + belly_pts

    R_inner = R_surface - d_3d

    R_inner = np.maximum(R_inner, 0)

    x_center = 0.1 * np.exp(-10 * (z_3d - v0) ** 2)

    x_inner = x_center + R_inner * np.cos(phi_3d)
    y_inner = R_inner * np.sin(phi_3d)
    z_inner = z_3d
    phi_flat = phi_3d.flatten()
    R_flat = R_inner.flatten()
    z_flat = z_3d.flatten()
    points_inner_cyl = np.column_stack((phi_flat, R_flat, z_flat))

    n_phi_pts = 5  # количество по углу
    n_z_pts = 3
    z_low = np.linspace(-0.35, -0.1, n_z_pts)
    z_high = np.array([0.07])
    z_lin_pts = np.concatenate([z_low, z_high])
    phi_right = np.linspace(-np.pi / 3, 0, 6)
    phi_left = np.linspace(np.pi / 12, np.pi / 4, 3)
    phi_pts = np.concatenate([phi_right, phi_left])
    phi_pts, z_lin_pts = np.meshgrid(phi_pts, z_lin_pts)

    R_base_pts = R_chest * np.exp(-a * (z_lin_pts - 0.3) ** 2) + R_hips * np.exp(
        -c * (z_lin_pts + 0.3) ** 2
    )

    belly_profile_pts = 1 / (1 + ((z_lin_pts - v0) / w) ** 4)

    front_pts = np.maximum(0, np.cos(phi_pts))
    belly_pts = R_belly * belly_profile_pts * front_pts

    R_pts = R_base_pts + belly_pts

    x_pts = R_pts * np.cos(phi_pts)
    x_pts += 0.1 * np.exp(-10 * (z_lin_pts - v0) ** 2)
    y_pts = R_pts * np.sin(phi_pts)
    z_pts = z_lin_pts
    x_center = 0.1 * np.exp(-10 * (z_lin_pts - v0) ** 2)
    R_true = np.sqrt(x_pts**2 + y_pts**2)
    phi_true = np.arctan2(y_pts, x_pts)
    phi_flat = phi_true.flatten()
    r_flat = R_true.flatten()
    z_flat = z_lin_pts.flatten()
    points_cyl = np.column_stack((phi_flat, r_flat, z_flat))
    refpos = np.array([points_cyl[4][1], points_cyl[4][0], -0.8])
    points_cyl_ref = np.vstack([points_cyl, refpos])
    fs = 200
    combs = list(combinations(range(points_cyl.shape[0]), 4))
    top_n = 30
    all_mix = []
    all_fecg = []

    for foetus in tqdm(range(points_inner_cyl.shape[0]), desc="Foetus generation"):
        params = SimulationParameters(
            elpos=points_cyl,
            fheart=[points_inner_cyl[foetus]],
            ntype=np.array([""]),
            SNRmn=6,
            fs=fs,
            n=fs * 20,
            ftraj=np.array(["none"]),
        )
        out = generate_ecg(params, ground=True).to_dict()
        all_mix.append(out["mixture"])
        all_fecg.append(out["fecg"][0])

    comb_ref = []
    for comb in combs:
        comb_ref.append([*comb])

    combs_my = [[2, 4, 12]]
    reults = main_parallel(fs=fs, all_mix=all_mix, all_fecg=all_fecg, comb_ref=combs_my)

    print("Done!")
    print(f"\nTop {len(combs_my)} triples:\n")
    for rank, comb_ref_id in enumerate(combs_my):
        pprint(
            f"{rank + 1}. comb = {comb_ref_id},\nmin = {np.min(reults[rank])},\nmedian = {np.round(np.median(reults[rank]), 3)},\nmean = {np.round(np.mean(reults[rank]), 3)},\nmax = {np.max(reults[rank])}\nsorted = {np.sort(reults[rank])}"
        )
