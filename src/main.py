import warnings
import numpy as np
from modules.generate_ecg import generate_ecg
from modules.subfunctions import SimulationParameters
from tqdm import tqdm
from itertools import combinations
from extractor import norm_corr, pipeline, filter_pipe

from functools import partial
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")


def process_comb(comb_ref_id, all_mix, all_fecg, fs):
    scores_across_foetus = []
    for mix, fecg in zip(all_mix, all_fecg):
        diff = mix[comb_ref_id][:-1] - mix[comb_ref_id][-1]
        filtered = pipeline(fs, diff)

        corr_vals = np.array(
            [norm_corr(filtered[i], fecg[comb_ref_id][i]) for i in range(3)]
        )
        score_current = np.round(corr_vals.mean(), 3)
        scores_across_foetus.append(score_current)

    pair_score = np.round(np.median(scores_across_foetus), 3)
    return (pair_score, comb_ref_id)


def main_parallel(fs, all_mix, all_fecg, comb_ref):
    with ThreadPoolExecutor(max_workers=4) as executor:
        func = partial(process_comb, all_mix=all_mix, all_fecg=all_fecg, fs=fs)
        results = list(
            tqdm(
                executor.map(func, comb_ref),
                total=len(comb_ref),
                desc="Parallel processing",
            )
        )
        return results


# for i, comb_ref_id in enumerate((tqdm(comb_ref, desc="Combination"))):

if __name__ == "__main__":
    R_chest = 0.35
    R_hips = 0.4
    R_belly = 0.1
    a = 8.0
    c = 6.0

    v0 = -0.1
    w = 0.3
    n_phi_hearts = 7
    n_z_hearts = 4
    phi_hearts = np.linspace(-np.pi / 3, np.pi / 3, n_phi_hearts)
    z_lin_hearts = np.linspace(-0.3, 0.1, n_z_hearts)
    n_r = 4

    d_vals = np.linspace(0.05, 0.15, n_r)  # расстояние от поверхности

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

    n_phi_pts = 7  # количество по углу
    n_z_pts = 3
    z_lin_pts = np.linspace(-0.35, 0.15, n_z_pts)
    phi_pts = np.linspace(-np.pi / 5, np.pi / 5, n_phi_pts)
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

    fs = 200
    combs = list(combinations(range(points_cyl.shape[0]), 3))
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
        out = generate_ecg(params).to_dict()
        all_mix.append(filter_pipe(fs, out["mixture"]).T)
        all_fecg.append(out["fecg"][0])

    comb_ref = []
    for comb in combs:
        for ref in range(points_cyl.shape[0]):
            if ref in comb:
                continue
            comb_ref.append([*comb, ref])
    reults = main_parallel(fs=fs, all_mix=all_mix, all_fecg=all_fecg, comb_ref=comb_ref)

    print("Done!")
    reults.sort(key=lambda x: x[0], reverse=True)
    top_30 = reults[:top_n]

    print(f"\nTop {top_n} triples:\n")
    for rank, (score, comb_ref_id) in enumerate(top_30, 1):
        print(
            f"{rank:2d}. comb = {comb_ref_id[:-1]}, ref = {comb_ref_id[-1]} -> score = {score}"
        )
