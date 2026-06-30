import os

import warnings
import numpy as np
from modules.generate_ecg import generate_ecg
from modules.subfunctions import SimulationParameters
from tqdm import tqdm
from itertools import combinations
from functools import partial
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import pickle
from math import comb

warnings.filterwarnings("ignore")


def signal_fraction(mix, fecg):
    x = mix - np.mean(mix)
    s = fecg - np.mean(fecg)
    a = np.dot(x, s) / np.dot(s, s)
    cov_xy = np.mean(x * s)
    var_x = np.var(x)
    var_s = np.var(s)
    if var_x == 0 or var_s == 0:
        fraction = 1.0 if np.allclose(x, a * s) else 0
    else:
        r2 = (cov_xy**2) / (var_x * var_s)
        fraction = max(0.0, min(1.0, r2))

    return fraction


def comb_iter(num, i):
    for comb_i in combinations(range(num), i):
        for ref in range(num):
            if ref not in comb_i:
                yield [*comb_i, ref]


def process_comb(comb_ref_id, all_mix, all_fecg, fs):
    scores_across_foetus = []
    for mix, fecg in zip(all_mix, all_fecg):
        diff = mix[comb_ref_id][:-1] - mix[comb_ref_id][-1]

        corr_vals = np.array(
            [
                signal_fraction(diff[i], fecg[comb_ref_id][i])
                for i in range(len(comb_ref_id) - 1)
            ]
        )
        score_current = np.max(np.round(corr_vals, 3))
        scores_across_foetus.append(score_current)

    pair_score = np.round(np.mean(scores_across_foetus), 3)
    return (pair_score, comb_ref_id)


def main_parallel(fs, all_mix, all_fecg, comb_ref, name, max_pending, total):
    # with ThreadPoolExecutor(max_workers=(os.process_cpu_count() or 1) + 4) as executor:
    #     func = partial(process_comb, all_mix=all_mix, all_fecg=all_fecg, fs=fs)
    #     with open(name, "wb") as f:
    #         batch = []
    #         for result in tqdm(
    #             executor.map(func, comb_ref),
    #             total=total,
    #             desc="Parallel processing",
    #         ):
    #             batch.append(result)
    #             if len(batch) >= BATCH_SIZE:
    #                 pickle.dump(result, f)
    #                 batch.clear()
    #         if batch:
    #             pickle.dump(batch, f)
    with ThreadPoolExecutor(max_workers=(os.process_cpu_count() or 1) + 4) as executor:
        func = partial(process_comb, all_mix=all_mix, all_fecg=all_fecg, fs=fs)
        futures = set()
        batch = []
        BATCH_SIZE = 50000

        comb_iterator = iter(comb_ref)

        # Заполняем начальный пул задач
        for _ in range(max_pending):
            try:
                comb = next(comb_iterator)
                futures.add(executor.submit(func, comb))
            except StopIteration:
                break

        with open(name, "ab") as f:
            with tqdm(desc="Processing", total=total) as pbar:
                while futures:
                    # Ждём завершения хотя бы одной задачи
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        # Получаем результат и сохраняем в батч
                        result = future.result()
                        batch.append(result)
                        pbar.update(1)

                        # Записываем батч, если накопилось достаточно
                        if len(batch) >= BATCH_SIZE:
                            pickle.dump(batch, f)
                            batch.clear()

                        # Убираем завершённый future
                        futures.remove(future)

                        # Добавляем новую задачу, если есть ещё комбинации
                        try:
                            comb = next(comb_iterator)
                            futures.add(executor.submit(func, comb))
                        except StopIteration:
                            pass

            # Записываем остаток
            if batch:
                pickle.dump(batch, f)


if __name__ == "__main__":
    R_CHEST = 0.35
    R_HIPS = 0.4
    R_BELLY = 0.1
    A = 8.0
    C = 6.0
    V0 = -0.1
    W = 0.3

    # HEARTS
    N_PHI_HEARTS_LEFT = 7
    N_PHI_HEARTS_RIGHT = 4
    N_Z_HEARTS = 4
    N_R = 2
    phi_left_hearts = np.linspace(0, 5 * np.pi / 18, N_PHI_HEARTS_LEFT)
    phi_right_hearts = np.linspace(-np.pi / 17, -np.pi / 4, N_PHI_HEARTS_RIGHT)
    phi_hearts = np.concatenate([phi_left_hearts, phi_right_hearts])
    z_low_hearts = np.linspace(-0.35, -0.07, N_Z_HEARTS)
    z_high_hearts = np.array([0, 0.025, 0.04])
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
    N_PHI_PTS_LEFT = 6
    N_PHI_PTS_RIGHT = 4
    N_Z_PTS = 3
    z_low_pts = np.linspace(-0.4, -0.1, N_Z_PTS)
    z_high_pts = np.array([0, 0.1])
    z_lin_pts = np.concatenate([z_low_pts, z_high_pts])
    phi_left_pts = np.linspace(0, np.pi / 3, N_PHI_PTS_LEFT)
    phi_right_pts = np.linspace(-np.pi / 3, -np.pi / 12, N_PHI_PTS_RIGHT)
    phi_pts = np.concatenate([phi_left_pts, phi_right_pts])
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
    # refpos = np.array([points_cyl[7][1], points_cyl[7][0], -0.8])
    # points_cyl_ref = np.vstack([points_cyl, refpos])

    FS = 200
    TIME = 20

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
            ground=False,
        )
        out = generate_ecg(params).to_dict()
        all_mix.append(out["mixture"])
        all_fecg.append(out["fecg"][0])

    num = points_cyl.shape[0]
    for i in [4, 5]:
        comb_ref = comb_iter(num, i)
        total = comb(num, i) * (num - i)
        # comb_ref = []
        # for comb in combs:
        #     for ref in range(points_cyl.shape[0]):
        #         if ref in comb:
        #             continue
        #         comb_ref.append([*comb, ref])
        main_parallel(
            fs=FS,
            all_mix=all_mix,
            all_fecg=all_fecg,
            comb_ref=comb_ref,
            name=f"data_{i + 1}.pkl",
            max_pending=50000,
            total=total,
        )
