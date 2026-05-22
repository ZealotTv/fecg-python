import numpy as np
import pickle
from numpy.linalg import svd
from itertools import combinations
from tqdm import tqdm


for i in tqdm(range(300), desc="Dataset loading"):
    with open(f"Foetus_{i}.pkl", "rb") as f:
        temp_file = pickle.load(f)


def svd_vals(A):
    return svd(A, compute_uv=False)


def metrics_for_set(S, all_H):
    sig3_vals = []
    volumes = []

    for H in all_H:
        A = H[list(S), :]
        if A.shape[0] < 3:
            sig3_vals.append(0.0)
            volumes.append(0.0)
            continue

        s = svd_vals(A)
        sig3_vals.append(s[-1])
        volumes.append(s[0] * s[1] * s[2])

    return np.array(sig3_vals), np.array(volumes)


def combined_score(S, all_H, q=10, w1=1.0, w2=0.2):
    sig3_vals, volumes = metrics_for_set(S, all_H)

    # робастность
    f1 = np.percentile(sig3_vals, q)

    # максимум объёма
    #    f2 = np.max(volumes)

    return w1 * f1


def find_top_triples_with_print(all_H, top_n=10, q=10, w1=1.0, w2=0.2):
    N_el = all_H[0].shape[0]
    scored = []
    combs = list(combinations(range(N_el), 3))
    for S in tqdm(combs, desc="Combinations(3)"):
        sc = combined_score(S, all_H, q=q, w1=w1, w2=w2)
        scored.append((sc, S))

    # сортировка по убыванию
    scored.sort(key=lambda x: x[0], reverse=True)

    # вывод топ-N
    print(f"\nTop {top_n} triples:\n")
    for i, (sc, S) in enumerate(scored[:top_n], 1):
        print(f"{i:2d}. Electrodes {S} -> score = {sc:.6f}")

    return scored[:top_n]


def find_top_quads(all_H, top_n=10, q=10, w1=1.0, w2=0.2):
    N_el = all_H[0].shape[0]
    scored = []
    combs = list(combinations(range(N_el), 4))
    for S4 in tqdm(combs, desc="Combinations(4)"):
        sc = combined_score(S4, all_H, q=q, w1=w1, w2=w2)
        scored.append((sc, S4))

    scored.sort(key=lambda x: x[0], reverse=True)
    filtered = []
    for sc, S in scored:
        keep = True

        for existing_sc, _ in filtered:
            if abs(sc - existing_sc) < 0.01:
                keep = False
                break

        if keep:
            filtered.append((sc, S))

        if len(filtered) >= top_n:
            break

    print(f"\nTop {top_n} quadruples:\n")
    for i, (sc, S) in enumerate(filtered, 1):
        print(f"{i:2d}. {S} -> score = {sc:.6f}")

    return filtered


def run_selection(all_H, top_n=50):
    top_quads = find_top_quads(all_H, top_n=top_n)
    return top_quads


# run_selection(np.array(all_H), top_n=50)
