import pickle
from pathlib import Path

from scipy.io import loadmat

data_path = "CHANGE_ME"
data_out_path = "CHANGE_ME"
Path(data_out_path).mkdir(parents=True, exist_ok=True)


p = Path(data_path)
# The EctopicBeatGaussians.mat was skipped
for file in p.iterdir():
    filename = file.stem
    if filename == "readme":
        continue
    data = loadmat(f"{data_path}/{filename}.mat")[f"{filename}"]
    with open(f"{data_out_path}/{filename}.pkl", "wb") as f:
        pickle.dump(data, f)
