import pickle

from .datatypes import GaussParameters

data_path = "src/modules/generation/subfunctions/data/vcg_sets/"


def load_gauss_parameters(
    vcgmodel: int = 0, type: str = "normal", dimention: str = "x"
) -> GaussParameters:
    vcgList = [
        "old_1",
        "old_2",
        "s0273lre",
        "s0291lre",
        "s0302lre",
        "s0303lre",
        "s0306lre",
        "s0491",
        "s0533",
    ]
    try:
        match type:
            case "normal":
                model = vcgList[vcgmodel]

                with open(f"{data_path}/{model}_{dimention}.pkl", "rb") as f:
                    data = pickle.load(f)
                    return GaussParameters(*data)
            case "ectopic":
                with open(f"{data_path}/ectopic_{dimention}.pkl", "rb") as f:
                    data = pickle.load(f)
                    return GaussParameters(*data)
    except Exception:
        print("Smth gone wrong!")
