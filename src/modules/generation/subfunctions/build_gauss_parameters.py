from .load_gauss_parameters import load_gauss_parameters


def build_gauss_parameters(gauss_type, mvcg=None):
    dims = ("x", "y", "z")
    fields = ("alpha", "beta", "theta")

    loaded = {d: load_gauss_parameters(mvcg, gauss_type, d) for d in dims}

    return {d: {f: getattr(loaded[d], f) for f in fields} for d in dims}
