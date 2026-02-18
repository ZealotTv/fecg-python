from dataclasses import dataclass


@dataclass
class HRVParameters:
    hr: int = 90
    lfhfr: float = 0.6
    hrstd: int = 2
    flo: int = 0
    fhi = 0.25
    acc: int = 0
    typeacc: str = "nsr"
    accmean: int = 0
    accstd: int = 1
