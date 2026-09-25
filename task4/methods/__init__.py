from methods.gcsc import GCSCMethod
from methods.proser import ProserMethod
from methods.rpl import RPLMethod
from methods.vanilla import VanillaMethod

METHODS = {"vanilla": VanillaMethod, "gcsc": GCSCMethod, "proser": ProserMethod, "rpl": RPLMethod}


def get_method(cfg: dict):
    name = cfg["method"]
    if name not in METHODS:
        raise KeyError(f"Unknown method {name!r}; choose from {sorted(METHODS)}")
    return METHODS[name](cfg)
