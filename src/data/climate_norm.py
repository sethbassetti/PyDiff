NORM_STATS = {"tas": {"min": -90, "max": 60}, "pr": {"min": 0, "max": 500}}

min_max_norm = {
    "tas": lambda x: (x - NORM_STATS["tas"]["min"])
    / (NORM_STATS["tas"]["max"] - NORM_STATS["tas"]["min"]),
    "pr": lambda x: (x ** (1 / 2) / NORM_STATS["pr"]["max"] ** (1 / 2)),
}

min_max_inv_norm = {
    "tas": lambda x: x * (NORM_STATS["tas"]["max"] - NORM_STATS["tas"]["min"])
    + NORM_STATS["tas"]["min"],
    "pr": lambda x: (x * NORM_STATS["pr"]["max"] ** (1/2)) ** 2,
}
