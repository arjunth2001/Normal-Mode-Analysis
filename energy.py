import numpy as np
import autograd
from pbc import pbc_sep


def get_energy(geom):
    eps = 0.238
    sigma = 3.4
    te = 0
    tgeom = np.array(geom)
    pairs = [(a, b) for idx, a in enumerate(tgeom)
             for b in tgeom[idx + 1:]]
    for pair in pairs:
        rij = np.linalg.norm(pbc_sep(pair[0], pair[1]))
        if rij == 0:
            continue
        te += 4*eps*((sigma/rij)**12-(sigma/rij)**6)

    return te
