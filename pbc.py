import numpy as np


def sep(p1, p2):
    return p2 - p1


def pbc_sep(p1, p2):
    arrow = sep(p1, p2)
    rem = np.mod(arrow, 18)  # The image in the first cube
    mic_separation_vector = np.mod(rem+18/2, 18)-18/2

    return np.array(mic_separation_vector)
