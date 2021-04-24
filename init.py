import numpy as np
from itertools import combinations
from pbc import pbc_sep


def get_config():
    P = []
    with open("molecule.xyz", "w") as f:
        print(108, file=f)
        print("\n", end="", file=f)
        while True:
            p = np.random.rand(3,) * 18
            skip = False
            for p_in_box in P:
                if np.linalg.norm(p - p_in_box) <= 3.4:
                    skip = True
                    # print("skipping...")
            # print(len(P))
            if not skip:
                P.append(p)
                if len(P) == 108:
                    break
        for point in P:
            print(f"C {point[0]} {point[1]} {point[2]}", file=f)
