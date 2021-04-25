import numpy as np
from itertools import combinations
from pbc import pbc_sep


def get_config():
    P = []
    count = 0
    relax = 4
    with open("molecule.xyz", "w") as f:
        print(108, file=f)
        print("\n", end="", file=f)
        while True:
            p = np.random.rand(3,) * 18
            skip = False
            for p_in_box in P:
                if np.linalg.norm(pbc_sep(p, p_in_box)) <= 3.4:
                    skip = True
                    # print("skipping...")
            print(len(P))
            if not skip:
                P.append(p)
                count = 0
                if len(P) == 108:
                    break
            else:
                count += 1
                if count >= 10000:
                    count = 0
                    tempP = []
                    tempP.append(p)
                    for j in range(len(P)):
                        if np.linalg.norm(pbc_sep(P[j], p)) >= 3.4:
                            tempP.append(P[j])
                    P = tempP

        for point in P:
            print(f"C {point[0]} {point[1]} {point[2]}", file=f)
