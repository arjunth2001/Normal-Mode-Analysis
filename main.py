from init import get_config
from energy import get_energy
from molecule import Molecule
from grad import gradient_descent
from hessian import Hessian
from frequencies import Frequencies
from pbc import pbc_sep
import numpy as np
if __name__ == '__main__':
    print("Starting to make initial configuration.")
    get_config()  # Q1
    init_mol = None
    with open("new_molecule.xyz") as f:
        print("Initial Configuration Generated for Arg LJ Model")
        print("Initial COnfiguration for Q1 will the saved to molecule.xyz")
        init_mol = Molecule(f, "Angstrom")
        print("Starting to do Q2 and Q3...")
        e = get_energy(np.array(init_mol.geom))  # Q2
        # print(e)
        # Q3 Steapest Descend is a heuristic.. Needs tuning of parameters...
        print("Starting Steepest Descent Algoithm to minimise Energy...")
        g = gradient_descent(0.135, 100, init_mol.geom)
        print("Steepest Descent Algoithm Ended.. Results below\n")
        print("Original Energy: ", e)
        print("New Energy:", np.min(np.array(g[1])))
        print()
        i = np.argmin(np.array(g[1]))
        print("New Configuration will be saved in new_molecule.xyz")
        with open("new_molecule.xyz", "w") as f2:
            print(len(g[0][i]), file=f2)
            print(file=f2)
            for p in g[0][i]:
                print(
                    f"C {p[0]} {p[1]} {p[2]}", file=f2)
    with open("new_molecule.xyz") as f2:
        print("Starting to generate Hessian Matrix... This will take some time. Take a break. Have a KitKat...( Approx 17 mins.. Python is kinda slow.)")
        mol = Molecule(f2, "Angstrom")
        mol.bohr()
        hessian = Hessian(mol, 0.00001)
        hessian.write_Hessian()
        print("Q4 Outputs have been written. as eigen_vectors.dat, eigen_values.dat, hessian.dat")
    with open("new_molecule.xyz", "r") as f2:
        print("Starting Q5..")
        mol = Molecule(f2)
        hessian = open("hessian.dat", "r").read()
        freq = Frequencies(mol, hessian)
        freq.frequency_output("modes.xyz")
        print("Modes have been written to modes.xyz")
        print("A histogram for the frequencies have been saved as hist.png")
        print("Kindly look at Report for the output format of these files and what they contain..")
        print("End")
