from molecule import Molecule
import sys
import numpy as np
from matplotlib import pyplot as plt
hartree2J = 4.359744e-18
amu2kg = 1.660538782e-27
bohr2m = 0.52917720859e-10
c = 2.99792458E8


class Frequencies:

    def __init__(self, mol, hessString):

        self.mol = mol
        self.hess = hessString
        self.N = mol.__len__()

        m = []
        for i in range(self.N):
            m += [1/(mol.masses[i])**0.5]*3
        self.MM = np.diag(m)
        self.m = m

    def get_MWhessian(self):

        H0 = np.matrix([i.split() for i in self.hess.splitlines()], float)
        mwH = np.dot(self.MM, np.dot(H0, self.MM))
        return mwH

    def get_frequencies(self):

        self.e, self.l = np.linalg.eigh(self.get_MWhessian())
        self.Q = np.matrix(self.MM)*np.matrix(self.l)
        freq = []
        conv = np.sqrt(hartree2J/(amu2kg*bohr2m**2)
                       ) / (c*2*np.pi)  # dimensional analysis
        # print(conv)
        for i in self.e:
            if i < 0:
                freq.append((-i)**0.5*conv)
            else:
                freq.append(i**0.5*conv)

        return freq

    def frequency_output(self, output):

        mol = self.mol
        freq = self.get_frequencies()

        t = open(output, "w")
        for i in range(3*self.N):
            t.write("%d\n%s cm^{-1}\n" % (self.N, str(freq[i])))
            for j in range(self.N):
                atom = mol.atoms[j]
                x, y, z = mol.geom[j, 0], mol.geom[j, 1], mol.geom[j, 2]
                dx, dy, dz = self.Q[3*j, i], self.Q[3*j+1, i], self.Q[3*j+2, i]
                t.write("{:s}{:12.7f}{:12.7f}{:12.7f}\n".format(atom, x, y, z))
            t.write("\n")
        t.close()
        a = np.array(freq)
        plt.hist(a, bins=100)
        plt.title("Frequency Histogram")
        plt.savefig("hist.png")
        # plt.show()

        return None
