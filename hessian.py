from molecule import Molecule
import numpy as np
import os
import re
from energy import get_energy
from multiprocessing import Pool, Manager
import functools


class Hessian(object):

    def __init__(self, mol, disp_size=0.005):

        self.mol = mol
        self.N = len(self.mol)
        self.h = disp_size
        self.energy = {}

    def find_E(self, i, j, hi, hj):
        """
        :params i,j: indices of atoms 1,2
        :params hi,hj: displacements of atoms 1,2 (-1, 0, or 1, corresponds to -h, 0, or h)
        """
        key = "X%dX%d_%d%d" % (i, j, hi, hj)
        return self.energy[key]

    def set_energy(self, key, geom, d=None):
        e = get_energy(np.array(geom))
        if d != None:
            d[key] = e
            return
        self.energy[key] = e

    def process(self, i, d):
        h, N, atoms, geom = self.h, self.N, self.mol.atoms, self.mol.geom

        print(i)
        for j in range(i):
            forward = "X%dX%d_11" % (i, j)
            reverse = "X%dX%d_-1-1" % (i, j)
            geom_copy2 = self.mol.copygeom()

            geom_copy2[i//3, i % 3] = geom_copy2[i//3, i % 3] + h
            geom_copy2[j//3, j % 3] = geom_copy2[j//3, j % 3] + h

            self.set_energy(forward, geom_copy2, d)

            geom_copy2[i//3, i % 3] = geom_copy2[i//3, i % 3] - 2*h
            geom_copy2[j//3, j % 3] = geom_copy2[j//3, j % 3] - 2*h

            self.set_energy(reverse, geom_copy2, d)

    def run_disps(self):

        h, N, atoms, geom = self.h, self.N, self.mol.atoms, self.mol.geom
        self.set_energy("X0X0_00", geom)

        ####   Run single displacements   ####
        for i in range(3*N):
            print(i)
            forward = "X%dX0_10" % i
            reverse = "X%dX0_-10" % i
            geom_copy = self.mol.copygeom()
            geom_copy[i//3, i % 3] = geom_copy[i//3, i % 3]+h
            self.set_energy(forward, geom_copy)

            geom_copy[i//3, i % 3] = geom_copy[i//3, i % 3]-2*h
            self.set_energy(reverse, geom_copy)
        ####   Run double displacements    ######
        mylist = [*range(3*N)]
        pool = Pool()
        D = Manager().dict()                     # Create a multiprocessing Pool
        pool.map(functools.partial(self.process, d=D), mylist)
        pool.close()
        pool.join()
        self.energy.update(D)

    def make_Hessian(self):

        self.run_disps()

        h, N = self.h, self.N
        E0 = self.find_E(0, 0, 0, 0)
        self.H = np.zeros((3*self.N, 3*self.N))
        for i in range(3*N):
            print(i)
            for i in range(3*N):
                self.H[i, i] = (self.find_E(i, 0, 1, 0) +
                                self.find_E(i, 0, -1, 0)-2*E0)/(h**2)
                for j in range(0, i):
                    self.H[i, j] = (self.find_E(i, j, 1, 1)+self.find_E(i, j, -1, -1)-self.find_E(
                        i, 0, 1, 0)-self.find_E(j, 0, 1, 0)-self.find_E(j, 0, -1, 0)-self.find_E(i, 0, -1, 0)+2*E0)
                    self.H[i, j] /= 2*h**2
                    self.H[j, i] = self.H[i, j]

    def make_eigh(self):
        w, v = np.linalg.eigh(self.H)
        np.savetxt("eigen_vectors.dat", v, "%15.7f", " ", "\n")
        np.savetxt("eigen_values.dat", w, "%15.7f", " ", "\n")

    def write_Hessian(self):
        self.make_Hessian()
        self.make_eigh()
        np.savetxt("hessian.dat", self.H, "%15.7f", " ", "\n")
