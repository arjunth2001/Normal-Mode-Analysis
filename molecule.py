import numpy as np
import sys


class Molecule:

    def __init__(self, xyz_file, units="Angstrom"):

        self.units = units
        self.read(xyz_file)
        # Mass of argon.. Sir mentioned its argon..
        self.masses = [float(39.96238312251) for i in self.atoms]

    def read(self, xyz_file):
        geom_str = xyz_file.read()
        self.atoms = []
        geom = []
        for line in geom_str.split('\n')[2:]:
            if line.strip() == '':
                continue
            atom, x, y, z = line.split()[:4]
            self.atoms.append(atom)
            geom.append([float(x), float(y), float(z)])
        self.geom = np.array(geom)

    def __len__(self):
        return len(self.geom)

    def __str__(self):
        out = "{:d}\n{:s}\n".format(len(self), self.units)
        for atom, xyz in zip(self.atoms, self.geom):
            out += "{:2s} {: >15.10f} {: >15.10f} {: >15.10f}\n".format(
                atom, *xyz)
        return out

    def bohr(self):
        if self.units == "Angstrom":
            self.geom *= 1.889725989
            self.units = "Bohr"
        return self.geom

    def angs(self):
        if self.units == "Bohr":
            self.geom /= 1.889725989
            self.units = "Angstrom"
        return self.geom

    def copygeom(self):
        return np.array(self.geom)
