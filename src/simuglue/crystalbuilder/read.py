###############################################################################
# MIT License
#
# Copyright (c) 2023 ArisSgouros
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###############################################################################

import sys
import os
import math as m
import numpy as np
import argparse
import copy as cp
from simuglue.crystalbuilder.net_types import Atom, Bond, AtomType, BondType

# Parse basis files subject to the constraint coordinate system used in Lammps [see https://docs.lammps.org/Howto_triclinic.html]
def ReadBasis(path_basis):
   n_atom = 0
   n_atom_types = 0
   xlo = 0.0
   xhi = 0.0
   ylo = 0.0
   yhi = 0.0
   zlo = 0.0
   zhi = 0.0
   xy = 0.0
   xz = 0.0
   yz = 0.0

   g = open(path_basis, 'r')
   lines = []
   line_num = 0
   while True:
      line = g.readline()
      # Check if this is the end of file; if yes break the loop.
      if line == '':
         break

      line_split = line.split()

      if "atoms" in line:
         n_atom = int(line_split[0])
      if "atom types" in line:
         n_atom_types = int(line_split[0])
      if "xlo xhi" in line:
         xlo = float(line_split[0])
         xhi = float(line_split[1])
      if "ylo yhi" in line:
         ylo = float(line_split[0])
         yhi = float(line_split[1])
      if "zlo zhi" in line:
         zlo = float(line_split[0])
         zhi = float(line_split[1])
      if "xy xz yz" in line:
         xy = float(line_split[0])
         xz = float(line_split[1])
         yz = float(line_split[2])
      if "Masses" in line:
         Masses_start = line_num + 2
      if "Atoms" in line:
         Atoms_start = line_num + 2
      lines.append(line)
      line_num += 1
   g.close()
   #
   # Read the Masses section
   atom_types    = {}
   try:
      for ii in range(n_atom_types):
         line  = lines[Masses_start+ii].split()
         itype = int(line[0])
         mass  = float(line[1])
         if len(line) >= 4:
            name = line[3]
         else:
            name = itype
         atom_types[itype] = AtomType(itype, mass, name)
   except:
      print("Problem with reading masses section/not existing")
      sys.exit()
   #
   # Read the Atoms section
   basis_atoms   = []
   try:
      for ii in range(n_atom):
         line  = lines[Atoms_start+ii].split()
         aid   = int(line[0])
         imol  = int(line[1])
         itype = int(line[2])
         qq    = float(line[3])
         xx    = float(line[4])
         yy    = float(line[5])
         zz    = float(line[6])
         rr    = np.array([xx, yy, zz])
         name  = atom_types[itype].name
         basis_atoms.append(Atom(aid, imol, itype, qq, rr, name))
   except:
      print("Problem with reading atom section")
      sys.exit()

   # Calculate the origin
   basis_orig = np.array([xlo, ylo, zlo])

   # Calculate the basis vector [see https://docs.lammps.org/Howto_triclinic.html]
   basis_vectors = []
   basis_vectors.append(np.array([xhi-xlo, 0.0    , 0.0]))
   basis_vectors.append(np.array([xy     , yhi-ylo, 0.0]))
   basis_vectors.append(np.array([xz     , yz     , zhi-zlo]))
   return [basis_vectors, basis_orig, basis_atoms, atom_types]
