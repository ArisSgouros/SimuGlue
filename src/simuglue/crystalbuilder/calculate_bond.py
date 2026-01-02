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
from simuglue.crystalbuilder.net_types import Atom, Bond, AtomType, BondType

def MinImag(rr, lx, ly, lz, xy, xz, yz, periodicity):
   rr_imag = np.array([ rr[0] - xy*round(rr[1]/ly) - xz*round(rr[2]/lz), \
                        rr[1] - yz*round(rr[2]/lz), \
                        rr[2] ])
   rr_imag[0] -= int(periodicity[0])*lx*round(rr_imag[0]/lx)
   rr_imag[1] -= int(periodicity[1])*ly*round(rr_imag[1]/ly)
   rr_imag[2] -= int(periodicity[2])*lz*round(rr_imag[2]/lz)
   return rr_imag

def CalculateBonds(atoms, box_lmp, periodicity, rc_list, drc):
   lx = box_lmp['xhi'] - box_lmp['xlo']
   ly = box_lmp['yhi'] - box_lmp['ylo']
   lz = box_lmp['zhi'] - box_lmp['zlo']
   xy  = box_lmp['xy']
   xz  = box_lmp['xz']
   yz  = box_lmp['yz']

   bonds = []

   natom = len(atoms)
   bid = 1

   for rc in rc_list:
      rmin2 = pow(rc - drc, 2)
      rmax2 = pow(rc + drc, 2)
      for ii in range(natom):
         ri = atoms[ii].rr
         iid = atoms[ii].aid
         for jj in range(ii+1, natom):
            rj = atoms[jj].rr
            rij = rj - ri
            rij = MinImag(rij, lx, ly, lz, xy, xz, yz, periodicity)
            rij2 = np.dot(rij, rij)
            if rmin2 < rij2 < rmax2:
               rlen = np.sqrt(rij2)
               bonds.append(Bond(bid, atoms[ii], atoms[jj], rlen))
               atoms[ii].neigh.append(atoms[jj])
               atoms[jj].neigh.append(atoms[ii])
               bid += 1

   return bonds
