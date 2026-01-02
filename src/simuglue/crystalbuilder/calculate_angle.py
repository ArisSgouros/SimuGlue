###############################################################################
# MIT License
#
# Copyright (c) 2024 ArisSgouros
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
from simuglue.crystalbuilder.net_types import Atom, AtomType, Angle

def MinImag(rr, lx, ly, lz, xy, xz, yz, periodicity):
   rr_imag = np.array([ rr[0] - xy*round(rr[1]/ly) - xz*round(rr[2]/lz), \
                        rr[1] - yz*round(rr[2]/lz), \
                        rr[2] ])
   rr_imag[0] -= int(periodicity[0])*lx*round(rr_imag[0]/lx)
   rr_imag[1] -= int(periodicity[1])*ly*round(rr_imag[1]/ly)
   rr_imag[2] -= int(periodicity[2])*lz*round(rr_imag[2]/lz)
   return rr_imag

def is_list_unique(list_in):
   return len(set(list_in)) == len(list_in)

def CalculateAngles(atoms):
   angles = []

   # populate the angle list
   id_ = 1
   for atom in atoms:
      atom.exist = False
   for iatom in atoms:
      for jatom in iatom.neigh:
         # angle ijk
         if jatom.exist: continue
         for katom in jatom.neigh:
            angle = [iatom, jatom, katom]
            if katom.exist or not is_list_unique(angle): continue
            angles.append(Angle(id_, iatom, jatom, katom))
            id_ += 1
         # angle jik
         for katom in iatom.neigh:
            angle = [jatom, iatom, katom]
            if katom.exist or jatom.aid >= katom.aid or not is_list_unique(angle): continue
            angles.append(Angle(id_, jatom, iatom, katom))
            id_ += 1
      iatom.exist = True

   return angles

def DifferentiateAngleSymmetry(angles):
  for angle in angles:
    ai  = angle.iatom.rr
    aj  = angle.jatom.rr
    ak  = angle.katom.rr

    # label angles with coplanar i and k atoms (same z)
    if m.fabs(ai[2] - ak[2]) < 1.e-5:
      angle.sym = 'T'
    # label angles formed between atoms stacked vertically (same x and y)
    elif m.fabs(ai[0] - ak[0]) < 1.e-5 and m.fabs(ai[1] - ak[1]) < 1.e-5:
      angle.sym = 'N'
    # label other angle types
    else:
      angle.sym = 'A'

  return

def DifferentiateAngleTheta(angles, box_lmp, periodicity):
   lx = box_lmp['xhi'] - box_lmp['xlo']
   ly = box_lmp['yhi'] - box_lmp['ylo']
   lz = box_lmp['zhi'] - box_lmp['zlo']
   xy  = box_lmp['xy']
   xz  = box_lmp['xz']
   yz  = box_lmp['yz']
   for angle in angles:
      ri  = angle.iatom.rr
      rj  = angle.jatom.rr
      rk  = angle.katom.rr

      rji = ri - rj
      rji = MinImag(rji, lx, ly, lz, xy, xz, yz, periodicity)
      rji /= np.linalg.norm(rji)

      rjk = rk - rj
      rjk = MinImag(rjk, lx, ly, lz, xy, xz, yz, periodicity)
      rjk /= np.linalg.norm(rjk)

      costheta = np.clip(np.dot(rji, rjk), -1.0, 1.0)
      theta = np.degrees(np.arccos(costheta))

      angle.theta = theta

   return
