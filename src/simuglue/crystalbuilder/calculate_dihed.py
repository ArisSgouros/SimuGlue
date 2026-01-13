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
from simuglue.crystalbuilder.net_types import Atom, AtomType, Dihed

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

def CalculateDiheds(atoms):
   diheds = []

   # populate the dihed list
   id_ = 1
   for atom in atoms:
      atom.exist = False
   for iatom in atoms:
      for jatom in iatom.neigh:
         # dihed ijkl
         if jatom.exist: continue
         for katom in jatom.neigh:
            if katom.exist or katom == iatom: continue
            for latom in katom.neigh:
               dihed = [iatom, jatom, katom, latom]
               if latom.exist or not is_list_unique(dihed): continue
               diheds.append(Dihed(id_, iatom, jatom, katom, latom))
               id_ += 1
         # dihed jikl
         for katom in iatom.neigh:
            if katom.exist or katom == jatom: continue
            for latom in katom.neigh:
               dihed = [jatom, iatom, katom, latom]
               if latom.exist or not is_list_unique(dihed): continue
               diheds.append(Dihed(id_, jatom, iatom, katom, latom))
               id_ += 1
      iatom.exist = True

   return diheds

# """Praxeolitic formula
# 1 sqrt, 1 cross product"""
# https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
def PhiDihed(r0, r1, r2, r3, box_lmp, periodicity):
  lx = box_lmp['xhi'] - box_lmp['xlo']
  ly = box_lmp['yhi'] - box_lmp['ylo']
  lz = box_lmp['zhi'] - box_lmp['zlo']
  xy  = box_lmp['xy']
  xz  = box_lmp['xz']
  yz  = box_lmp['yz']

  b0 = -1.0*(r1 - r0)
  b1 = r2 - r1
  b2 = r3 - r2

  b0 = MinImag(b0, lx, ly, lz, xy, xz, yz, periodicity)
  b1 = MinImag(b1, lx, ly, lz, xy, xz, yz, periodicity)
  b2 = MinImag(b2, lx, ly, lz, xy, xz, yz, periodicity)

  # normalize b1 so that it does not influence magnitude of vector
  # rejections that come next
  b1 /= np.linalg.norm(b1)

  # vector rejections
  # v = projection of b0 onto plane perpendicular to b1
  #   = b0 minus component that aligns with b1
  # w = projection of b2 onto plane perpendicular to b1
  #   = b2 minus component that aligns with b1
  v = b0 - np.dot(b0, b1)*b1
  w = b2 - np.dot(b2, b1)*b1

  # angle between v and w in a plane is the torsion angle
  # v and w may not be normalized but that's fine since tan is y/x
  x = np.dot(v, w)
  y = np.dot(np.cross(b1, v), w)
  return np.arctan2(y, x)

def DifferentiateCisTrans(diheds, lbox, periodicity):
  for dihed in diheds:
    ai  = dihed.iatom.rr
    aj  = dihed.jatom.rr
    ak  = dihed.katom.rr
    al  = dihed.latom.rr

    phi = PhiDihed(ai, aj, ak, al, lbox, periodicity)

    if abs(phi) > 0.5*np.pi:
      # trans
      dihed.orient = 'trans'
    else:
      # cis
      dihed.orient = 'cis'
    dihed.phi = phi
  return
