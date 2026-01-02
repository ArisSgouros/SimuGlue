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

class Cell:
  def __init__(self, id_, cx, cy):
    self.id = id_
    self.cx = cx
    self.cy = cy
    self.atoms = []
    self.neigh = []
    return

class GridXy:
  def __init__(self, box_lmp, rc):
    self.lx      = box_lmp['xhi'] - box_lmp['xlo']
    self.ly      = box_lmp['yhi'] - box_lmp['ylo']
    self.rc      = rc
    self.ncellx  = m.floor(self.lx / rc)
    self.ncelly  = m.floor(self.ly / rc)
    self.ncells  = self.ncellx*self.ncelly
    self.celldx  = self.lx / self.ncellx
    self.celldy  = self.ly / self.ncelly
    self.cells   = []
    for id_ in range(self.ncells):
       cx = int(id_%self.ncellx)
       cy = int(id_/self.ncellx)
       self.cells.append(Cell(id_, cx, cy))
    return

def CalculateBondsGridXy(atoms, box_lmp, periodicity, rc_list, drc):
   lx = box_lmp['xhi'] - box_lmp['xlo']
   ly = box_lmp['yhi'] - box_lmp['ylo']
   lz = box_lmp['zhi'] - box_lmp['zlo']
   xy  = box_lmp['xy']
   xz  = box_lmp['xz']
   yz  = box_lmp['yz']

   if m.fabs(xy) > 1.e-5 or \
      m.fabs(xz) > 1.e-5 or \
      m.fabs(yz) > 1.e-5:
       print('ERROR(CalculateBondsGridXy): the gridxy optimization does not work with skewed boxes..')
       sys.exit()

   rc_cell = np.max(rc_list[0])+drc
   grid = GridXy(box_lmp, rc_cell)

   #print("lx    : ", grid.lx)
   #print("ly    : ", grid.ly)
   #print("rc    : ", grid.rc)
   #print("ncellx: ", grid.ncellx)
   #print("ncelly: ", grid.ncelly)
   #print("ncells: ", grid.ncells)

   for cell in grid.cells:
      #print(cell.id, cell.cx, cell.cy)
      for ineigh in [[0, 0], [1, 0], [1, 1], [0, 1], [-1, 1]]:
         jcx = cell.cx + ineigh[0]
         jcy = cell.cy + ineigh[1]
         
         if jcx == grid.ncellx or jcx == -1:
            jcx = (grid.ncellx - 1)*(jcx == -1)
         if jcy == grid.ncelly or jcy == -1:
            jcy = (grid.ncelly - 1)*(jcy == -1)
         jid = jcx + jcy*grid.ncellx

         cell.neigh.append(grid.cells[jid])

   #for icell in grid.cells:
   #   for jcell in icell.neigh:
   #      print(icell.id, jcell.id)

   # place atoms in cells
   for ii in range(len(atoms)):
      ri = atoms[ii].rr
      
      # place atom in box
      rx = ri[0] - grid.lx*m.floor(ri[0]/grid.lx)
      ry = ri[1] - grid.ly*m.floor(ri[1]/grid.ly)

      if rx >= grid.lx: rx = 0.0
      if ry >= grid.ly: ry = 0.0

      cx = int(rx/grid.celldx)
      cy = int(ry/grid.celldy)

      cid = cx + cy*grid.ncellx
      #print("CID: ", cid, cx, cy)
      grid.cells[cid].atoms.append(atoms[ii])

   bonds = []

   natom = len(atoms)
   bid = 1

   for rc in rc_list:
      rmin2 = pow(rc - drc, 2)
      rmax2 = pow(rc + drc, 2)
      for icell in grid.cells:
         for iatom in icell.atoms:
            ri = iatom.rr
            for jcell in icell.neigh:
               for jatom in jcell.atoms:
                  #if iatom == jatom: continue
                  if icell.id == jcell.id and iatom.aid >= jatom.aid: continue
                  rj = jatom.rr

                  rij = rj - ri
                  rij = MinImag(rij, lx, ly, lz, xy, xz, yz, periodicity)
                  rij2 = np.dot(rij, rij)
                  if rmin2 < rij2 < rmax2:
                     rlen = np.sqrt(rij2)
                     bonds.append(Bond(bid, iatom, jatom, rlen))
                     iatom.neigh.append(jatom)
                     jatom.neigh.append(iatom)
                     bid += 1
   return bonds
