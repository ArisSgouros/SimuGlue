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
from simuglue.crystalbuilder.aux import SortTypes

def ExportLammpsDataFile(filename, box_lmp, atom_types, bond_types, angle_types, dihed_types, atoms, bonds, angles, diheds):
   xlo = box_lmp['xlo']
   xhi = box_lmp['xhi']
   ylo = box_lmp['ylo']
   yhi = box_lmp['yhi']
   zlo = box_lmp['zlo']
   zhi = box_lmp['zhi']
   xy  = box_lmp['xy']
   xz  = box_lmp['xz']
   yz  = box_lmp['yz']
   is_triclinic = (xy != 0.0 or xz != 0.0 or yz != 0.0)
   is_angle = len(angle_types) > 0
   is_dihed = len(dihed_types) > 0

   f = open(filename, 'w')

   f.write('# LAMMPS data file\n')
   f.write('\n')
   f.write('%d atoms\n' % len(atoms))
   f.write('%d bonds\n' % len(bonds))
   if is_angle:
      f.write('%d angles\n' % len(angles))
   if is_dihed:
      f.write('%d dihedrals\n' % len(diheds))
   f.write('\n')
   f.write('%d atom types\n' % len(atom_types))
   f.write('%d bond types\n' % len(bond_types))
   if is_angle:
      f.write('%d angle types\n' % len(angle_types))
   if is_dihed:
      f.write('%d dihedral types\n' % len(dihed_types))
   f.write('\n')
   f.write('%-16.9f %-16.9f xlo xhi\n' % (xlo, xhi))
   f.write('%-16.9f %-16.9f ylo yhi\n' % (ylo, yhi))
   f.write('%-16.9f %-16.9f zlo zhi\n' % (zlo, zhi))
   if is_triclinic:
      f.write('%-16.9f %-16.9f %-16.9f xy xz yz\n' % (xy, xz, yz))
   f.write('\n')
   f.write('Masses\n')
   f.write('\n')
   for atom_type in atom_types.values():
      f.write("%d %-16.9f # %s\n" % (atom_type.type, atom_type.mass, atom_type.name))
   f.write('\n')
   f.write('Atoms\n')
   f.write('\n')
   for iat in atoms:
      f.write("%d %d %d %-16.9f %-16.9f %-16.9f %-16.9f # %s\n" % ( iat.aid, iat.molid, iat.type, iat.qq, iat.rr[0], iat.rr[1], iat.rr[2], atom_types[iat.type].name ))
   if bonds:
      f.write('\n')
      f.write('Bonds\n')
      f.write('\n')
      for bond in bonds:
         f.write("%d %d %d %d # %s\n" % (bond.bid, bond.type, bond.iatom.aid, bond.jatom.aid, bond.type_str))
   if is_angle:
      f.write('\n')
      f.write('Angles\n')
      f.write('\n')
      for angle in angles:
         f.write("%d %d %d %d %d # %s\n" % (angle.bid, angle.type, angle.iatom.aid, angle.jatom.aid, angle.katom.aid, angle.type_str ))
   if is_dihed:
      f.write('\n')
      f.write('Dihedrals\n')
      f.write('\n')
      for dihed in diheds:
         f.write("%d %d %d %d %d %d # %s\n" % (dihed.bid, dihed.type, dihed.iatom.aid, dihed.jatom.aid, dihed.katom.aid, dihed.latom.aid, dihed.type_str ))
   f.close()

def ExportLammpsDumpFile(filename, box_lmp, atoms):
   xlo = box_lmp['xlo']
   xhi = box_lmp['xhi']
   ylo = box_lmp['ylo']
   yhi = box_lmp['yhi']
   zlo = box_lmp['zlo']
   zhi = box_lmp['zhi']
   xy  = box_lmp['xy']
   xz  = box_lmp['xz']
   yz  = box_lmp['yz']
   is_triclinic = (xy != 0.0 or xz != 0.0 or yz != 0.0)

   fout = open(filename, 'w')
   fout.write("ITEM: TIMESTEP\n")
   fout.write("0\n")
   fout.write("ITEM: NUMBER OF ATOMS\n")
   fout.write("%-8d\n" % len(atoms))

   if is_triclinic:
      xlo_bound = xlo + min(0.0,xy,xz,xy+xz)
      xhi_bound = xhi + max(0.0,xy,xz,xy+xz)
      ylo_bound = ylo + min(0.0,yz)
      yhi_bound = yhi + max(0.0,yz)
      zlo_bound = zlo
      zhi_bound = zhi
      fout.write("ITEM: BOX BOUNDS xy xz yz\n")
      fout.write("%-16.9f %-16.9f %-16.9f\n" % (xlo_bound, xhi_bound, xy) )
      fout.write("%-16.9f %-16.9f %-16.9f\n" % (ylo_bound, yhi_bound, xz) )
      fout.write("%-16.9f %-16.9f %-16.9f\n" % (zlo_bound, zhi_bound, yz) )
   else:
      fout.write("ITEM: BOX BOUNDS pp pp pp\n")
      fout.write("%-16.9f %-16.9f\n" % (xlo, xhi) )
      fout.write("%-16.9f %-16.9f\n" % (ylo, yhi) )
      fout.write("%-16.9f %-16.9f\n" % (zlo, zhi) )
   fout.write("ITEM: ATOMS id mol type xu yu zu\n")

   for iat in atoms:
      fout.write("%-d %-d %-d %-f %-f %-f\n" % (iat.aid, iat.molid, iat.type, iat.rr[0], iat.rr[1], iat.rr[2]))

   fout.close()

def ExportXyzFile(filename, box_lmp, atoms):
   xlo = box_lmp['xlo']
   xhi = box_lmp['xhi']
   ylo = box_lmp['ylo']
   yhi = box_lmp['yhi']
   zlo = box_lmp['zlo']
   zhi = box_lmp['zhi']
   xy  = box_lmp['xy']
   xz  = box_lmp['xz']
   yz  = box_lmp['yz']

   fout = open(filename, 'w')
   fout.write("%-8d\n" % len(atoms))
   fout.write("%-16.9f %-16.9f %-16.9f %-16.9f %-16.9f %-16.9f %-16.9f %-16.9f %-16.9f \n" % (xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz) )
   for iat in atoms:
      fout.write("%-d %-f %-f %-f\n" % (iat.type, iat.rr[0], iat.rr[1], iat.rr[2]))
   fout.close()

def ExportTypes(filename, atom_types, bond_types, angle_types, dihed_types):
   fout = open(filename, 'w')

   fout.write('\n')
   fout.write('\n')

   # Export number of types
   if len(atom_types) > 0:  fout.write('%d atom types\n' % len(atom_types))
   if len(bond_types) > 0:  fout.write('%d bond types\n' % len(bond_types))
   if len(angle_types) > 0: fout.write('%d angle types\n' % len(angle_types))
   if len(dihed_types) > 0: fout.write('%d dihedral types\n' % len(dihed_types))

   # Export types of components
   fout.write('\nAtom Types\n\n')
   for key in atom_types:
     obj = atom_types[key]
     fout.write("%d %s\n" %(obj.type, obj.name))

   if bond_types:
     fout.write('\nBond Types\n\n')
     for key in bond_types:
       obj = bond_types[key]
       fout.write("%d %s\n" %(obj.type, key))

   if angle_types:
     fout.write('\nAngle Types\n\n')
     for key in angle_types:
       obj = angle_types[key]
       fout.write("%d %s\n" %(obj.type, key))

   if dihed_types:
     fout.write('\nDihedral Types\n\n')
     for key in dihed_types:
       obj = dihed_types[key]
       fout.write("%d %s\n" %(obj.type, key))

   fout.close()
   return
