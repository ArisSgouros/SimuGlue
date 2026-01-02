#!/usr/bin/env python3

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
from simuglue.crystalbuilder.export import ExportLammpsDataFile, ExportLammpsDumpFile, ExportXyzFile, ExportTypes
from simuglue.crystalbuilder.read import ReadBasis
from simuglue.crystalbuilder.net_types import Atom, Bond, AtomType, BondType, AngleType, DihedType
from simuglue.crystalbuilder.calculate_bond import CalculateBonds
from simuglue.crystalbuilder.calculate_bond_gridxy import CalculateBondsGridXy
from simuglue.crystalbuilder.calculate_angle import CalculateAngles, DifferentiateAngleSymmetry, DifferentiateAngleTheta
from simuglue.crystalbuilder.calculate_dihed import CalculateDiheds, DifferentiateCisTrans

# constants
kDim = 3
kTol = 1.e-8

def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Generate crystal structure')
    parser.add_argument('path_basis', type=str, help='Path of basis file (Lammps datafile)')
    parser.add_argument('cells', type=str, help='delimited list of replications')
    parser.add_argument('-periodic', '--periodic', type=str, default="1,1,1", help='delimited list of periodicity directions')
    parser.add_argument('-rc', '--rc', type=str, default="", help='list of cutoff radii of bonded interactions')
    parser.add_argument('-drc', '--drc', type=float, default=0.1, help='tolerance of bonded interactions')
    parser.add_argument('-file_types', '--file_types', type=str, default="", help='Export table with atom/bond/angle/.. types')
    parser.add_argument('-file_pos', '--file_pos', type=str, default='pos.dat', help='Name of the Lammps data file')
    parser.add_argument('-file_dump', '--file_dump', type=str, default='dump.lammpstrj', help='Name of the Lammps dump file')
    parser.add_argument('-file_xyz', '--file_xyz', type=str, default='dump.xyz', help='Name of the xyz file')
    parser.add_argument('-bond', '--bond', type=int, default='0', help='Calculate bonds')
    parser.add_argument('-diff_bond_len', '--diff_bond_len', type=int, default='0', help='Differentiate bond types based on length')
    parser.add_argument('-diff_bond_fmt', '--diff_bond_fmt', type=str, default='%.2f', help='Set the fmt of bond lengths')
    parser.add_argument('-angle', '--angle', type=int, default='0', help='Calculate angles')
    parser.add_argument('-angle_symmetry', '--angle_symmetry', type=int, default='0', help='Differentiate coplanar/vertical angles')
    parser.add_argument('-diff_angle_theta', '--diff_angle_theta', type=int, default='0', help='Differentiate angle types based on theta')
    parser.add_argument('-diff_angle_theta_fmt', '--diff_angle_theta_fmt', type=str, default='%.2f', help='Set the fmt of angles')
    parser.add_argument('-dihed', '--dihed', type=int, default='0', help='Calculate dihedrals')
    parser.add_argument('-diff_dihed_theta', '--diff_dihed_theta', type=int, default='0', help='Differentiate dihed types based on theta')
    parser.add_argument('-diff_dihed_theta_abs', '--diff_dihed_theta_abs', type=int, default='1', help='Absolute dihedral angle')
    parser.add_argument('-diff_dihed_theta_fmt', '--diff_dihed_theta_fmt', type=str, default='%.2f', help='Set the fmt of diheds')
    parser.add_argument('-cis_trans', '--cis_trans', type=int, default='0', help='Differentiate cis/trans dihedrals')
    parser.add_argument('-grid', '--grid', type=str, default='none', help='Enable grid for neighbor lists')
    parser.add_argument('-type_delimeter', '--type_delimeter', type=str, default=" ", help='String which joins different types')
    parser.add_argument('-molid_inc_z', '--molid_inc_z', type=int, default=0, help='increase the molids for each z-axis duplication')
    return parser

def main(argv=None, prog: str | None = None) -> int:
   parser = build_parser(prog=prog)
   args = parser.parse_args(argv)

   path_basis = args.path_basis
   cells = [int(item) for item in args.cells.split(',')]
   periodicity = [int(item) for item in args.periodic.split(',')]
   rc_list = [float(item) for item in args.rc.split(',')]
   drc = args.drc
   calc_bond = args.bond
   diff_bond_len = args.diff_bond_len
   diff_bond_fmt = args.diff_bond_fmt
   calc_angle = args.angle
   diff_angle_theta = args.diff_angle_theta
   diff_angle_theta_fmt = args.diff_angle_theta_fmt
   diff_dihed_theta = args.diff_dihed_theta
   diff_dihed_theta_abs = args.diff_dihed_theta_abs
   diff_dihed_theta_fmt = args.diff_dihed_theta_fmt
   calc_angle_symmetry = args.angle_symmetry
   calc_dihed = args.dihed
   calc_cis_trans = args.cis_trans
   grid_type = args.grid
   type_delimeter = args.type_delimeter
   molid_inc_z = args.molid_inc_z

   if grid_type != 'none':
     print("Warning: grid does not work with triclinic cells")

   file_pos = args.file_pos
   file_dump = args.file_dump
   file_xyz  = args.file_xyz
   file_types = args.file_types

   print('path        : ', path_basis)
   print('cells       : ', cells)
   print('periodicity : ', periodicity)
   print('calc_bond   : ', calc_bond)
   print('rc(s)       : ', rc_list)
   print('drc         : ', drc)

   print("Reading the datafile:", path_basis)
   basis_vectors, basis_orig, basis_atoms, atom_types = ReadBasis(path_basis)
   print()

   print("basis origin:")
   print('   ', basis_orig)
   print("basis vectors:")
   for basis_vector in basis_vectors:
      print('   ', basis_vector)
   basis_volume = np.linalg.norm(np.dot(np.cross(basis_vectors[0], basis_vectors[1]), basis_vectors[2]))
   print("basis volume:")
   print('   ', basis_volume)
   print("basis density:")
   print('   ', len(basis_atoms) / basis_volume)
   print("atom types:")
   for atom_type in atom_types.values():
      print('   ',atom_type.type, atom_type.mass, atom_type.name)
   print("basis atoms:")
   print("   %-8s %-8s %-16s %-16s %-16s %-16s" % ("id", "type", "charge", "x", "y", "z"))
   for atom in basis_atoms:
      print("   %-8d %-8d %-16.9f %-16.9f %-16.9f %-16.9f" % (atom.aid, atom.type, atom.qq, atom.rr[0], atom.rr[1], atom.rr[2]))
   print("box vectors:")
   box_vectors = []
   box_orig = []
   for ii in range(kDim):
      box_vectors.append(basis_vectors[ii]*cells[ii])
      box_orig.append(basis_orig[ii]*cells[ii])
      print('   ', box_vectors[ii])
   print("box vectors (lammps constraints):")
   box_lmp = {'xlo':box_orig[0], 'xhi':box_orig[0]+box_vectors[0][0], \
              'ylo':box_orig[1], 'yhi':box_orig[1]+box_vectors[1][1], \
              'zlo':box_orig[2], 'zhi':box_orig[2]+box_vectors[2][2], \
              'xy':box_vectors[1][0], 'xz':box_vectors[2][0], 'yz':box_vectors[2][1]}
   for key in box_lmp:
      print("   %-3s : %-16.9f" % (key, box_lmp[key]))

   print()
   print("Generating the atomic coordinates..")
   atoms = []
   aid = 1
   imol = 1
   for iz in range(cells[2]):
      for iy in range(cells[1]):
         for ix in range(cells[0]):
            for atom in basis_atoms:
               rr = atom.rr + basis_vectors[0]*ix + basis_vectors[1]*iy +basis_vectors[2]*iz
               imol = atom.molid + iz*molid_inc_z
               atoms.append(Atom(aid, imol, atom.type, atom.qq, rr, atom.name))
               aid+=1

   bonds = []
   bond_types = {}
   if calc_bond:
      print("Generating bonds..")
      if grid_type == 'none':
        bonds = CalculateBonds(atoms, box_lmp, periodicity, rc_list, drc)
      elif grid_type == 'xy':
        bonds = CalculateBondsGridXy(atoms, box_lmp, periodicity, rc_list, drc)
      print("  %d" % (len(bonds)))

      print("Generating bond types..")
      itype = 1
      for bond in bonds:
         type_sort = SortTypes([bond.iatom.name, bond.jatom.name])
         type_str = type_delimeter.join(ii for ii in type_sort)
         if diff_bond_len:
            fmt_tmp = str(diff_bond_fmt %(bond.len))
            type_str = "%s%s%s" %(type_str, type_delimeter, fmt_tmp)
         if not bond_types.get(type_str):
            bond_types[type_str] = BondType(itype, type_sort)
            itype += 1
         bond.type = bond_types[type_str].type
         bond.type_str = type_str
      print("  %d" % (len(bond_types)))

   angles = []
   angle_types = {}
   if calc_angle:
      print("Generating angles..")
      angles = CalculateAngles(atoms)
      print("  %d" % (len(angles)))

      if calc_angle_symmetry:
         DifferentiateAngleSymmetry(angles)

      if diff_angle_theta:
         DifferentiateAngleTheta(angles, box_lmp, periodicity)

      print("Generating angle types..")
      itype = 1
      for angle in angles:
         type_sort = SortTypes([angle.iatom.name, angle.jatom.name, angle.katom.name])
         type_str = type_delimeter.join(ii for ii in type_sort)
         if calc_angle_symmetry:
            type_str = "%s%s%s" %(type_str, type_delimeter, angle.sym)
         if diff_angle_theta:
            fmt_tmp = str(diff_angle_theta_fmt %(angle.theta))
            type_str = "%s%s%s" %(type_str, type_delimeter, fmt_tmp)
         if not angle_types.get(type_str):
            angle_types[type_str] = AngleType(itype, type_sort, angle.sym)
            itype += 1
         angle.type = angle_types[type_str].type
         angle.type_str = type_str
      print("  %d" % (len(angle_types)))

   diheds = []
   dihed_types = {}
   if calc_dihed:
      print("Generating diheds..")
      diheds = CalculateDiheds(atoms)
      print("  %d" % (len(diheds)))

      if calc_cis_trans or diff_dihed_theta:
         DifferentiateCisTrans(diheds, box_lmp, periodicity)

      print("Generating dihed types..")
      itype = 1
      for dihed in diheds:
         if dihed.orient == 'cis':
            continue
         type_sort = SortTypes([dihed.iatom.name, dihed.jatom.name, dihed.katom.name, dihed.latom.name])
         type_str = type_delimeter.join(ii for ii in type_sort)
         if calc_cis_trans:
            type_str = "%s%s%s" %(type_str, type_delimeter, dihed.orient)
         if diff_dihed_theta:
            phi_deg = m.degrees(dihed.phi)
            if diff_dihed_theta_abs:
               phi_deg = abs(phi_deg)
            fmt_tmp = str(diff_dihed_theta_fmt %(phi_deg))
            type_str = "%s%s%s" %(type_str, type_delimeter, fmt_tmp)
         if not dihed_types.get(type_str):
            print("Add dihed %s" %(type_str))
            dihed_types[type_str] = DihedType(itype, type_sort, dihed.orient)
            itype += 1
         dihed.type = dihed_types[type_str].type
         dihed.type_str = type_str
      for dihed in diheds:
         type_sort = SortTypes([dihed.iatom.name, dihed.jatom.name, dihed.katom.name, dihed.latom.name])
         type_str = type_delimeter.join(ii for ii in type_sort)
         if calc_cis_trans:
            type_str = "%s%s%s" %(type_str, type_delimeter, dihed.orient)
         if diff_dihed_theta:
            phi_deg = m.degrees(dihed.phi)
            if diff_dihed_theta_abs:
               phi_deg = abs(phi_deg)
            fmt_tmp = str(diff_dihed_theta_fmt %(phi_deg))
            type_str = "%s%s%s" %(type_str, type_delimeter, fmt_tmp)
         if not dihed_types.get(type_str):
            print("Add dihed %s" %(type_str))
            dihed_types[type_str] = DihedType(itype, type_sort, dihed.orient)
            itype += 1
         dihed.type = dihed_types[type_str].type
         dihed.type_str = type_str
      print("  %d" % (len(dihed_types)))

   print()
   print("Generating a lammps datafile with name " + file_pos + " ..")
   ExportLammpsDataFile(file_pos, box_lmp, atom_types, bond_types, angle_types, dihed_types, atoms, bonds, angles, diheds)
   if file_dump != '':
     print()
     print("Generating a lammps dump file with name " + file_dump + " ..")
     ExportLammpsDumpFile(file_dump, box_lmp, atoms)
   if file_xyz != '':
     print()
     print("Generating a XYZ dump file with name " + file_xyz + " ..")
     ExportXyzFile(file_xyz, box_lmp, atoms)

   if file_types != "":
     print()
     print("Exporting interaction table " + file_types + " ..")
     ExportTypes(file_types, atom_types, bond_types, angle_types, dihed_types)
