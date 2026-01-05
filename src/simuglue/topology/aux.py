import sys
import os
import math as m
import numpy as np
import argparse
import copy as cp

def SortTypes(chem_list):
   aux_list = [str(chem) for chem in chem_list]

   # bond, angle chems
   if aux_list[0] > aux_list[-1]:
      aux_list.reverse()

   # dihedral chems
   if len(aux_list) == 4 and (aux_list[0] == aux_list[3] and aux_list[1] > aux_list[2]):
      aux_list.reverse()

   return aux_list
