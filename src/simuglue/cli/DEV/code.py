import numpy as np
import ase
from ase.io import espresso, write
path_in = "i.01.in"
path_out = "o.01.out"
atoms = espresso.read_espresso_in(path_in)

write(path_out, atoms, format='extxyz')
