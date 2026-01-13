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

class AtomType:
   def __init__(self, type, mass, name):
      self.type = type
      self.mass = mass
      self.name = name

class Atom:
   def __init__(self, aid, mol, type, qq, rr, name):
      self.aid = aid
      self.molid = mol
      self.type = type
      self.qq = qq
      self.rr = rr
      self.neigh = []
      self.name = name

class BondType:
   def __init__(self, type, types):
      self.type = type
      self.types = types

class Bond:
   def __init__(self, bid, iatom, jatom, rr):
      self.bid = bid
      self.iatom = iatom
      self.jatom = jatom
      self.len = rr
      self.type = -1
      self.type_str = ''

class AngleType:
   def __init__(self, type, types, sym):
      self.type = type
      self.types = types
      self.sym = sym

class Angle:
   def __init__(self, bid, iatom, jatom, katom):
      self.bid = bid
      self.iatom = iatom
      self.jatom = jatom
      self.katom = katom
      self.theta = None
      self.sym = ''
      self.type = -1
      self.type_str = ''

class DihedType:
   def __init__(self, type, types, orient):
      self.type = type
      self.types = types
      self.orient = orient

class Dihed:
   def __init__(self, bid, iatom, jatom, katom, latom):
      self.bid = bid
      self.iatom = iatom
      self.jatom = jatom
      self.katom = katom
      self.latom = latom
      self.orient = ''
      self.type = -1
      self.type_str = ''
