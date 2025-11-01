#!/usr/bin/env python3
from dataclasses import dataclass
from types import MappingProxyType
from enum import Enum

ANGSTROM_TO_NM: float = 0.1
ANGSTROM_TO_METER: float = 1e-10
BOHR_TO_ANGSTROM: float = 0.52917721067121

RY_TO_EV: float = 13.6057039763
EV_TO_JOULE: float = 1.602176634E-19

ATM_TO_PA: float = 101325
PA_TO_ATM: float = 1.0 / 101325 

RY_TO_JOULE: float = RY_TO_EV * EV_TO_JOULE
RY_BOHR3_TO_ATM: float = RY_TO_JOULE / (BOHR_TO_ANGSTROM * ANGSTROM_TO_METER)**3 * PA_TO_ATM
