#!/bin/bash
sgl-toqe --header header.in --xyz single.xyz  --prefix pref_single
cp pref_single.in ../gold/.
