#!/bin/bash
sgl-pwi2json i.01.in --pretty -o o.02.json
sgl-json2xyz o.02.json -o o.03.xyz
sgl-xyz2qe --header i.01.header --xyz o.03.xyz -o o.04.in
cp o.04.in ../gold/.
