#!/bin/bash
sgl wf cij run --config cij_2d.yaml
sgl wf cij post --config cij_2d.yaml
sgl wf cij run --config cij_3d.yaml
sgl wf cij post --config cij_3d.yaml
