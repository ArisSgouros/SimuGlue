### Installation in conda environment
conda create -n simuglue python=3.11
conda activate simuglue

pip install -e ".[dev]"

### pytest notes
pytest --keep-failed
pytest --vv
pytest -v -s # do not suppress prints
pytest -v -k "module name"
pytest --basetemp=./_pytest_tmp
