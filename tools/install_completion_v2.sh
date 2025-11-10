# in the activated env
conda install -c conda-forge argcomplete 2>/dev/null || pip install argcomplete

mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
printf '%s\n' 'eval "$('"$CONDA_PREFIX"'/bin/register-python-argcomplete sgl)"' \
  >> "$CONDA_PREFIX/etc/conda/activate.d/argcomplete.sh"

conda deactivate && conda activate "$(basename "$CONDA_PREFIX")"

