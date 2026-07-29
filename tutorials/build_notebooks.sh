#!/usr/bin/env bash
# Rebuild tutorial notebooks from their percent-format sources in `_src/`.
#
#   ./build_notebooks.sh                                         # rebuild all
#   ./build_notebooks.sh day2_bayes_toolkit/_src/pymc-intro.py   # rebuild one
#
# Layout, and why:
#
#   day2_bayes_toolkit/_src/pymc-intro.py    <- source of truth: readable, diffable
#   day2_bayes_toolkit/pymc-intro.ipynb      <- artifact: the published page
#
# The artifact is committed WITH stored outputs, because the Pages CI has no
# Python and never executes anything (tutorials/_metadata.yml sets
# `execute: enabled: false`).
#
# Source and artifact share a stem, which is only safe because they are NOT
# siblings. Quarto silently drops an .ipynb that has a same-stem .py next to it
# — it still prints "Output created" and writes no file. Putting the sources one
# level down sidesteps that, so they can keep their real names. `_src` is also
# kept off the site twice over: the render globs in _quarto.yml are one level
# deep (`tutorials/*/*.ipynb`), and Quarto ignores any path segment starting
# with an underscore.
set -euo pipefail
cd "$(dirname "$0")"

sources=("$@")
if [ ${#sources[@]} -eq 0 ]; then
  # `_src/` also holds build helpers (precompute_ddm_grid.py) that jupytext
  # must not touch, and tutorials/ holds a
  # marimo file (_molab_probe.py) it cannot read. Select on the jupytext
  # percent-format header rather than on the path, so this stays correct as
  # files are added.
  # bash 3.2 (macOS default) has no readarray.
  sources=()
  while IFS= read -r line; do sources+=("$line"); done \
    < <(grep -l 'format_name: percent' \
          $(find . -path '*/_src/*.py' -not -path './.venv/*') | sort)
fi

for src in "${sources[@]}"; do
  # The artifact goes in the day folder, i.e. the PARENT of `_src/`.
  out="$(dirname "$(dirname "$src")")/$(basename "$src" .py).ipynb"
  echo "==> $src  ->  $out"
  uv run --with jupytext --with nbconvert --with ipykernel \
    jupytext --to ipynb "$src" -o "$out" --quiet
  uv run --with jupytext --with nbconvert --with ipykernel \
    jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=1800 "$out"
  echo "    done: $(du -h "$out" | cut -f1)"
done
