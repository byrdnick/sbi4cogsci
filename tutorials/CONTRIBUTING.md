# Adding your tutorial

The repo imposes one small, tool-agnostic contract. Everything else is an
optional layer — adopt any, none, or all. Use your own stack.

## The contract (every tutorial)

1. One folder: `tutorials/day<day>_<short_name>/` (e.g. `day3_sbi_tools/`).
   A folder may hold several sessions from the same day.
2. A `README.md` in it (copy `_template/README.md`) stating: title,
   instructor, day/session, requirements, and EXACTLY ONE run command —
   or the sentence "Materials only, not runnable."
3. One row in `tutorials/index.qmd` per **rendered page** — so a day folder
   with three site pages gets three rows (see the column notes at the top of
   that file).
4. Nothing in your folder may need to execute when the site builds — the
   Pages CI has no Python/R/MATLAB. Site builds never run code
   (`tutorials/_metadata.yml` sets `execute: enabled: false`; do not
   override it).
5. No large files in git (> a few MB). Fetch data at runtime — e.g.
   `huggingface_hub.hf_hub_download(..., revision=<pinned>)` from a dataset
   repo — into `data/` inside your folder (gitignored).
6. Merges to `main` go live immediately. Before your PR: `quarto render`
   locally, or ask a maintainer to check the preview.

## Optional layers (pick freely)

- **Site page**: commit a `.ipynb` WITH stored outputs (or a `.qmd` whose
  code is display-only). Quarto renders it into the website without executing
  it; link it from your index row. No artifact → your row links to GitHub
  only.
- **Python env** (recommended default if you're happy to adopt it: uv):
  (a) a [PEP 723](https://peps.python.org/pep-0723/) header in your
  script/notebook — runs via `uv run --script` / `uvx marimo edit --sandbox`;
  or (b) a tutorial-local `pyproject.toml` + `uv.lock` (`uv sync` inside your
  folder); or (c) the shared BayesFlow/HSSM/JAX env at
  `tutorials/pyproject.toml` (`cd tutorials && uv sync`, documented in
  `tutorials/README.md`) — join it if your stack already fits, but don't add
  heavy single-tutorial deps to it. Need a different shared stack? Ask before
  creating a second one.
- **marimo**: `.py` notebook as source of truth + baked `.ipynb` export as
  the site page. Working example: `toy_example/`. Name the export
  `<name>_page.ipynb`, NOT `<name>.ipynb` — Quarto silently skips a notebook
  whose stem matches a sibling `.py` file.
- **Jupyter**: commit the `.ipynb` with outputs; optionally add a Colab
  badge; name your environment in the README.
- **Non-Python / not runnable**: a README-only folder is fully valid
  (MATLAB, R, slides-only).

## Don'ts

- No `execute: enabled: true` (or `--execute` assumptions) in any file.
- No `.Rmd` (needs R at render time) — for R, use README-only or a
  notebook-with-outputs.
- Only `*.ipynb` and `*.qmd` at the top level of your folder become site
  pages (`_quarto.yml` render globs); `README.md` and other `.md` files stay
  repo-only. Don't commit draft `.ipynb`/`.qmd` you don't want published.
- **Solutions and drafts: prefix the filename with `_`.** Quarto skips
  `_`-prefixed files, so `_lasenet_tutorial_solution.ipynb` stays in the repo
  but never becomes a page. This is the ONLY thing standing between an answer
  key and a public URL — the repo is public, so a `_` file is still readable
  on GitHub. If students must not see it at all before the session, keep it
  off `main` until afterwards.
- Don't commit `.venv/`, `data/`, `__marimo__/` (gitignored).
- **Trained networks are the exception** — commit them. A notebook that trains
  a network at run time is unusable in a session, so the small ones are checked
  in and loaded by default: `day3_sbi_tools/checkpoints/ddm_nre.keras` (1.7 MB)
  and `day4_complex_joint/checkpoints/compositional.keras` (3.0 MB). Nothing
  gitignores `checkpoints/`, so `git add` them deliberately and keep them under
  the size limit in rule 5.
