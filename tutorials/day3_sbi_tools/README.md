# Day 3: Tools for SBI

Materials for [Day 3 — Tools for SBI](https://stefanradev93.github.io/sbi4cogsci/schedule.html#day-3-wednesday-july-29-tools-for-sbi).

This folder holds several sessions from the same day. Each is listed below with
its own instructor, stack, and run command.

---

## HSSM (09:30)

- **Instructor:** Alexander Fengler
- **Stack:** Python 3.12 · the shared `tutorials/` uv environment
- **Run:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/stefanradev93/sbi4cogsci/blob/main/tutorials/day3_sbi_tools/hssm-intro.ipynb) &nbsp; or locally, `cd tutorials && uv sync` then open `hssm-intro.ipynb`

The shortest working HSSM model, the two defaults that fail silently
(`[-1, +1]` response coding and a `p_outlier=0.05` lapse), the SSM-specific fit
checks, and — in section 7 — how to **learn** a likelihood from nothing but a
simulator and hand it back to HSSM.

That last section takes a DDM out of `ssm-simulators`, trains a BayesFlow
`RatioApproximator` (NRE) on it, registers the result with
`hssm.register_model`, and fits it beside HSSM's exact analytic likelihood so
the two posteriors can be compared directly. The trained network is committed
as `checkpoints/ddm_nre.keras` (1.7 MB) and loaded by default; set
`FORCE_TRAIN = True` to retrain, which takes about 25 minutes — note that
retraining **overwrites the committed file**, so `git checkout` it afterwards
unless you meant to replace it. On Colab the bootstrap cell downloads the same
network, so section 7 loads rather than trains there too.

## BayesFlow — amortized inference for the DMC (10:00)

- **Instructors:** Stefan T. Radev
- **Stack:** Python 3.12 · the shared `tutorials/` uv environment
- **Run:** `cd tutorials && uv sync`, then open `dmc-bayesflow.ipynb`

## MCMC for hierarchical Bayesian models (11:00)

- **Instructors:** Alexander Fengler, Brandon Turner
- **Stack:** Python 3.12 · the shared `tutorials/` uv environment
- **Run:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/stefanradev93/sbi4cogsci/blob/main/tutorials/day3_sbi_tools/hierarchical-mcmc.ipynb) &nbsp; or locally, `cd tutorials && uv sync` then open `hierarchical-mcmc.ipynb`

Taught live from the notebook, start to finish. Why you would want a hierarchy
at all, funnel geometry, centered vs non-centered, where the crossover between
them actually falls, and per-parameter parameterization in HSSM.

## Companion: hierarchical modelling from scratch

- **Instructors:** Brandon Turner, Alexander Fengler
- **Stack:** Python 3.12 · the shared `tutorials/` uv environment
- **Run:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/stefanradev93/sbi4cogsci/blob/main/tutorials/day3_sbi_tools/hierarchical-binomial.ipynb) &nbsp; or locally, `cd tutorials && uv sync` then open `hierarchical-binomial.ipynb`

A narrated Python translation of `binomial_Bayes_hier.R` from the Day 3 slides —
the alien-coins example. Builds the hierarchy by hand (a Metropolis-within-Gibbs
sampler in thirty lines of numpy), shows what shrinkage buys and what the
hyperprior costs, then reproduces the whole thing in six lines of PyMC. Read it
before `hierarchical-mcmc.ipynb`, which picks up where this one stops and asks
what the hierarchy does to the *geometry* of the posterior.

## Recurrent networks for dynamic data (12:00)

- **Instructor:** Ti-Fen Pan
- **Stack:** Google Colab — nothing to install
- **Run:** open the Colab link below

**LaseNet** &nbsp; [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1d9y-Svzs2z8EBRb5sY26Ni-n4p4tmd_W?usp=sharing)

---

## Requirements

Everything except the LaseNet Colab notebook runs in the shared environment
defined by `tutorials/pyproject.toml` and pinned by `tutorials/uv.lock`. Install
[uv](https://docs.astral.sh/uv/), then:

```bash
cd tutorials && uv sync
```

Select the `tutorials/.venv` kernel in your notebook editor. The first non-DDM
HSSM model downloads an ONNX likelihood network from HuggingFace, so that step
needs internet access.

**Runtime for a full top-to-bottom run** (measured, Apple silicon, CPU only):

| notebook | run "all cells" |
|---|---|
| `hssm-intro.ipynb` | ~2.5 min (loading the committed network, not training it) |
| `hierarchical-mcmc.ipynb` | ~1.5 min — dominated by the section 4 sweep (28 fits) |
| `hierarchical-binomial.ipynb` | ~1 min — a hand-written sampler plus three PyMC fits |

`hssm-intro` is dominated by sampling the DDM over all 3,988 trials and by the
posterior predictive. Note it passes `draws=100` to
`sample_posterior_predictive`: the default regenerates a response for every
posterior draw of every trial, which costs ~100 s here, while the plots consume
only 20 samples.

## Repo-only files

Files and folders beginning with `_` are deliberately kept off the published
website — Quarto skips `_`-prefixed paths. Note the repository itself is public,
so a `_` prefix hides a file from the site but not from GitHub.

- `_compare-models.ipynb` — an unfinished draft.
- `_lasenet_tutorial_solution.ipynb` — an answer key.
- `_src/` — sources for the generated notebooks. See below.

## Editing `hssm-intro`, `hierarchical-mcmc` and `hierarchical-binomial`

**Do not edit those three `.ipynb` files directly — your changes will be
overwritten.** Each is generated from a percent-format Python source in `_src/`:

```
_src/hssm-intro.py     <- edit this
hssm-intro.ipynb       <- generated; committed with outputs because the
                          Pages CI has no Python and never executes anything
```

Rebuild after editing:

```bash
cd tutorials && ./build_notebooks.sh day3_sbi_tools/_src/hssm-intro.py
```

The split exists because a `.ipynb` diff is unreviewable — a one-line change to
a plot arrives as thousands of lines of re-encoded PNG. The `.py` is the
readable history. The sources live one level down because a same-stem `.py`
sitting next to a `.ipynb` makes Quarto drop the page silently.

The other notebooks here (`dmc-bayesflow.ipynb`, `lasenet_tutorial.ipynb`) have
no such source and are edited directly.

`hierarchical-mcmc` gets its figures from `tutorials/sbi4cogsci_figures.py`,
which keeps the expensive fits and the cheap plotting in separate functions —
see that module's docstring before adding to it.

This session used to open with a revealjs deck
(`hierarchical-mcmc-slides.qmd`, plus baked PNGs under `figures/` and a
`_src/bake_slide_figures.py` to produce them). All of that was removed once the
session moved into the notebook alone; recover it from git history if needed.
