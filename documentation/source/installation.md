# Installation and documentation build

## Install M4E 3D

From the repository root, install the package in editable mode:

```console
python -m pip install -e .
```

This installs the core dependencies. To run the numerical ODE integration,
install the optional JAX and diffrax dependencies:

```console
python -m pip install "jax" "diffrax>=0.7"
```

The standard JAX installation targets CPU. For GPU or accelerator-specific
setups, follow the current
[JAX installation instructions](https://docs.jax.dev/en/latest/installation.html).
The symbolic model and kinematics can be inspected without running a simulation;
integration requires JAX and diffrax.

## Build the HTML documentation

Install the documentation dependencies:

```console
python -m pip install -e ".[docs]"
```

From the repository root, build the HTML site directly with Sphinx:

```console
python -m sphinx -n -W --keep-going documentation/source documentation/build/html
```

Or use the Sphinx make target from the documentation directory:

```console
cd documentation
make html
```

On Windows, if GNU Make is not installed, use the included batch helper instead:

```powershell
cd documentation
.\make.bat html
```

Activate the Python environment where `.[docs]` was installed before running
either helper, so `sphinx-build` is available on `PATH`. The `make html` form
also requires GNU Make to be installed.

Open `documentation/build/html/index.html` in a browser. `-n -W` treats
unresolved references and warnings as errors, which helps keep the published
docs complete.

The API pages use Sphinx autodoc/autosummary and may mock the optional numerical
backend imports. Building the docs does not run the examples or validate the
numerical solver.
