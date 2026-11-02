# Setup

GPEC is a Julia package.
Installing it means installing Julia, cloning the repository and letting Julia's package manager fetch the dependencies; the same steps apply on Linux, macOS and Windows (WSL).

## Install

1. **Julia 1.11 or later.** The simplest route on every platform is [juliaup](https://github.com/JuliaLang/juliaup), which also keeps Julia updated:
   - *Linux*: `curl -fsSL https://install.julialang.org | sh`, or unpack a tarball from [julialang.org/downloads](https://julialang.org/downloads/) and add its `bin/` to `PATH`. On a cluster, check `module avail julia` first.
   - *macOS*: `brew install juliaup` (or `brew install julia`), or the `.dmg` from the downloads page. Install the Xcode command-line tools (`xcode-select --install`) for `git`.
   - *Windows*: install WSL with an Ubuntu distribution (`wsl --install` in PowerShell), then follow the Linux steps inside it. Keep the repository in the WSL file system, not under `/mnt/c`.

   Check with `julia --version`.

2. **Clone the repository.**
   ```bash
   git clone https://github.com/OpenFUSIONToolkit/GPEC.git
   cd GPEC
   ```

3. **Install the dependencies** into the project environment and check that the package loads:
   ```bash
   julia --project=. -e 'using Pkg; Pkg.instantiate()'
   julia --project=. -e 'using GeneralizedPerturbedEquilibrium'
   ```
   The first call downloads and precompiles every dependency, which takes several minutes; the second builds GPEC itself, which also runs a small precompile workload.

Every `julia` command on GPEC needs `--project=.` (or `--project=/path/to/GPEC`): without it Julia uses the default environment and cannot find the package.
Never run Julia with `sudo`.

## First run

Run one of the shipped examples; the deck, the equilibrium file and the coil geometry are all in the directory:

```bash
./gpec examples/DIIID-like_ideal_example
```

The first run compiles the code it needs and takes a few minutes; later runs are faster.
The run logs each stage and writes `examples/DIIID-like_ideal_example/gpec.h5`.
A quick look at three results, from the shell or from Julia:

```bash
h5ls -r examples/DIIID-like_ideal_example/gpec.h5 | less
```

```julia
using HDF5
h5open("examples/DIIID-like_ideal_example/gpec.h5") do f
    read(f, "Equilibrium/q_95")
    read(f, "ForceFreeStates/FreeBoundaryStability/eigenmode_energies")  # positive first entry: ideal-kink stable
    read(f, "PerturbedEquilibrium/SingularCoupling/resonant_area_weighted_field")  # resonant field per rational surface
end
```

The [Workflow](workflow.md) page explains what each stage computed and which datasets to read next, and `examples/DIIID-like_ideal_example/analyze_example.jl` plots the output with the [Analysis](analysis.md) module.

## Running GPEC

The `gpec` script in the repository root runs the pipeline on a directory holding a `gpec.toml` deck (the current directory when no path is given) and writes the `[Run] output_file` next to it:

```bash
./gpec path/to/directory
```

The same pipeline is available from Julia, either whole or stage by stage:

```julia
using GeneralizedPerturbedEquilibrium

# The full pipeline of the deck, returning every stage result
results = GeneralizedPerturbedEquilibrium.main(["path/to/directory"])

# One stage: the equilibrium of a g-file
using GeneralizedPerturbedEquilibrium.Equilibrium
eq = PlasmaEquilibrium("path/to/g-file"; eq_type="efit")
```

A stage runs only when its deck section is present; the section index on the [Workflow](workflow.md) page lists them, and the [Scripting API](api.md) page shows the `solve` calls behind each stage.

## Editor

[VS Code](https://code.visualstudio.com/) with the Julia extension is a convenient environment: open the `GPEC` folder, and the integrated terminal runs `julia --project=.` or `./gpec` as above.
On Windows, use the *Remote - WSL* extension and open the folder from inside WSL.

## Developing

Pre-commit hooks, the precompile workload and Revise are described under *Developer tips* on the [Developer Notes](developer_notes.md) page.
