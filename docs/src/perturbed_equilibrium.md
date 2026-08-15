# Perturbed Equilibrium

The `PerturbedEquilibrium` module computes the plasma response to external magnetic perturbations.

## Types

```@docs
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.PerturbedEquilibriumControl
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.PerturbedEquilibriumInternal
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.PerturbedEquilibriumState
```

## Functions

```@docs
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.compute_perturbed_equilibrium
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.write_outputs_to_HDF5
```

## Plotting per-surface results against ψ or q

`SingularCoupling/` quantities are indexed by rational-surface **index**, not by q: with
multi-n runs a single q value can host several resonances, so the index is the only
unambiguous axis. Both `rational_psi` and `rational_q` are attached to that axis as HDF5
dimension scales, so plotting against either is direct:

```julia
h5open("gpec.h5", "r") do f
    g = f["PerturbedEquilibrium/SingularCoupling"]
    q = read(g["rational_q"])
    b_res = abs.(read(g["resonant_area_weighted_field"]))
    scatter(q, b_res; xlabel="q", ylabel="|b^r| [T]")   # or read(g["rational_psi"]) for ψ_N
end
```

In Python the same scales are visible through `h5py`'s dimension API
(`dset.dims[0]["psi_rational"]`, `dset.dims[0]["q_rational"]`), so xarray-style tooling can
label the axis automatically.
