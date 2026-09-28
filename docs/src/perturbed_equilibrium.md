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

## Dominant resonant-coupling mode

The singular-coupling matrix `C_resonant_area_weighted_field` maps an applied
root-area-weighted field spectrum `b̃` on the control surface to the resonant field at each
rational surface. Its singular-value decomposition over a chosen set of rational surfaces
ranks the applied spectra by how strongly they drive resonant field there: the first right
singular vector is the **dominant mode**, the spectrum the plasma is most sensitive to, and the
singular values are coordinate-invariant.

The run always writes the full coupling matrix, so the surface window is an analysis choice
made afterwards — never a reason to re-run. A `ResonantCoupling` bundles the matrix with the
labels and normalization needed to evaluate arbitrary applied spectra against it, and is built
the same way from a finished run in memory or from its `gpec.h5`:

```julia
using GeneralizedPerturbedEquilibrium.PerturbedEquilibrium

rc = ResonantCoupling("gpec.h5")                 # post hoc; or ResonantCoupling(pe_state, ffs) in memory
dom = dominant_coupling(rc; psi_low=0.0, psi_high=0.9)   # SVD over the surfaces in the window

b̃ = rootarea_field(rc, coil_modes)               # unit-norm forcing modes → root-area-weighted field
c = coupling_overlap(dom, b̃)                     # Vᴴ·b̃: c[1] is the overlap with the dominant mode
dom.singular_values[1] * abs(c[1])               # resonant field the dominant mode drives
```

`rootarea_field` takes care of the mode ordering and the `R⁻¹` conform between the unit-norm
convention the forcing loaders and coil integration produce and the b̃ basis the matrix acts on;
`coupling_overlap` takes care of the conjugation. Singular vectors carry an arbitrary global
phase, so compare `abs` of overlaps across runs, not the complex value.

As a summary the run also stores the core-window decomposition (`ψ_N ≤ CORE_PSI_HIGH = 0.9`) under
`PerturbedEquilibrium/SingularCoupling/DominantMode/`, with `forcing_overlap` holding the run's
own forcing coefficients `Vᴴ·b̃_x`.

```@docs
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.ResonantCoupling
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.DominantCoupling
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.dominant_coupling
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.CORE_PSI_HIGH
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.rootarea_field
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.coupling_overlap
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.check_mode_basis
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.compute_dominant_coupling!
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
