# Perturbed Equilibrium

`PerturbedEquilibrium` computes the response of the plasma to an external non-axisymmetric field: the three-dimensional perturbed equilibrium that the force-free states and the applied field together form.
It is the stage that turns a stability calculation into the quantities of error-field and RMP physics: the field that actually reaches each rational surface, the islands it would open, and the energy and torque the perturbation carries.

## What it computes

The force-free states of [Stability](stability.md) give the displacement inside the plasma for any boundary value, and the [Vacuum](vacuum.md) response gives the field outside it.
On the control surface at `psilim` the two are joined by requiring continuity of the normal field, which fixes the plasma response to an applied spectrum uniquely ([Park et al. (2007a)](citations.md#PerturbedEquilibrium), [Park et al. (2009)](citations.md#PerturbedEquilibrium)).
`plasma_response(ffs)` builds the operators of that response once, with no reference to a drive; `perturbed_equilibrium(resp, rmp)` applies them to one external field from [Forcing Terms](forcing_terms.md), so a scan over fields reuses one response.
`compute_perturbed_equilibrium` is the lower-level form of the same call.

The resonant part of the response is reported at every rational surface as the singular coupling: the pitch-resonant field and current, the island half-width and the Chirikov overlap parameter.
In an ideal solve the plasma shields every rational surface perfectly, so the stored resonant field is the effective one, the field the ideal response would deliver to the surface before shielding; a kinetic solve (`kinetic = "KineticForces"`) gives a finite penetrated field. See the [Conventions Reference](conventions.md).

### Dominant resonant-coupling mode

The coupling matrix `C_resonant_area_weighted_field` maps an applied root-area-weighted spectrum ``\tilde b`` on the control surface to the resonant field at each rational surface.
Its singular-value decomposition over a window of rational surfaces ranks the applied spectra by how strongly they drive resonant field there: the first right singular vector is the **dominant mode**, the spectrum the plasma is most sensitive to, and the singular values are coordinate-invariant.
The run writes the full matrix, so the window is an analysis choice made afterwards: a `ResonantCoupling` bundles the matrix with its labels and normalization, from the result in memory or from `gpec.h5`, and `dominant_coupling` takes the SVD over any window; the run also stores the core-window decomposition (``\psi_N \le`` `CORE_PSI_HIGH` ``= 0.9``) under `PerturbedEquilibrium/SingularCoupling/DominantMode/`, with `forcing_overlap` the run's own forcing coefficients ``V^H \tilde b_x``.

```julia
using GeneralizedPerturbedEquilibrium.PerturbedEquilibrium

rc  = ResonantCoupling("gpec.h5")                        # or ResonantCoupling(resp) in memory
dom = dominant_coupling(rc; psi_low=0.0, psi_high=0.9)   # SVD over the surfaces in the window
c   = coupling_overlap(dom, rootarea_field(rc, coil_modes))   # c[1]: overlap of a forcing with the dominant mode
dom.singular_values[1] * abs(c[1])                       # resonant field the dominant mode drives
```

Singular vectors carry an arbitrary global phase, so compare `abs` of overlaps across runs, not the complex value.

In a multi-n run the matrix is block-diagonal in n, so every singular mode has a single n; the ranking spans all n, so the dominant mode, and the stored `DominantMode`, belong to the most strongly coupled n.
For one n, decompose its block, `dominant_coupling(rc.C[rows, cols], rc.rational_psi[rows])` with `rows = rc.rational_n .== n` and `cols = rc.n_modes .== n`, and project `rootarea_field(rc, modes)[cols]`.

The run writes up to five coupling matrices under `PerturbedEquilibrium/SingularCoupling/`, each with one row per rational surface and the same applied ``\tilde b`` columns:

| Dataset | Maps ``\tilde b`` to | Units |
|---|---|---|
| `C_resonant_area_weighted_field` | the resonant field ``\bar b^r`` | dimensionless |
| `C_resonant_current` | the pitch-resonant current | A/T |
| `C_island_width_sq` | the squared island half-width | 1/T |
| `C_penetrated_area_weighted_field` | the penetrated field, when the run computes it | dimensionless |
| `C_Delta_prime` | the forcing-driven Δ′ | 1/T |

`ResonantCoupling` wraps the first; any of the others decomposes over any window of surfaces through the bare-matrix form of `dominant_coupling`, with the ``\tilde b`` conform taken from a `ResonantCoupling` of the same file, since the bare matrix carries no (m, n) labels for `check_mode_basis` to guard the projection:

```julia
C, ψr = h5open(f -> (read(f, "PerturbedEquilibrium/SingularCoupling/C_resonant_current"),
                     read(f, "PerturbedEquilibrium/SingularCoupling/rational_psi")), "gpec.h5")
dom = dominant_coupling(C, ψr; psi_low=0.3, psi_high=0.6)
dom.singular_values .* coupling_overlap(dom, rootarea_field(rc, coil_modes))   # in C's units times T
```

### Torque response matrices

A kinetic forward solve (`integrator = "Forward"`, `kinetic = "KineticForces"`) carries the drift-kinetic response inside its Euler–Lagrange energy: the complex plasma energy inside a flux surface has an anti-Hermitian part that is the neoclassical toroidal viscosity torque ([Logan et al. (2013)](citations.md#KineticForces)).
With `compute_torque_response = true`, `plasma_response` refers that quadratic form back to the applied root-area-weighted field ``\tilde b`` and stores it at every node as the torque response matrix ``T_{xe}(\psi)``, cumulative in ψ, under `PerturbedEquilibrium/TorqueResponse/`; it is the `T_xe` of the Fortran GPEC, in the basis of the stored response matrices, in N·m/T².
``T_{xe}`` is built on the ``+n`` harmonic alone and is therefore twice the physical value, so the cumulative complex torque of an applied spectrum is ``T(\psi) = \tilde b^\dagger T_{xe}(\psi)\, \tilde b / 2`` (`torque_profile` applies the 1/2), with ``\operatorname{Re} T`` the toroidal torque inside ψ, equal to `Energies/toroidal_torque` at the control surface, and ``\operatorname{Im} T = 2n\,\delta W``.
Only the Hermitian part ``(T_{xe} + T_{xe}^\dagger)/2`` contributes to the torque; its eigenvectors are the applied spectra of extremal torque, and the anti-Hermitian part carries the energy.
Each `perturbed_equilibrium(resp, rmp)` call contracts ``T_{xe}`` with its field into a `TorqueResponse`: the applied torque ``T_\mathrm{applied}`` and the source form ``T_\mathrm{source} = M^\dagger T M`` with one column of ``M`` per forcing source, so that driving the sources at scale factors ``s`` gives ``s^\dagger T_\mathrm{source}\, s / 2``.

## Governing equations

The applied field is specified as its normal-flux spectrum ``\Phi_x`` on the control surface, one entry per (m, n).
The plasma inductance ``\Lambda`` relates the total flux to the surface current the plasma response induces, and the surface inductance ``L`` is the vacuum inductance of the same surface; the plasma response is the permeability

```math
\Phi = P\,\Phi_x, \qquad P = \Lambda L^{-1},
```

and the reluctance ``\varrho = L^{-1}(\Lambda^\dagger - L)L^{-1}`` gives the surface current per unit applied flux.
The perturbed plasma energy and the toroidal torque of the response are quadratic forms in the total flux,

```math
\delta W_\mathrm{plasma} = \tfrac14\,\Phi^\dagger \Lambda^{-1}\Phi, \qquad
T_\phi = -\tfrac{2n}{4}\,\operatorname{Im}\!\left(\Phi^\dagger \Lambda^{-1}\Phi\right),
```

with the vacuum and surface energies written alongside them.
The stored matrices act on the coordinate-invariant root-area-weighted field ``\tilde b``; the area-weighted field ``\bar b`` and the flux follow from ``\bar b = S\tilde b`` and ``\Phi = A\bar b`` with the operator ``S`` and the control-surface area ``A`` stored next to them (see *Field amplitudes and units* on the [Conventions Reference](conventions.md)).

At each rational surface q = m/n the singular coupling evaluates the resonant flux ``\Phi^r`` of the response and reports the area-weighted resonant field ``\bar b^r = \Phi^r / A^r`` in tesla, the pitch-resonant current, and the island the resonant flux would open in vacuum superposition, with half-width

```math
w_s \propto \sqrt{|\delta\psi_s|}
```

in normalized flux, and the Chirikov parameter ``\sigma_s = w_s / d_s``, the half-width over the half-distance to the neighbouring rational surface, so ``\sigma > 1`` means the islands overlap.
The coupling matrices `C_*` map the applied ``\tilde b`` to each of these quantities.

## Running it

The response needs a force-free-states solve with `integrator = "Forward"` and no `[Wall]`.
The operators are built from a dense ``\xi`` solution with its free-boundary δW, which a `"Forward"` solve and a `"Galerkin"` solve with `error_field_response = true` and `vac_flag = true` both provide: a `"Riccati"` solve (the deck default) makes `plasma_response` warn that the response is skipped and return empty operators, so the failure surfaces only when a field is applied, while a `"Galerkin"` solve without those flags or a `[Match]`ed solve, whose singular coupling is not defined yet, is an error. The ψ-resolved torque response (`compute_torque_response`) needs a `"Forward"` kinetic solve.
A solve with a wall is an error too, because the surface inductance is the free-space one and ``\Lambda`` from a wall run would mix two vacuum models; the wall case of a stability study is therefore a separate deck.

From `examples/DIIID-like_ideal_example/gpec.toml`, after the `[ForceFreeStates]` solve and a `[ForcingTerms]` drive:

```toml
[ForceFreeStates]
integrator = "Forward"             # the dense ξ solution the response needs ("Riccati" is the default); no [Wall]

[PerturbedEquilibrium]
compute_singular_coupling = true   # resonant fields, islands and Chirikov parameter
regularization_width = 0.05        # regularization of the ideal response at each rational surface (default 0)
```

`regularization_width` is for the NTV torque of an ideal solve only; see [Kinetic Forces](kinetic_forces.md).
`compute_torque_response = true` adds the torque response matrices of a kinetic forward solve.
In a script the same two steps are

```julia
using GeneralizedPerturbedEquilibrium

resp = plasma_response(ffs; compute_singular_coupling=true)
pe   = perturbed_equilibrium(resp, rmp; regularization_width=0.05)   # rmp: an RMPField, see the Scripting API
save_to_hdf5("gpec.h5", pe)
```

with the `RMPField` built from coils or a file as on the [Scripting API](api.md) page; computing the response does not write a file.

## Outputs

| Dataset | Meaning | Units |
|---|---|---|
| `PerturbedEquilibrium/mode_m`, `mode_n` | the (m, n) of each entry of the mode axis, m fastest, one block per n | |
| `PerturbedEquilibrium/ForcingModes/m`, `n`, `amplitude` | the applied spectrum | T |
| `PerturbedEquilibrium/forcing_b`, `forcing_b_root_area`, `forcing_b_area` | the applied control-surface spectrum as bare, root-area-weighted and area-weighted normal field | T |
| `PerturbedEquilibrium/response_b`, `response_b_root_area`, `response_b_area` | the total (applied plus plasma) spectrum in the same three forms | T |
| `PerturbedEquilibrium/ResponseMatrices/plasma_inductance`, `surface_inductance`, `permeability`, `reluctance` | ``\tilde\Lambda``, ``\tilde L``, ``\tilde P``, ``\tilde\varrho`` in root-area-weighted field space | |
| `PerturbedEquilibrium/ResponseMatrices/rootarea_to_area_weight_operator`, `surface_area` | ``S`` and ``A`` of the control surface | m² (`surface_area`) |
| `PerturbedEquilibrium/Response/psi` | the ψ_N grid of the profiles | |
| `PerturbedEquilibrium/Response/xi_n`, `xi_R`, `xi_Z`, `xi_phi` | the displacement in mode space: normal and cylindrical components | m |
| `PerturbedEquilibrium/Response/xi_psi`, `Jxi_*`, `xi_cov_*`, `xi_clebsch_*` | its contravariant, Jacobian-weighted, covariant and Clebsch components | |
| `PerturbedEquilibrium/Response/b_n`, `b_R`, `b_Z`, `b_phi`, `b_psi_area_weighted` | the perturbed field in mode space: normal, cylindrical and area-weighted normal components | T |
| `PerturbedEquilibrium/Response/Jb_*`, `b_cov_*` | its Jacobian-weighted and covariant components | |
| `PerturbedEquilibrium/SingularCoupling/rational_psi`, `rational_q`, `rational_m`, `rational_n`, `rational_area` | the resonant (surface, n) rows and their surface areas | m² (`rational_area`) |
| `PerturbedEquilibrium/SingularCoupling/resonant_area_weighted_field` | ``\bar b^r`` at each rational surface | T |
| `PerturbedEquilibrium/SingularCoupling/penetrated_area_weighted_field` | the penetrated area-weighted field | T |
| `PerturbedEquilibrium/SingularCoupling/resonant_current` | the pitch-resonant current | A |
| `PerturbedEquilibrium/SingularCoupling/island_width_sq`, `island_half_width` | ``w_s^2`` and ``w_s`` in ψ_N | |
| `PerturbedEquilibrium/SingularCoupling/chirikov_parameter` | ``\sigma_s`` | |
| `PerturbedEquilibrium/SingularCoupling/Delta_prime` | the forcing-driven tearing Δ′ per surface | |
| `PerturbedEquilibrium/SingularCoupling/C_*` | the coupling matrices from the applied ``\tilde b`` to each quantity above | |
| `PerturbedEquilibrium/SingularCoupling/DominantMode/singular_values`, `right_singular_vectors`, `left_singular_vectors`, `rational_index`, `forcing_overlap` | the core-window SVD of the resonant-field coupling | |
| `PerturbedEquilibrium/Energies/plasma_energy`, `vacuum_energy`, `surface_energy` | the perturbed energies | J |
| `PerturbedEquilibrium/Energies/toroidal_torque` | the boundary-response torque ``T_\phi`` | N·m |
| `PerturbedEquilibrium/TorqueResponse/psi`, `m`, `n`, `T_xe`, `T_source`, `T_applied`, `total_torque` | the torque response matrices of a kinetic forward solve and their contraction with the run's forcing | N·m/T², N·m |

The `Analysis` module plots these; see [Analysis](analysis.md).
`SingularCoupling/` quantities are indexed by rational-surface index, since with several n one q value can host several resonances; `rational_psi` and `rational_q` are attached to that axis as HDF5 dimension scales, so plotting against either is direct (`read(g["rational_q"])` in Julia, `dset.dims[0]["q_rational"]` through `h5py`).

## Numerical notes

With several toroidal modes (`nn = [1, 2, ...]`) every `mode` axis in `PerturbedEquilibrium/` runs over the full `numpert_total = mpert·npert` space, m fastest and one block of `mpert` entries per n, labelled by `mode_m` and `mode_n`.
The equilibrium is axisymmetric, so different n never couple: ``\Lambda``, ``L``, ``P`` and ``\varrho`` are block-diagonal in n with each block equal to the single-n matrix, the energies and the torque are sums of the single-n values evaluated on the diagonal blocks alone, `Response/` holds one block of columns per n, and `SingularCoupling/` has one row per resonant (surface, n) pair sorted by ψ, so an integer-q surface appears once for each n that resonates there and those rows are not neighbours in the Chirikov parameter.
`truncate_at_rational_offset` is ignored for multi-n runs, so set the edge with `qhigh` or `psihigh` away from any rational m/n in the range; `KineticForces` still takes a single n.

## API reference

```@docs
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.PlasmaResponse
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.PerturbedEquilibriumState
```

```@docs
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.plasma_response
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.compute_perturbed_equilibrium
```

```@docs
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.ResonantCoupling
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.DominantCoupling
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.dominant_coupling
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.CORE_PSI_HIGH
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.rootarea_field
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.coupling_overlap
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.check_mode_basis
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.build_dominant_coupling!
```

```@docs
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.TorqueResponse
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.torque_response_matrix
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.torque_profile
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.coil_flux_spectra
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.build_torque_response!
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.apply_torque_response!
```
