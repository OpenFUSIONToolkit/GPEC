# Perturbed Equilibrium

`PerturbedEquilibrium` computes the response of the plasma to an external non-axisymmetric field: the three-dimensional perturbed equilibrium that the force-free states and the applied field together form.
It is the stage that turns a stability calculation into the quantities of error-field and RMP physics: the field that actually reaches each rational surface, the islands it would open, and the energy and torque the perturbation carries.

## What it computes

The force-free states of [Stability](stability.md) give the displacement inside the plasma for any boundary value, and the [Vacuum](vacuum.md) response gives the field outside it.
On the control surface at `psilim` the two are joined by requiring continuity of the normal field, which fixes the plasma response to an applied spectrum uniquely ([Park et al. (2007a)](citations.md#PerturbedEquilibrium-Module), [Park et al. (2009)](citations.md#PerturbedEquilibrium-Module)).
`plasma_response(ffs)` builds the operators of that response once, with no reference to a drive; `perturbed_equilibrium(resp, rmp)` applies them to one external field from [Forcing Terms](forcing_terms.md), so a scan over fields reuses one response.
`compute_perturbed_equilibrium` is the lower-level form of the same call.

The resonant part of the response is reported at every rational surface as the singular coupling: the pitch-resonant field and current, the island half-width and the Chirikov overlap parameter.
In an ideal solve the plasma shields every rational surface perfectly, so the stored resonant field is the effective one, the field the ideal response would deliver to the surface before shielding; a kinetic solve (`kinetic = "KineticForces"`) gives a finite penetrated field. See the [Conventions](conventions.md) page.

## Governing equations

The applied field is specified as its normal-flux spectrum ``\Phi_x`` on the control surface, one entry per (m, n).
The plasma inductance ``\Lambda`` relates the total flux to the surface current the plasma response induces, and the surface inductance ``L`` is the vacuum inductance of the same surface; the plasma response is the permeability

```math
\Phi = P\,\Phi_x, \qquad P = \Lambda L^{-1},
```


oindent and the reluctance ``arrho = L^{-1}(\Lambda^\dagger - L)L^{-1}`` gives the surface current per unit applied flux.
The perturbed plasma energy and the toroidal torque of the response are quadratic forms in the total flux,

```math
\delta W_\mathrm{plasma} = 	frac14\,\Phi^\dagger \Lambda^{-1}\Phi, \qquad
T_\phi = -	frac{2n}{4}\,\operatorname{Im}\!\left(\Phi^\dagger \Lambda^{-1}\Phi
ight),
```


oindent with the vacuum and surface energies written alongside them.
The stored matrices act on the coordinate-invariant root-area-weighted field ``	ilde b``; the area-weighted field ``ar b`` and the flux follow from ``ar b = S	ilde b`` and ``\Phi = Aar b`` with the operator `S` and the control-surface area `A` stored next to them (see *Field amplitudes* on the [Conventions](conventions.md) page).

At each rational surface q = m/n the singular coupling evaluates the resonant flux ``\Phi^r`` of the response and reports the area-weighted resonant field ``ar b^r = \Phi^r / A^r`` in tesla, the pitch-resonant current, and the island the resonant flux would open in vacuum superposition, with half-width

```math
w_s \propto \sqrt{|\delta\psi_s|}
```


oindent in normalized flux and the Chirikov parameter ``\sigma_s = w_s / d_s``, the half-width over the half-distance to the neighbouring rational surface, so ``\sigma > 1`` means the islands overlap.
The coupling matrices `C_*` map the applied ``	ilde b`` to each of these quantities, and their singular-value decomposition ranks the applied spectra by resonant drive (see below).

## Running it

From `examples/DIIID-like_ideal_example/gpec.toml`, after the `[ForceFreeStates]` solve and a `[ForcingTerms]` drive:

```toml
[PerturbedEquilibrium]
compute_singular_coupling = true   # resonant fields, islands and Chirikov parameter
regularization_width = 0.05        # width of the regularization of the ideal response at each rational surface
```

`compute_torque_response = true` adds the torque response matrices of a kinetic forward solve (below).
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
| `PerturbedEquilibrium/ResponseMatrices/plasma_inductance`, `surface_inductance`, `permeability`, `reluctance` | ``	ilde\Lambda``, ``	ilde L``, ``	ilde P``, ``	ildearrho`` in root-area-weighted field space | |
| `PerturbedEquilibrium/ResponseMatrices/rootarea_to_area_weight_operator`, `surface_area` | ``S`` and ``A`` of the control surface | m² (`surface_area`) |
| `PerturbedEquilibrium/Response/psi` | the ψ_N grid of the profiles | |
| `PerturbedEquilibrium/Response/xi_n`, `xi_R`, `xi_Z`, `xi_phi` | the displacement in mode space: normal and cylindrical components | m |
| `PerturbedEquilibrium/Response/xi_psi`, `Jxi_*`, `xi_cov_*`, `xi_clebsch_*` | its contravariant, Jacobian-weighted, covariant and Clebsch components | |
| `PerturbedEquilibrium/Response/b_n`, `b_R`, `b_Z`, `b_phi`, `b_psi_area_weighted` | the perturbed field in mode space: normal, cylindrical and area-weighted normal components | T |
| `PerturbedEquilibrium/Response/Jb_*`, `b_cov_*` | its Jacobian-weighted and covariant components | |
| `PerturbedEquilibrium/SingularCoupling/rational_psi`, `rational_q`, `rational_m`, `rational_n`, `rational_area` | the resonant (surface, n) rows and their surface areas | m² (`rational_area`) |
| `PerturbedEquilibrium/SingularCoupling/resonant_area_weighted_field` | ``ar b^r`` at each rational surface | T |
| `PerturbedEquilibrium/SingularCoupling/penetrated_area_weighted_field` | the penetrated area-weighted field | T |
| `PerturbedEquilibrium/SingularCoupling/resonant_current` | the pitch-resonant current | A |
| `PerturbedEquilibrium/SingularCoupling/island_width_sq`, `island_half_width` | ``w_s^2`` and ``w_s`` in ψ_N | |
| `PerturbedEquilibrium/SingularCoupling/chirikov_parameter` | ``\sigma_s`` | |
| `PerturbedEquilibrium/SingularCoupling/Delta_prime` | the forcing-driven tearing Δ′ per surface (Riccati solves) | |
| `PerturbedEquilibrium/SingularCoupling/C_*` | the coupling matrices from the applied ``	ilde b`` to each quantity above | |
| `PerturbedEquilibrium/SingularCoupling/DominantMode/` | the core-window SVD of the resonant-field coupling (below) | |
| `PerturbedEquilibrium/Energies/plasma_energy`, `vacuum_energy`, `surface_energy` | the perturbed energies | J |
| `PerturbedEquilibrium/Energies/toroidal_torque` | the boundary-response torque ``T_\phi`` | N·m |
| `PerturbedEquilibrium/TorqueResponse/` | the torque response matrices of a kinetic forward solve (below) | N·m/T² |

The `Analysis` module plots these; see [Analysis](analysis.md).

## Types

```@docs
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.PlasmaResponse
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.PerturbedEquilibriumState
```

## Functions

```@docs
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.plasma_response
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.compute_perturbed_equilibrium
```

## Several toroidal modes

With several toroidal modes (`nn = [1, 2, ...]`) every `mode` axis in `PerturbedEquilibrium/` runs over the full
`numpert_total = mpert·npert` space, with m varying fastest and one block of `mpert` entries for each n.
`PerturbedEquilibrium/mode_m` and `mode_n` label each entry. The equilibrium is axisymmetric, so
different n never couple:

  - Λ, L, P and ϱ are block-diagonal in n, and each block equals the matrix of a single-n run.
  - Energies and torque are sums of the single-n values; the plasma energy Φ_n†·Λ_nn⁻¹·Φ_n/4 and the
    torque −2n·Im(Φ_n†·Λ_nn⁻¹·Φ_n)/4 use only the diagonal n blocks of Λ: off-n blocks vanish for an
    axisymmetric equilibrium, so numerical leakage there must not enter.
  - Profiles in `Response/` hold one block of columns for each n.
  - `SingularCoupling/` has one row for each resonant (surface, n) pair, sorted by ψ across all n.
    An integer-q surface appears once for each n that resonates there. Those rows are not
    neighbors of each other when the Chirikov parameter is computed.

`truncate_at_rational_offset` is ignored for multi-n runs, so set the edge with `qhigh` or `psihigh`,
away from any rational m/n in the range. KineticForces still takes a single n.

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

rc = ResonantCoupling("gpec.h5")                 # post hoc; or ResonantCoupling(resp) in memory
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
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.build_dominant_coupling!
```

## Torque response matrices

A kinetic forward solve (`integrator = "Forward"`, `kinetic = "KineticForces"`) carries the
drift-kinetic response inside its Euler-Lagrange energy: the complex plasma energy inside a
flux surface, `δW(ψ) = ξ†·U₂(ψ)·U₁(ψ)⁻¹·ξ/(2μ₀)`, has an anti-Hermitian part that is the
neoclassical toroidal viscosity torque (Logan & Park 2013, eq. 19). On request,
`plasma_response(ffs; compute_torque_response=true)` (deck: `compute_torque_response = true`
in `[PerturbedEquilibrium]`) refers that quadratic form back to the
applied root-area-weighted control-surface field b̃ — through the permeability, the
boundary flux-to-displacement factor `χ₁(m − n·q_lim)·2πi` and the b̃ → Φ conform `R` — and
stores the result at every stored node as the **torque response matrix** `T_xe(ψ)`, cumulative
in ψ, in `PerturbedEquilibrium/TorqueResponse/`. It is the Fortran GPEC `T_xe`
(`gpout_dw_matrix`), in the same basis as the stored response matrices, with units N·m/T².

Two conventions are worth stating explicitly:

  - **Factor of 2.** `T_xe` matches the Fortran array, whose quadratic form is built on the `+n`
    harmonic alone and is therefore twice the physical value. The physical cumulative complex
    torque of an applied spectrum is `T(ψ) = b̃†·T_xe(ψ)·b̃/2` — `torque_profile` applies the
    1/2 — with `real(T)` the toroidal torque on the plasma inside ψ and `imag(T)` its `2n·δW`
    (ideal plus kinetic). At the control surface `real(T)` equals `Energies/toroidal_torque`.
  - **Hermitian part.** Only the Hermitian part `(T_xe + T_xe†)/2` contributes to the torque;
    its eigenvectors are the applied spectra of extremal torque and its eigenvalues their torque
    per unit `‖b̃‖²/2`. The anti-Hermitian part carries the energy.

Each `perturbed_equilibrium(resp, rmp)` call contracts it with that field into a `TorqueResponse`:
the applied torque `T_applied` and the forcing-source form `T_source(ψ) = M†·T·M`, with one column
of `M` per forcing source, each as built: every coil set at its deck currents,
or the forcing file's spectrum as a single source. Driving the sources at scale factors `s` gives
`s†·T_source·s/2`.

```julia
using GeneralizedPerturbedEquilibrium.PerturbedEquilibrium
using LinearAlgebra

tr = pe_state.torque_response                       # a TorqueResponse
T = torque_profile(tr, pe_state.forcing_b_rootarea) # cumulative complex torque of the run's forcing
real(T[end])                                        # total toroidal torque [N·m]

T_h = Hermitian((tr.T_xe[:, :, end] + tr.T_xe[:, :, end]') / 2)
λ, V = eigen(T_h)                                   # V[:, end]: applied b̃ of maximum torque
```

```@docs
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.TorqueResponse
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.torque_response_matrix
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.torque_profile
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.coil_flux_spectra
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.build_torque_response!
GeneralizedPerturbedEquilibrium.PerturbedEquilibrium.apply_torque_response!
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
