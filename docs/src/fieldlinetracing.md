# Field-Line Tracing

The `FieldLineTracing` module integrates magnetic field lines through the
GPEC-computed field to reveal its three-dimensional topology: magnetic islands at
rational surfaces, the stochastic layers that form where islands overlap, connection
lengths, and divertor footprints. These are the diagnostics familiar from vacuum
field-line codes such as TRIP3D and MAFOT, computed here directly from the GPEC
perturbed-equilibrium response.

The traced perturbation is selectable — the externally applied (`vacuum`) field, the
plasma-response field, or their sum (`total`) — and integration runs either in
straight-field-line coordinates (interior) or in real cylindrical space (inside and
outside the control surface).

## Pipeline placement

Field-line tracing runs as the final stage of the analysis pipeline, consuming the
singular-coupling response already computed by `PerturbedEquilibrium`:

```
   Equilibrium ─► ForceFreeStates ─► PerturbedEquilibrium ─────────────► FieldLineTracing ─► gpec.h5
    (q, rzphi)      (eigenmodes)      (singular coupling:                 (Poincaré,           │
                                       C_island_width_sq,                  connection length,  ▼
                                       forcing/response b̃)                 footprints,       Analysis
                                                                           island widths)     (plots)
```

The stage is gated on a `[FieldLineTracing]` section in `gpec.toml`; because it is
placed after `PerturbedEquilibrium`, it inherits the existing `force_termination`
early-exit semantics and is skipped for equilibrium- or stability-only runs.

## Choosing what to trace

```
  tracing_coords ─┬─ "flux"  (default) ── interior (ψ,θ,ζ) Hamiltonian map
                  └─ "real"            ── cylindrical (R,Z,φ); inside AND outside the LCFS

  tracing_field  ─┬─ "total"  (default) ── applied field + plasma response   (Φ_tot = P·Φ_x)
                  ├─ "vacuum"           ── externally applied field only      (Φ_x)
                  └─ "plasma"           ── plasma-response part only          (total − vacuum)
```

## Flux-coordinate field-line map

In straight-field-line coordinates ``(\psi,\theta,\zeta)`` (``\theta`` and ``\zeta``
normalized to ``[0,1)``), field lines follow a Hamiltonian map whose background winding
is the inverse safety factor and whose perturbation is a helical flux ``\Omega``:

```math
\frac{d\psi}{d\zeta} = -\frac{\partial\Omega}{\partial\theta},
\qquad
\frac{d\theta}{d\zeta} = \frac{1}{q(\psi)} + \frac{\partial\Omega}{\partial\psi}.
```

The map is area-preserving (Hamiltonian ``H = \int d\psi/q + \Omega``), so island
topology is well posed. With no perturbation, ``d\psi/d\zeta = 0`` and
``d\theta/d\zeta = 1/q`` — field lines lie on nested surfaces ``\psi = \text{const}``.

### True field drive

The helical flux is built from the **actual perturbed field** of the selected source — not
an idealized island model. The normalized radial drive is projected from the physical field
onto the flux surfaces,

```math
R_\psi(\psi,\theta,\zeta)
= \frac{\mathbf{B}_{\text{pert}}\cdot\nabla\psi_N}{\mathbf{B}_0\cdot\nabla\zeta},
\qquad
\mathbf{B}_0\cdot\nabla\zeta = \frac{F}{(2\pi R)^2},
\quad
\mathbf{B}_{\text{pert}}\cdot\nabla\psi_N = |\nabla\psi_N|\,(\mathbf{B}_{\text{pert}}\cdot\hat{n}),
```

where ``\hat{n}`` is the outward flux-surface normal (``\nabla\psi`` direction, from rotating
the poloidal tangent ``\partial(R,Z)/\partial\theta`` by ``-90^\circ``). Fourier-decomposing
over ``(\theta,\zeta)`` gives the full harmonic set ``R_{\psi,m}(\psi)`` at toroidal number
``n``, and the helical flux follows from ``-\partial\Omega/\partial\theta = R_\psi``:

```math
\Omega(\psi,\theta,\zeta) = \operatorname{Re}\!\sum_m \frac{i\,R_{\psi,m}(\psi)}{2\pi m}\,
  e^{\,i\,2\pi(m\theta - n\zeta)},
\qquad
\frac{d\psi}{d\zeta} = \operatorname{Re}\!\sum_m R_{\psi,m}(\psi)\,e^{\,i\,2\pi(m\theta - n\zeta)}.
```

All poloidal harmonics enter, so islands come out textured and sheared (real X-points and
separatrices) and evolve with the applied field — matching a direct field-line integration.
The perturbed field per source:

- **vacuum** — the exact coil Biot-Savart field (`ForcingTerms.compute_biot_savart_boundary!`),
  sampled on a ``(\psi,\theta,\zeta)`` grid. Unscreened → finite resonant radial field →
  **real islands**.
- **total** — the reconstructed physical cylindrical field ``b_{R},b_{Z}``. Ideal MHD
  screens the resonant component (``\propto(m-nq)\to0`` at rationals), so `total` correctly
  yields **nested surfaces with no islands** (the ideal result), not a bug.
- **plasma** — `total − vacuum`.

The reconstructed cylindrical components (`total`/`plasma`) are flagged beta (~20 % vs
Fortran); the `vacuum` coil field is exact.

!!! note "Island-width diagnostic vs the trace"
    The `islands/` HDF5 group separately reports GPEC's singular-coupling island half-width
    ``\sqrt{|\text{island\_width\_sq}|}`` and Chirikov overlap — an independent, validated
    metric. It does **not** drive the trace; the Poincaré section is a true field-line
    integration.

### Chirikov overlap and stochasticity

Adjacent islands overlap — producing a stochastic field-line region — when the sum of
their half-widths approaches their separation. The Chirikov parameter reported per
neighbouring pair is

```math
\sigma_i = \frac{w_{\tfrac12,\,i} + w_{\tfrac12,\,i+1}}{\lvert\psi_{s,i+1}-\psi_{s,i}\rvert},
```

with ``\sigma \gtrsim 1`` signalling overlap. In the traced Poincaré section this appears
as island chains merging into a scattered stochastic sea, typically first at the plasma
edge where the rational surfaces crowd together.

## Field sources

The three sources differ only in the control-surface field amplitude fed to the same
resonant coupling matrix. Working in the coordinate-invariant root-area-weighted field
basis ``\tilde{b}`` (the basis of ``C_{\text{island\_width\_sq}}``):

```math
\text{island\_width\_sq}^{\text{(src)}}
= C_{\text{island\_width\_sq}}\cdot \tilde{b}^{\text{(src)}},
\qquad
\tilde{b}^{\text{(src)}} =
\begin{cases}
\tilde{b}_{\text{forcing}} & \text{vacuum } (\Phi_x)\\[2pt]
\tilde{b}_{\text{response}} & \text{total } (P\,\Phi_x)\\[2pt]
\tilde{b}_{\text{response}} - \tilde{b}_{\text{forcing}} & \text{plasma}
\end{cases}
```

Because the coupling is linear in the applied spectrum, the `vacuum` result reproduces
GPEC's stored `perturbed_equilibrium/singular_coupling/island_half_width` exactly. The
`total` result can be larger than `vacuum` when the marginally-stable plasma amplifies
the resonant field (resonant field amplification) or smaller when it screens it.

## Real-space field-line map

For tracing inside and outside the LCFS, field lines are integrated in cylindrical
coordinates:

```math
\frac{dR}{d\varphi} = R\,\frac{B_R}{B_\varphi},
\qquad
\frac{dZ}{d\varphi} = R\,\frac{B_Z}{B_\varphi},
\qquad
\mathbf{B} = \mathbf{B}_{\text{eq}} + \delta\mathbf{B}.
```

The axisymmetric background is recovered by inverting ``(R,Z)\to(\psi,\theta)`` (Newton
iteration on the forward `rzphi` map). Inside the control surface,

```math
B_{\text{pol}} = \frac{\Psi_0\,|\nabla\psi|}{R},
\qquad
B_\varphi = \frac{F}{R},
```

with the poloidal field directed along the flux-surface tangent ``d(R,Z)/d\theta`` and
its sense set by the helicity ``\operatorname{sign}(B_t)\operatorname{sign}(I_p)``.
Outside the control surface only the vacuum toroidal field ``B_\varphi = F_{\text{edge}}/R``
is retained. The perturbation ``\delta\mathbf{B}`` is the exact coil Biot–Savart field
for the `vacuum` source (from `ForcingTerms.compute_biot_savart_boundary!`), or the
reconstructed cylindrical spectra for `plasma`/`total`.

!!! warning "Beta accuracy of the reconstructed real-space perturbation"
    The reconstructed cylindrical components ``b_R,b_Z,b_\varphi`` carry up to ~20 %
    discrepancy versus the Fortran reference (see [`Perturbed Equilibrium`](perturbed_equilibrium.md)).
    The `vacuum` coil path (exact) and the flux-coordinate map (validated against GPEC
    island widths) are the trustworthy routes.

## Diagnostics

| Diagnostic | Description | Mode |
|---|---|---|
| **Poincaré section** | Puncture points at a toroidal plane over many launched lines; island chains and stochastic regions | flux + real |
| **Connection length / laminar** | Toroidal transits (or metres) before a line escapes the control surface or strikes the wall; laminar vs stochastic vs private-flux | flux + real |
| **Divertor footprints** | Strike points where lines intersect the wall polygon | real (needs a wall) |
| **Island width / Chirikov** | Per-surface half-width ``\sqrt{\lvert\text{island\_width\_sq}\rvert}`` and overlap ``\sigma`` | flux + real |

Field lines are integrated with `OrdinaryDiffEq` (`Vern9`), recording punctures at the
requested section planes; escape from the control surface (flux mode) or wall
intersection (real mode) terminates a line via a callback.

## Configuration

The `[FieldLineTracing]` section of `gpec.toml`:

```toml
[FieldLineTracing]
tracing_coords = "flux"      # "flux" (interior) or "real" ((R,Z,φ), inside+outside)
tracing_field  = "total"     # "total", "vacuum", or "plasma"
n_lines        = 60          # number of field lines launched
psi_start      = 0.05        # innermost launch surface (normalized poloidal flux)
psi_end        = 0.98        # outermost launch surface
n_transits     = 400         # toroidal transits integrated per line
phi_planes     = [0.0]       # toroidal angle(s) of the Poincaré section [rad]
tol            = 1e-9        # relative ODE tolerance
compute_connection_length = true
compute_footprints        = true
compute_island_width      = true
```

See [`FieldLineTracingControl`](@ref GeneralizedPerturbedEquilibrium.FieldLineTracing.FieldLineTracingControl)
for the full field list, including the real-mode launch chord (`R_start`, `R_end`,
`Z_launch`) and the `region` selector.

## HDF5 output

Results are appended to `gpec.h5` under the `field_line_tracing/` group:

```
field_line_tracing/
├── config/            tracing_coords, tracing_field, nn
├── punctures/         R, Z, psi, theta, line_id, phi     (Poincaré points)
├── connection_length/ R, Z, psi, length, class           (0 confined, 2 escaped/wall)
├── footprints/        R, Z, phi                           (wall strike points)
└── islands/           m, n, psi, half_width, chirikov
```

The `Analysis.FieldLineTracing` submodule reads this group directly to produce the
Poincaré, connection-length, footprint, and island-width plots (see the
[Analysis Module](analysis.md)).

## Types

```@docs
GeneralizedPerturbedEquilibrium.FieldLineTracing.FieldLineTracingControl
GeneralizedPerturbedEquilibrium.FieldLineTracing.FieldLineTracingInternal
GeneralizedPerturbedEquilibrium.FieldLineTracing.FieldLineTracingState
```

## Functions

```@docs
GeneralizedPerturbedEquilibrium.FieldLineTracing.compute_field_line_tracing
GeneralizedPerturbedEquilibrium.FieldLineTracing.write_to_hdf5!
```
