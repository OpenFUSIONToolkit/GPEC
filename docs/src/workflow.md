# Workflow

GPEC follows a five-stage analysis pipeline driven by a single `gpec.toml` configuration file. Each stage produces structured output consumed by the next.

```
                              gpec.toml
                  (user options for every module)
                                 │
        ┌────────────┬───────────┴───┬─────────────────────┐
        ▼            ▼               ▼                     ▼
   Equilibrium ──► Vacuum ──► ForceFreeStates ──► PerturbedEquilibrium ──► gpec.h5
        ▲            │                                    ▲   ▲
        │            └────────────────────────────────────┘   │
        │                                                     │
   equilibrium file                                      ForcingTerms
   kinetic profiles (planned)                  (external perturbation file)
```

The single `gpec.toml` file supplies user-selected options to every module. The two primary input files — the equilibrium file and (planned) the kinetic profiles file — are read by the `Equilibrium` module, while the external perturbation specification enters at `PerturbedEquilibrium` via `ForcingTerms`. `Vacuum` feeds both `ForceFreeStates` and `PerturbedEquilibrium` directly: the response matrices it builds are consumed by both downstream stages.

---

## Stage 1: Equilibrium

**Module**: `Equilibrium`

**Purpose**: Reconstruct the axisymmetric MHD equilibrium. Solves the Grad-Shafranov equation on a poloidal flux grid, computes field-aligned coordinates, and builds bicubic splines for all field quantities used by downstream modules.

**Configuration**: `[Equilibrium]` section of `gpec.toml`

**Inputs**:
- Equilibrium data file in one of the supported formats:
  - `efit` — EFIT g-file from experimental reconstruction
  - `chease` / `chease2` — CHEASE equilibrium code output
  - `lar` — Large aspect ratio analytical model
  - `sol` — Solov'ev analytical equilibrium
- Kinetic profiles file (planned) — temperature and density profiles for the kinetic analysis path
- Grid resolution and solver settings

**Outputs** (`PlasmaEquilibrium`):
- Bicubic splines for R(ψ,θ), Z(ψ,θ), and all field quantities
- Safety factor profile q(ψ) and singular surface locations
- Global parameters: magnetic axis (R₀, Z₀), β, q₀, q_a, plasma volume
- Flux surface geometry and separatrix location

---

## Stage 2: Vacuum Response

**Module**: `Vacuum`

**Purpose**: Compute the magnetostatic vacuum response matrices that couple the plasma boundary to external fields. These matrices are used by ForceFreeStates and PerturbedEquilibrium to apply boundary conditions and decompose fields in poloidal mode space.

**Configuration**: `[Wall]` section of `gpec.toml`

**Inputs**:
- `PlasmaEquilibrium` from Stage 1 (plasma boundary coordinates, toroidal mode number n)
- Wall geometry: shape (conformal, elliptical, dee-shaped, or custom) and dimensions

**Outputs**:
- `wv` — Vacuum response matrix (scaled by the singular factor (m - nq)(m' - nq), see Chance 1997)
- `I_v` — Vacuum surface-current matrix Iᵛ when `compute_Iv=true` (otherwise zeros); PerturbedEquilibrium inverts this to surface inductance `L`

**Key references**: [Chance et al. (1997)](citations.md#Vacuum-Module), [Chance et al. (2007)](citations.md#Vacuum-Module)

---

## Stage 3: Ideal MHD Stability

**Module**: `ForceFreeStates`

**Purpose**: Determine ideal MHD stability via Newcomb's direct criterion. Integrates the Euler-Lagrange equations across the plasma volume, crossing singular surfaces (where q = m/n) with prescribed jump conditions. Computes the potential and kinetic energy matrices that define the eigenvalue problem for the free-boundary stability.

**Configuration**: `[ForceFreeStates]` section of `gpec.toml`

**Inputs**:
- `PlasmaEquilibrium` from Stage 1
- Vacuum response matrices from Stage 2
- Poloidal mode range (m_low, m_pert) and toroidal mode number n

**Outputs**:
- Force-free eigenfunctions ξ(ψ, θ) — the normal displacement field
- Fixed boundary energy W_fixed (negative → unstable without wall)
- Free boundary energy eigenvalues `et` (sign determines ideal stability with wall)
- Mercier stability criterion as a function of ψ
- Singular surface data at each rational surface q = m/n:
  - Flux surface location ψ_s
  - Tearing stability parameter Δ'
  - Small solution coefficients (used by PerturbedEquilibrium)

**Key references**: [Glasser (2016) Newcomb](citations.md#ForceFreeStates-Module), [Glasser (2018) Riccati](citations.md#ForceFreeStates-Module)

---

## Stage 4: External Forcing

**Module**: `ForcingTerms`

**Purpose**: Load the external magnetic perturbation specification — the amplitude and phase of each (m, n) Fourier component of the applied field at the plasma boundary. This is typically computed externally (e.g. from a coil geometry code or measured from sensors).

**Configuration**: `[ForcingTerms]` section of `gpec.toml`

**Inputs**:
- External perturbation data file (ASCII or HDF5 format)

**Outputs** (array of `ForcingMode`):
- For each (m, n) mode: poloidal mode number, toroidal mode number, complex amplitude (magnitude + phase)

---

## Stage 5: Perturbed Equilibrium

**Module**: `PerturbedEquilibrium`

**Purpose**: Compute the self-consistent plasma response to the external perturbation specified in Stage 4. Uses the force-free eigenfunctions and vacuum matrices to project the response into mode space, then evaluates singular coupling diagnostics at each rational surface.

**Configuration**: `[PerturbedEquilibrium]` section of `gpec.toml`

**Inputs**:
- ForceFreeStates results from Stage 3 (eigenfunctions, singular surface data)
- Vacuum response matrices from Stage 2
- `ForcingMode` array from Stage 4

**Outputs** (`PerturbedEquilibriumState`):
- Mode-space normal displacement ξ(m, ψ) across the plasma
- Mode-space magnetic perturbation b(m, ψ)
- At each rational surface q = m_s/n:
  - Resonant flux δψ_s
  - Resonant current sheet amplitude
  - Island half-width w_s (proportional to √|δψ_s|)
  - Chirikov overlap parameter σ (ratio of adjacent island widths to their separation)

**Key references**: [Park et al. (2007a)](citations.md#PerturbedEquilibrium-Module), [Park et al. (2009)](citations.md#PerturbedEquilibrium-Module), [Park et al. (2017)](citations.md#Kinetic-Forces)

---

## Configuration File: `gpec.toml`

All stages are controlled by a single TOML file. A minimal example structure:

```toml
[Equilibrium]
eq_type = "efit"
eq_filename = "g-file.txt"

[Wall]
shape = "conformal"
a = 0.3

[ForceFreeStates]
mlow = -3
mpert = 7
n = 1
force_termination = false

[PerturbedEquilibrium]
output_file = "gpec.h5"

[ForcingTerms]
forcing_filename = "coil_fields.h5"
```

Setting `force_termination = true` in any section stops the pipeline after that stage (useful for equilibrium-only or stability-only runs).

Example configuration files are provided in:
- `examples/Solov'ev_ideal_example/gpec.toml`
- `examples/DIIID-like_ideal_example/gpec.toml`

---

## Output File: `gpec.h5`

All results are written to a single HDF5 file (default: `gpec.h5`). The top-level groups are organized by physics topic (see the schema conventions in `docs/development/hdf5-conventions.md`):

| Group | Contents |
|---|---|
| `Info/` | Run metadata: git version, mode-number ranges, ψ limit, `Runtimes/` (per-stage wall-clock seconds) |
| `Input/` | Self-contained rerun snapshot: merged TOML blob, raw equilibrium/forcing/coil inputs |
| `Equilibrium/` | Equilibrium scalars (`beta_N`, `q_axis`, `q_95`, …), 1-D profiles (`Profiles/`), 2-D geometry (`Geometry/`) |
| `ForceFreeStates/` | Stability solve: `Solutions/{ForwardIntegration,GalerkinIntegration}`, `EulerLagrangeMatrices/`, `FreeBoundaryStability/`, `EdgeScan/` |
| `LocalStability/` | Mercier D_I, resistive interchange D_R, ballooning Δ' profiles |
| `SingularSurfaces/` | Per-rational-surface data: ψ_s, q, m/n, GGJ coefficients, Δ' matrices (`GalerkinDeltaPrime/`), kinetic surfaces (`Kinetic/`) |
| `PerturbedEquilibrium/` | Plasma response: `ForcingModes/`, `Response/`, `ResponseMatrices/`, `SingularCoupling/`, `Energies/` |
| `KineticForces/` | NTV torque per method: energy integrals, kinetic matrices |
| `Tearing/` | SLAYER/GGJ inner-layer growth rates: `PerSurface/`, `Roots/`, `LayerWidths/`, `Diagnostics/`, `Scan/` |
| `SurfaceGeometries/` | Plasma and wall surface point clouds for visualization |
