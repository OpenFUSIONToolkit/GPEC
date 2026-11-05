# KineticForces code map (GPEC develop @ 4376e717)

Survey made at hand-off (2026-11-05) by a read-only agent over the clone; line numbers refer to that commit. Verify a line before relying on it if the file has changed since.

Repo root: `gpec/` (this clone). All `src/` paths below are relative to that root.


Line numbers refer to the files as read. Names are verbatim.

## 0. Findings you will trip over

1. **The torque path ignores three `[KineticForces]` keys.** `kinetic_torque` -> `integrate_psi_quadgk` (`Compute.jl:142-147`) -> `tpsi!` (`Torque.jl:32`) -> `calculate_gar` (`Torque.jl:177`) never forwards `ctrl.nutype`, `ctrl.f0type` or `ctrl.collisionality_factor`.
   - `calculate_gar` therefore runs with its defaults: `nutype="harmonic"`, `f0type="maxwellian"`, `nufac=1.0`, `qt=false`, `nlmda=128`, `ntheta=128` (`Torque.jl:184-186`).
   - Only the calculated-matrix path forwards them (`CalculatedKineticMatrices.jl:148`: `nutype=kf_ctrl.nutype, f0type=kf_ctrl.f0type, nufac=kf_ctrl.collisionality_factor`).
   - So the `nutype`/`f0type` lines in `examples/DIIID-like_ideal_example/gpec.toml:88-95` and `examples/DIIID-like_error_field_example/gpec.toml:246` are inert for the post-PE torque. They do act in a `kinetic = "KineticForces"` solve.
   - `nlmda` and `ntheta` are not config keys anywhere.
2. **The `dJdJ` column holds the normalized `wbbar`, not the physical `wb`.** The `BounceData` docstring says "ωb|δJ|²/ro²", but the code is `wbbar * abs(bj_integral)^2 / 2.0 / ro^2` (`BounceAveraging.jl:1119`). `wbbar` is `ro*2π/((2-σ)*total_wb)` (`:355`), with the thermal-speed factor `bhat` stripped. The physical dimensions are restored by the `tnorm` prefactor (`ro/bo`, `n_s`, `T_s`).
3. **`bo` is not `equil.params.b0`.** `KineticForcesInternal.bo` is the axis toroidal field `|F_spline(0)|/(2π·ro)` (`KineticForcesStructs.jl:235`). `equil.params.b0` is used only in the resonance-node and validity helpers (`Utils.jl:82,212`).
4. **No guiding-center or orbit-following code exists in the repo.** See §8.
5. **Only the λ-grid and θ-subgrid are internal.** There is no energy grid: `x` is integrated adaptively in `u=√x` over `[0, X_ENERGY_MAX=100]`. λ is integrated adaptively with `quadgk!` over a cubic spline of per-λ bounce data built on a fixed 128-point grid.
6. **The ψ coordinate is normalized, with ψ=0 at the axis** (`psilow` is the axis side). Physical-ψ derivatives divide by `chi1 = 2π·psio`.

## 1. `KineticForcesStructs.jl` (453 lines)

**`METHOD_REGISTRY` (`:16-32`).** A tuple of NamedTuples `(name, flag, kind, doc)`.
- `kind=:gar` for `fgar tgar pgar fwmm twmm pwmm ftmm ttmm ptmm fkmm tkmm pkmm frmm trmm prmm`.
- `kind=:fcgl` for `fcgl` only.
- `method_kind(name)` is at `:40`. The first letter of the method name selects the population: f full, t trapped, p passing.
- Suffix letters (`Torque.jl:244`, `method[2:4]`):
  - `gar`: scalar torque and energy.
  - `wmm`: energy from the matrices (`rex=0`).
  - `tmm`: torque from the matrices (`imx=0`).
  - `kmm` / `rmm`: matrix norms.

**`KineticForcesControl` (`:75-133`).** An immutable `@kwdef` struct, the `[KineticForces]` toml section. Deck mapping is `kinetic_forces_control(inputs)` at `Deck.jl:237`.

| Field | Default | Meaning |
|---|---|---|
| `fgar_flag` ... `prmm_flag` (16 flags) | `fgar_flag=true`, rest `false` | which methods run |
| `electron::Bool` | `false` | add electron NTV on top of the ions |
| `bounce_harmonic_max::Int` | `1` | ℓ = -max:max (the ℓ-range config key) |
| `atol_pitch` | `1e-8` | absolute tolerance, pitch integral (median-normalized kernel) |
| `rtol_pitch` | `NaN` | NaN means `psi_tolerance_margin × rtol_psi` (`torque_pitch_rtol`, `:142`) |
| `psi_tolerance_margin` | `1e-1` | |
| `atol_x`, `rtol_x` | `NaN` | NaN means derived: `nested_tolerance_margin × pitch tol` |
| `nested_tolerance_margin` | `1e-2` | |
| `rtol_psi` | `1e-2` | primary knob |
| `atol_psi` | `0.0` | N·m |
| `maxevals_psi` | `2000` | |
| `density_factor`, `temperature_factor` | `1.0` | scale n and T |
| `ExB_rotation_factor` | `1.0` | scales ω_E |
| `magnetic_drift_factor` | `1.0` | becomes `wdfac`, scaling ω_D |
| `toroidal_rotation_factor` | `1.0` | scales ω_φ = ω_E+ω_*n+ω_*T |
| `collisionality_factor` | `1.0` | matrix path only, see §0.1 |
| `divxi_factor` | `1.0` | becomes `divxfac` on ∇·ξ⊥ |
| `axis_validity_suppression` | `true` | |
| `nutype::String` | `"harmonic"` | `"zero"`, `"small"`, `"krook"`, `"harmonic"` |
| `f0type::String` | `"maxwellian"` | `"maxwellian"`, `"park"`, `"cgl"` |
| `psilims::Vector{Float64}` | `[0.0, 1.0]` | ψ integration limits |

Other items in this file:
- `MATRIX_RTOL_PITCH = 1e-3` (`:146`) and `matrix_pitch_rtol` (`:155`).
- `GeomInterp` (`:162`) is the concrete CubicSeriesInterpolant type.
- `KineticForcesInternal` (`@kwdef mutable`, `:179`). Fields:
  - `ro` (R of axis, m), `bo` (axis B_t, T), `chi1` (= 2π·psio), `mthsurf`.
  - `mfac` (poloidal mode numbers), `nlow`, `nhigh`, `npert`, `mlow`, `mhigh`, `mpert`, `numpert_total`.
  - `dbob_m` (δB/B modes over ψ), `divx_m` (∇·ξ⊥ modes over ψ), `psilim`, `sing_psis`.
  - `tpsi_xs` (θ knots, `range(0,1,mthsurf+1)`).
  - `smats tmats xmats ymats zmats` (geometric matrices), `xs_m` (`[ξ^ψ, ∂ξ^ψ/∂ψ, ξ^α]` interpolants), `method`, `verbose`.
- The constructor `KineticForcesInternal(equil; verbose)` is at `:231`.
- `set_perturbation_data!(kf_intr, pe_state, ffs, equil, metric)` (`:254`) is the bridge from PE to KF.
  - It errors unless `ffs.npert == 1` (`:258`).
  - Inputs are `pe_state.xi_modes.clebsch_psi`, `.clebsch_psi1`, `.clebsch_alpha` and `pe_state.psi_grid`.
  - Geometry comes from `ForceFreeStates.build_kinetic_metric_matrices(equil, ffs, metric)`.
  - Per ψ: `jbb_kapx = S·ξ^ψ + T·ξ^α`; `jbb_divx = X·∂ξ^ψ/∂ψ + Y·ξ^ψ + Z·ξ^α`; `jbb_dbob = -(jbb_divx + jbb_kapx)` (`:345-351`).
  - Then `_jbb_deweight!` (`:373`) divides by `J·B²` in θ space: `equil.rzphi_jac(pt)` and `equil.eqfun_B(pt)`, `pt=(psi, θ)`, θ∈[0,1).
  - The convention is δB/B = -(∇·ξ⊥ + κ·ξ⊥), so κ·ξ⊥ = -(δB/B + ∇·ξ⊥).
- `MethodResult` (`@kwdef mutable`, `:402`):
  - `method`, `nn`, `torque_profile`, `total_torque::ComplexF64` (Re = T_φ [N·m], Im = 2n·δW_k, Logan 2013 Eq. 19), `total_energy::Float64` (= Im(T)/(2n), J).
  - `psi_grid`, `dtdpsi`, `t_cumulative`, `dtdpsi_ell` (npsi × length(ell)), `ell`, `psi_nsteps`, `panel_psis`, `resonance_psis`.
- `KineticForcesState` (`:428`): `method_results::Dict{String,MethodResult}` and `kinetic_matrices::Dict{String,Array{ComplexF64,3}}` (numpert_total × numpert_total × 6).
- `KineticForcesResult` (`:447`): `total`, `species::Vector{Pair{String,KineticForcesState}}`, `axis_psi_c`, `pe`, `control`.

**Where the requested quantities live.** No struct holds them as such.
- Bounce and precession frequencies, the λ grid and the trapped/passing flag are in `BounceData` and `BounceGeometry` (`BounceAveraging.jl:42, 70`).
- ω_E, ω_*n, ω_*T and ν are NamedTuple entries from `_setup_surface_state`: `welec`, `wdian`, `wdiat`, `nuk` (`Torque.jl:467-473`), then `PitchGARParams`/`EnergyParams` fields `we`, `wn`, `wt`, `nuk`.
- There is no energy-grid or distribution-function struct. The distribution function is only `f0type` and the `f0type_id` code in `EnergyParams` (§3).

## 2. `BounceAveraging.jl` (1194 lines)

**Doc header (`:1-10`).** Computes ωb, ωd and the perturbation action |δJ|² as functions of pitch angle λ. Reference is Logan et al. 2013, following the Fortran `torque.F90` bounce loop.

### Structs

`BounceData` (`:42`):
- `nlmda::Int`.
- `lambda::Vector{Float64}`.
- `sigma::Vector{Int}`: **0 trapped, 1 passing** (`:35`).
- `wb`, `wd`: ωb(λ) and ωd(λ) in rad/s.
- `dJdJ`: real, scalar-torque path (see §0.2).
- `wmats_vs_lambda::Union{Nothing,Matrix{ComplexF64}}`: (nlmda, `nqty_matrix(mpert)`), packed. `nqty_matrix(mpert) = 3*(mpert*(mpert+1))÷2 + 3*mpert^2` (`:98`).

`BounceGeometry` (`:70`) is the harmonic-independent part:
- `nlmda lambda sigma wb wd`.
- `wbbar` (normalized ω_b, `wb = wbbar*bhat`).
- `total_wb` (full bounce-action integral).
- `fsi_wb::Matrix{Float64}` (ntheta × nlmda, cumulative bounce action along the orbit).
- `jvtheta::Matrix{ComplexF64}` (action integrand samples).
- `wmu_mt`, `wen_mt` (mpert × ntheta × nlmda, W_μ and W_E samples, matrix path only).
- `valid::Vector{Bool}`.

`BounceScratch` (`:443`) holds per-surface buffers, including `int_w` and `cumint_W`, the exact-cubic quadrature weights from `_quadrature_weights(ntheta)` (`:509`).

`SurfaceBField` (`:593`) is the B(θ) periodic-cubic decomposition. Built by `_surface_b_field(B_vpar)` (`:607`): per-cell polynomials plus stationary points from a quadratic. The bounce-point finders `_bounce_points_at_lambda!` (`:842`), `_monotone_segment_root` (`:814`) and `_cell_level_root` (`:727`) use closed-form cubic roots.

### Call chain

`compute_bounce_data` (`:245`) = `compute_bounce_geometry` (`:280`, harmonic-free) followed by `bounce_harmonic` (`:376`, applies ℓ). `calculate_gar` calls `compute_bounce_data` or `bounce_harmonic` (`Torque.jl:195-202`).

### Pitch variable

- Definition: **λ = μ·B0/E**, with B0 = `bo` (axis field), per `PitchIntegration.jl:7` and `KineticForcesStructs.jl:234`.
- `v∥/v` squared: `vpar = 1 - (λ/bo)*B(θ)` (`_vpar_from_spline`, `:571`; the same expression is inlined at `:1014`).
- Trapped/passing boundary: `lmdatpb = bo/bmax`; maximum `lmdamax = bo/bmin` (`:306-307`).
- `sigma = lmda > (bo/bmax) ? 0 : 1` (`:332`): trapped when λ > bo/bmax.
- There is no κ or Λ variable in the code. The docs page uses Λ for the same thing.

### λ grid (`_build_lambda_grid`, `:538`)

- Method letter `'t'`: trapped, `powspace(lmdatpb, lmdamax, 1, 2+nlmda, "both")`, endpoints dropped.
- `'p'`: passing, on (0, lmdatpb).
- `'f'`: `nlmda÷2` passing points (`powspace(...,2,...,"upper")`) plus the rest trapped (`"lower"`), endpoints excluded.
- `nlmda = 128` and `ntheta = 128` are hard-coded keyword defaults (`:254, :289`).

### θ sub-grid (`_find_bounce_points_and_grid`, `:882`)

- Trapped: bounce points t1, t2 are roots of vpar; `_find_deepest_well` (`:928`) picks the pair with the largest midpoint v∥. The grid is `powspace(t1,t2,4,ntheta,"both")`.
- Passing: `t1 = theta_bmax`, `t2 = theta_bmax + 1.0`, grid `powspace(...,2,...,"both")`.
- θ is normalized to [0,1] (θ_n, not 2π-scaled).

### Integrands (`_bounce_samples!`, `:970`)

Per θ sample, with `dt` the `powspace` weight and `jac = tspl[4]` (J/chi1):

- `g_wb = dt * jac * B_val / sqrt_vpar` (`:1036`).
- `g_wd = dt*jac*dBdpsi*(1.0 - 1.5*lmda*B_val/bo)/sqrt_vpar + dt*djdpsi*B_val*sqrt_vpar` (`:1037-1038`).
  - `dBdpsi = tspl[2]` is already divided by `chi1`.
  - `djdpsi = tspl[5]` is dJ/dψ/chi1².
- Mode sums: `expm[mi] = cis(twopi*mfac[mi]*θ)`; `dbob = Σ dbob_m_f[mi]*expm[mi]`; `divx = divxfac * Σ divx_m_f[mi]*expm[mi]` (`:1041-1050`).
- **Phase** (toroidal e^{-inφ} mapped to the SFL angle): `phase = cis(-twopi*n*q*(θ - theta0))`, with `theta0 = tdt_pts[1]` (`:1053`). The poloidal phase e^{2πi mθ} is `expm`.
- **Action integrand**:
  `jvtheta[i] = dt*jac*B_val*(divx*sqrt_vpar + dbob*(1.0 - 1.5*lmda*B_val/bo)/sqrt_vpar)*phase` (`:1054-1056`).
  - It uses only δB/B (`dbob`) and ∇·ξ⊥ (`divx`), both pre-built in §1. ξ itself and δB∥ never enter this function.
  - The weight for δB/B is (1 - 1.5λB/B0)/√(1-λB/B0), i.e. the (2 - 3Λ…) bounce factor. The weight for ∇·ξ⊥ is √(1-λB/B0).
- Matrix path (`do_matrices`), `:1060-1064`:
  - `wmu_mt[mi,i] = dt*(lmda/bo) * expm[mi]/sqrt_vpar * phase/(2*chi1)`
  - `wen_mt[mi,i] = dt * expm[mi]/(B_val*sqrt_vpar) * phase/(2*chi1)`
- Negative v∥ near turning points: samples are restarted or held (`:1016-1031`), with backfill at `:1070`.
- Totals use exact-cubic quadrature, not a trapezoid sum:
  - `mul!(cum_wb, scr.cumint_W, g_wb)`; `total_wb = cum_wb[ntheta]`; `total_wd = dot(scr.int_w, g_wd)` (`:1080-1083`).
  - `fsi_wb` is the cumulative bounce action.

### Frequencies (`compute_bounce_geometry`, `:324-361`)

- `bhat = sqrt(2*T_s/mass)/ro` (`:325`); `dhat = (T_s/chrg)/(bo*ro^2)` (`:326`).
- `wbbar = ro*twopi/((2 - sigma)*total_wb)` (`:355`); `wb_arr = wbbar*bhat`.
  - So ω_b = 2π / [(2-σ)∫ J B dθ/√vpar] × v_th/R0. This is 2π over the full bounce time for trapped (the factor 2) and over the transit time for passing.
- `wdbar = ro*bo*wdfac*wbbar*2*(2 - sigma)*total_wd` (`:356`); `wd_arr = wdbar*dhat`.
  - This is the bounce-averaged precession at thermal energy; the energy scaling ω_D·x happens in the resonance denominator.
  - `wdfac` is `ctrl.magnetic_drift_factor`.
- A degenerate orbit (`total_wb ≈ 0`) leaves the outputs zero with `valid=false`.

### Harmonic and bounce integral (`bounce_harmonic` `:376`, `_bounce_harmonic_tail` `:1095`)

- Effective resonance number `lnq = l + sigma*n*q` (`:393`). Passing particles get ℓ + nq, trapped get ℓ. This is the same `leff` as in `PitchIntegration.jl`.
- Bounce phase: `pl[i] = cis(-twopi*lnq*fsi_wb[i]/pl_denom)`, `pl_denom = (2 - sigma)*total_wb` (`:1108`).
- `bj_samples[i] = conj(jvtheta[i])*(pl[i] + one_minus_sigma/pl[i])` (`:1114`). Trapped gets pl + 1/pl (both legs); passing gets pl.
- `bj_integral = dot(scr.int_w, bj_samples)`.
- **`dJdJ_val = wbbar * abs(bj_integral)^2 / 2.0 / ro^2`** (`:1119`).
- Matrix path, `:1126-1190`:
  - `wmu_ba[m]` and `wen_ba[m]` are the bounce averages of `conj(W)·(pl + (1-σ)/pl)`.
  - `wxmt = wmmt * xmat`.
  - `wymt = wmmt*(3*smat + ymat) - 2.0*wemt*smat`.
  - `wzmt = wmmt*(3*tmat + zmat) - 2.0*wemt*tmat`.
  - `scale = wbbar/ro^2`.
  - Packed outer products (rank-1, `conj(w[i])*w[j]*scale`) in the order A-tri (W_Z†W_Z), D-tri (W_X†W_X), H-tri (W_Y†W_Y), B-full (W_Z†W_X), C-full (W_Z†W_Y), E-full (W_X†W_Y). These are Logan 2015 Eqs 7.30-7.35.

**Grid helper** `powspace` (`:133`) is a port of Fortran `powspace_sub`; the antiderivative is `_powspace_antideriv` (`:175`, pow 1-9).

## 3. `PitchIntegration.jl` (248 lines) and `EnergyIntegration.jl` (396 lines)

### `PitchGARParams{F}` (`PitchIntegration.jl:23`)

- `wn` (density-gradient diamagnetic drift frequency), `wt` (temperature-gradient), `we` (electric/E×B precession frequency), `nuk` (effective collision frequency).
- `bobmax` (B0/Bmax), `epsr`, `q`, `ell`, `n`, `psi`, `method`, `nutype`, `f0type`, `nufac`, `qt`.
- `energy_atol`, `energy_rtol`, `rex`, `imx`, `nqty`.
- `fbnce`: CubicSeriesInterpolant `fbnce(λ) -> [wb, wd, f₁, f₂, ...]`.
- `fbnce_hint`, `fvals`, `esegbuf`.

### `integrate_pitch_gar_quadgk` (`:~63-101`)

- Adaptive Gauss-Kronrod via `quadgk!` over λ in `[first(fbnce.cache.x), last(fbnce.cache.x)]`, split at `bobmax_clip = clamp(bobmax, ...)`.
- Output is a length-`nqty` complex vector (slot 1 is the scalar torque).
- `integrate_pitch_gar_quadgk_wt` is the dual-output variant for the matrix path: a `[wmm | tmm]` buffer of length `2*nqty`.

### Kernel `_pitch_gar_kernel_quadgk!` (`:~108-145`)

- `is_circulating = lambda <= p.bobmax`.
- `leff = is_circulating ? Float64(p.ell) + p.n*p.q : Float64(p.ell)`.
- `nueff = is_circulating ? p.nuk : p.nuk/(2*p.epsr)`.
- Circulating: `xint = integrate_energy(..., wd, +wb, ...) + integrate_energy(..., wd, -wb, ...)`, co- plus counter-passing. Trapped: a single call.
- `xint_decomposed = complex(p.rex*real(xint), p.imx*imag(xint))`.
- `out[i] = fvals[i+2] * xint_decomposed`.
- `rex/imx`: default 1,1. `wmm`/`kmm` give `rex=0`; `tmm`/`rmm` give `imx=0` (`Torque.jl:249-256`).

### `EnergyParams` (`EnergyIntegration.jl:15`)

`wn wt we wd wb nuk leff n nutype f0type nufac qt nutype_id f0type_id`.
- Note the comment on `wb` ("bounce frequency / sqrt(x)") is misleading; the denominator multiplies `wb` by `√x`.
- `_NUTYPE_IDS = Dict("zero"=>1,"small"=>2,"krook"=>3,"harmonic"=>4)` and `_F0TYPE_IDS = Dict("maxwellian"=>1,"park"=>2,"cgl"=>3)` (`:32-33`).

### Resonance denominator (`_energy_integrand_real`, `:122-134`), exact

```julia
denom = complex(-nux, p.leff * p.wb * sqx + p.n * (p.we + p.wd * x))
```

This is `iΩ(x) - ν` with Ω(x) = leff·wb·√x + n·(we + wd·x) and ν = `nux = _energy_collision_frequency(x, p)`. It matches `docs/src/conventions.md:189-200`.

Your recollection is right in form but not in the ω_E term:
- The code has `n*(we + wd*x)`, with `we` inside the n(...) bracket. The docs have `n ω_E + ℓ_eff ω_b √x + n ω_D x`, the same thing.
- Code identifiers: `wb`=ω_b, `wd`=ω_D, `we`=ω_E, `leff`=ℓ_eff.

### Collision frequency `ν(x)` (`:51-63`)

`nux` by `nutype_id`:
- `"zero"`: 0.
- `"small"`: `1e-5*abs(p.we)`.
- `"krook"`: `p.nuk`.
- `"harmonic"`: `p.nuk*(1 + 0.25*p.leff^2)/(x*sqrt(x))`, and `floatmax` at x≤0.

The result is multiplied by `p.nufac`.

### Numerator `_energy_numerator` (`:74-88`)

Without the denominator and without e^{-x}:
- `maxwellian`: `(p.we + p.wn + p.wt*(x - 1.5)) * x^2.5`.
  - Drive: ω_E + ω_*n + ω_*T(x - 3/2).
- `park`: `(p.we + p.wn + p.wt*2) * x^2.5`.
- `cgl`: `complex(0.0, -x25/p.n)` = x^2.5/(i·n), with no resonance denominator.
- `qt` (heat-flux flag, never set from config): multiplies by `(x - 2.5)`.

Full integrand: `N(x)·exp(-x)/denom(x)` (`energy_integrand_scalar`, `:142`; Logan, Park et al. 2013 Eq. 8). The x^{5/2} is the Jacobian times v⁴ weight.

### Energy variable and method

- `x = E/T`, T the species temperature in J.
- Range `[0, X_ENERGY_MAX]` with `X_ENERGY_MAX = 100.0` (`:178`).
- `integrate_energy` (`:350`) dispatches: `cgl` is integrated directly by `quadgk(..., 0.0, Inf)`; otherwise `_integrate_energy_resonant` (`:275`).
- Pole handling (Sokhotski-Plemelj):
  - `find_resonance_energies(leff, wb, n, we, wd)` (`:155`) solves `n·wd·s² + leff·wb·s + n·we = 0` for s=√x>0.
  - Each pole is `x_pole = xr - i·ν/Ω′`.
  - The residue is `R = N(xr)·exp(-xr)/(i·Ω′)`, with `Ω′ = leff*wb/(2√xr) + n*wd`.
  - The add-back is `R·[log(X_ENERGY_MAX - x_pole) - log(-x_pole)]`.
  - The smooth remainder is integrated in `u=√x` with `quadgk` (`:307`), using breakpoints at each pole and at ±2 and ±10 widths for narrow poles (`NARROW_POLE_FRACTION=0.05`, `NARROW_POLE_XMAX=25`).
  - The collisionless limit is the exact ν→0⁺ limit, with the analytic regular part from `_real_pole_regular_part` (`:232`).
- Energy tolerances: `atol_energy`/`rtol_energy` come from `tpsi!` (`Torque.jl:46-47`).

### Diagnostics

`evaluate_energy_integrand(x_grid; ...)` (`:388`) and `energy_integrand_scalar` are exported.

### ℓ range

- Config key: `bounce_harmonic_max` (default 1).
- ψ-quadrature loop: `nharm = 1 + 2*nl`, `l = ell_idx - 1 - nl` (`Compute.jl:111, 137-139`). So ℓ = -nl:nl.

## 4. `Torque.jl` (738 lines) and `Output.jl` (220 lines)

### Torque normalization (`calculate_gar`, `:177-346`)

```julia
tnorm = (-2 * n^2 / sqrt(π)) * (ro / bo) * n_s * T_s * (chi1 / twopi)      # :266-267
tpsi_val = tnorm * (lxint[1] / fbnce_norm[1])                              # :268
```

- It is Eq. 19 of Logan et al. 2013, with `chi1/twopi` converting ψ to ψ_n.
- `lxint[1]` is `∫ dλ  dJdJ(λ) · xint_decomposed`.
- The same `tnorm` appears in `kinetic_energy_matrices_for_euler_lagrange!` (`:629`).
- **The 2n·Im δW identity**: T = T_φ + 2in·δW_k.
  - `tpsi!` docstring (`:5`): "Im(T) = 2n·δW_k".
  - `Compute.jl:417-418`: `total_energy = imag(total_torque) / (2 * intr.nlow)`.
  - `Output.jl:55-56`: `mg["total_torque"] = real(result.total_torque)`; `mg["total_energy"] = result.total_energy`.
- Matrices: `energy_factor = tnorm * (1/(2*im*n))` (`:273`); the torque integrand divided by 2in becomes an energy integrand.
  - DCON normalization (`:324-326`): `op_wmats .*= 2*μ₀`; `op_wmats[:,:,1:3] ./= chi1`; `op_wmats[:,:,1] ./= chi1`.
  - Slot map: `k=1` A, `2` B, `3` C, `4` D, `5` E, `6` H (`Output._KINETIC_MATRIX_LETTERS`).
  - `kmm`/`rmm` methods return the sum of the largest singular values of `mat'*mat` over the 6 blocks (`:329-339`).
- Pre-normalization: columns of `fbnce_data` are divided by their median (`:228-238`); `lxint` is divided back by `fbnce_norm`.

### Fluid CGL (`calculate_fcgl`, `:103-154`)

Only ℓ=0. Uses `jac_val*0.5*abs(divx)^2` and `jac_val*0.5*abs(divx + 3.0*kapx)^2` with `kapx = -(dbob + divx)`. Then `T = 2n·i·n_s·T_s·(0.5·(5/3)·I1 + 0.5·(1/3)·I2)`.

### Per-surface setup (`_setup_surface_state`, `:358-474`)

- Species: electrons `chrg=-e`, `mass=me`; ions `chrg=zi*e`, `mass=mi*mp`. Constants are at `KineticForces.jl:42-45`.
- B(θ) and J sampled on `mthsurf+1` θ knots:
  - `equil.eqfun_B(pt; hint=hB)`.
  - `equil.eqfun_B(pt; deriv=DerivOp(1,0), hint=hB)/intr.chi1` (dB/dψ).
  - `equil.eqfun_B(pt; deriv=DerivOp(0,1), ...)` (dB/dθ).
  - `equil.rzphi_jac(pt)/chi1` and the dJ/dψ derivative /chi1².
- `tspl = cubic_interp(xs, Series(hcat(B_vals, dBdpsi_vals, dBdtheta_vals, jac_vals, djdpsi_vals)); bc=PeriodicBC())`. `tspl(θ)` returns `[B, dB/dψ, dB/dθ, J, dJ/dψ]`.
- `B_vpar = cubic_interp(xs, B_vals; bc=PeriodicBC())`.
- `bmax`, `bmin`, `theta_bmax` (the nodal knot of the maximum, with a refined `bmin`), via `find_sign_change_roots` on dB/dθ.
- Profiles:
  - `q = equil.profiles.q_spline(psi)`.
  - Ion: `n_s = kinetic_profiles.ni_spline(psi)`, `T_s = kinetic_profiles.Ti_spline(psi)`, `dn_s_dpsi = ni_deriv(psi)`, `dT_s_dpsi = Ti_deriv(psi)`, `nu_s = nui_spline(psi)`.
  - Electron: the `ne_`/`Te_`/`nue_` equivalents.
  - `welec = kinetic_profiles.omegaE_spline(psi)`.
- **Diamagnetic frequencies** (Logan 2013 Eq. 7):
  - `wdian = -twopi * T_s * dn_s_dpsi / (chrg * intr.chi1 * n_s)` (:452), i.e. ω_*n
  - `wdiat = -twopi * dT_s_dpsi / (chrg * intr.chi1)` (:453), i.e. ω_*T
  - `nuk = nu_s`
- `epsr = avg_r_spline(psi)/avg_R_spline(psi)` (`equil.geometry`).
- Note the PitchGARParams names `wn`, `wt`, `we` receive `wdian`, `wdiat`, `welec`.

### `_torque_surface` (`:485`) and kinetic matrices

- `_torque_surface` adds `dbob_m_f = intr.dbob_m(psi; hint=Ref(1))` and `divx_m_f = intr.divx_m(psi; ...)`, the geometric matrix blocks via `ForceFreeStates.kinetic_metric_block(intr.smats, psi, intr.mpert, k)`, and the harmonic-free `compute_bounce_geometry` (the `geometry` field).
- `kinetic_energy_matrices_for_euler_lagrange!` (`:550`) fills `kwmat` (fwmm half: rex=0, imx=1) and `ktmat` (ftmm half: rex=1, imx=0). Mirror is Hermitian in kwmat and anti-Hermitian in ktmat.
- `compute_kinetic_matrices_at_psi!` (`:703`) is the single-ψ wrapper.

### ψ quadrature (`Compute.jl`)

- `integrate_psi_quadgk` (`:77`) uses `QuadGK.BatchIntegrand`; integrand = Σ_ℓ `tpsi!` × `kinetic_axis_validity_envelope`.
- Domain `x0 = max(psi_min, equil.config.psilow, psi_c)` to `xout = min(psi_max, intr.psilim, 1.0 - 1e-6)`.
- Panels at `intr.sing_psis`, `kinetic_resonance_psi_nodes(...)` and `2*psi_c`.
- Returns `total`, `torque_profile` (`psi, dtdpsi, t_cumulative, dtdpsi_ell, ell`), `matrix_integrated`, `psi_nsteps`, `psi_quad_error`, `panel_psis`, `resonance_psis`.
- `t_cumulative` is a trapezoid cumulative integral of the sorted evaluation points.
- Per-ℓ profiles come from the first n only.

### Torque density vs ψ

`dTdpsi` is dT/dψ_N, complex. Divide by `dVdpsi` for dT/dV (`Output.jl:78` writes `dVdpsi` from `ffs.equil.profiles.dVdpsi_spline`).

### HDF5 output (`Output.jl`)

`save_to_hdf5(h5file, kf::KineticForcesResult)` (`:143`) writes into `gpec.h5`:

```
KineticForces/
  <method>/                 e.g. fgar, fcgl, ...
    n                       toroidal mode number
    total_torque            Re(T), N*m
    total_energy            dW_k = Im(T)/(2n), J
    psi_nsteps
    panel_psi               (if non-empty)
    resonance_psi           (if non-empty)
    psi                     ψ_N at quadrature points
    dTdpsi                  complex dT/dψ_N
    T                       cumulative complex T(ψ_N)
    ell, dTdpsi_ell         (npsi x length(ell))
    dVdpsi                  (if spline given)
    KineticMatrices/{A,B,C,D,E,H}   (matrix methods only)
  PerSpecies/<label>/<method>/   same layout, multi-species runs
  Validity/                 psi, rho_i, rho_banana, rho_theta, w_potato, r_minor,
                            L_p, L_q, d_separatrix, psi_c, envelope, is_valid
```

- `<label>` is `ion_z<z>_m<m>`, `impurity_z<z>_m<m>` or `electron`.
- The annotation table is `KF_METHOD_H5_ANNOTATIONS` (`:111`); the validity table is `KF_VALIDITY_H5_ANNOTATIONS` (`:201`).
- `Validity` is written by `write_validity!` (`:170`); `ForceFreeStates.write_kinetic_validity!` is extended at `:156`.
- The group name is `KineticForces`. Per `docs/development/hdf5-conventions.md:28`, a new dataset needs a table entry in the same commit or the schema test fails.

## 5. Top-level entry and config

### `kinetic_torque` (`Compute.jl:464`)

```julia
kinetic_torque(pe::PerturbedEquilibrium.PerturbedEquilibriumState, ctrl::KineticForcesControl; verbose::Bool=false) -> KineticForcesResult
```

Exported at `KineticForces.jl:55`, together with `KineticForcesResult`. The root module re-exports `kinetic_torque, KineticForcesControl, KineticForcesResult` (`GeneralizedPerturbedEquilibrium.jl:82`). Its steps:
1. `ffs = pe.response.ffs`; `equil = ffs.equil`. It errors if `equil.kinetic === nothing`.
2. `kinetic_profiles = ntv_profiles(ctrl, equil)` and `species = ntv_species(ctrl, equil)` (`Utils.jl:312, 330`).
3. `axis_psi_c = axis_validity_boundary(...)` (`Utils.jl:346`), cleared of the rational surfaces `ffs.surfaces[].psifac`.
4. `kf_intr = KineticForcesInternal(equil; verbose)`, then `set_perturbation_data!(kf_intr, pe, ffs, equil, ffs.metric)`.
5. `species === nothing`: one state through `compute_torque_all_methods!` (`:344`). Otherwise loop over `sp in species` with `sp.profiles`, `z=sp.z`, `m=sp.m`, `electron=sp.electron`, then `combine_species_states` (`:272`, sums per-species `dtdpsi` on the union ψ grid).

### Accessors used from the other modules

| Quantity | Accessor |
|---|---|
| B on (ψ,θ) | `equil.eqfun_B((psi, theta); hint=, deriv=DerivOp(i,j))` |
| Jacobian | `equil.rzphi_jac((psi, theta); ...)` |
| ψ grid, θ grid | `equil.rzphi_xs`, `equil.rzphi_ys`; PE ψ grid: `pe_state.psi_grid` |
| q | `equil.profiles.q_spline(psi)` |
| F (as 2πF) | `equil.profiles.F_spline` |
| P (as μ0P) | `equil.profiles.P_spline` |
| dV/dψ | `equil.profiles.dVdpsi_spline` |
| ε | `equil.geometry.avg_r_spline`, `equil.geometry.avg_R_spline` |
| major radius, axis | `equil.ro`, `equil.zo`, `equil.psio` |
| axis field | `equil.params.b0` (not what KF uses as `bo`) |
| mode numbers | `ffs.mlow/mhigh/mpert/nlow/nhigh/npert/numpert_total`, `ffs.psilim` |
| displacements | `pe_state.xi_modes.clebsch_psi / clebsch_psi1 / clebsch_alpha` |

- `clebsch_alpha` = ξ^α/χ₁ (regularized); `clebsch_psi1` is regularized by `Q²/(Q²+w²)` with Q = m - nq (`PerturbedEquilibrium/FieldReconstruction.jl:53-55, 285-335`).

### Config keys and deck flow

- `[KineticForces]` keys are exactly the `KineticForcesControl` field names (§1). Unknown keys error. Removed keys are listed in `Deck.jl:45-52`: `ion_species`, `ion_charge`, `ion_mass`, `impurity_*`, `nn`, `rlar_flag`, `clar_flag`, `kinetic_file`. The deprecated list is `Deck.jl:10-14`.
- The kinetic file and composition are not under `[KineticForces]`:
  - `[Equilibrium] kinetic_file`, `kinetic_group` (HDF5 group), `kinetic_format` (`"legacy"` 6-column, or `"rates-v1"` 9-column) (`Equilibrium/Config.jl:60-64, 99-101`).
  - `impurity_charge` and `impurity_mass` under `[Equilibrium]`.
  - Species list: `[[Equilibrium.ion_species]]` with `z`, `m`, and exactly one of `fraction` (share of n_i) or `density` (named `n_*` dataset). Struct `IonSpecies` at `Equilibrium/Config.jl:8`. The default is one deuterium ion with fraction 1.
- Stage flow (`GeneralizedPerturbedEquilibrium.jl:150-225`):
  - A torque stage runs when `deck.kinetic_forces !== nothing && pe !== nothing` (`:202-208`).
  - `[ForceFreeStates] kinetic = "KineticForces"` selects the self-consistent matrix source (`Deck.jl:283`); `kinetic_factor` is a uniform scale on those matrices.
  - Validation: `Deck.jl:461-464`. `[KineticForces]` without `[PerturbedEquilibrium]` and without `kinetic = "KineticForces"` errors; `kinetic_file` is required.
  - `has_ntv_fields` requires `n_e`, `T_i`, `T_e`, `omega_E`.
- `ErrorFields` also calls the torque (`[ErrorFields.NTV] method = "fgar"`; `GeneralizedPerturbedEquilibrium.jl:214`).

## 6. `Utils.jl` (398 lines) and `CalculatedKineticMatrices.jl` (199 lines)

### Species, charge, mass, profiles

- Constants are at `KineticForces.jl:42-45` (`mp, me, e`); `Equilibrium/KineticProfiles.jl:16-23` has its own copies (`_E_CHARGE=1.602e-19`, `_MP`, `_ME`).
- `main_ion(equil) -> (z, m)` (`Utils.jl:295`): the first of `equil.kinetic.ion_species`.
- `ntv_species` returns `nothing` for a single main ion with `fraction == 1.0` and no electrons. In that case the single-species path in §5 runs.
- `ResolvedNTVSpecies{P}` (`Equilibrium/KineticProfiles.jl:72`): `z, m, electron, label, profiles`.
- `resolve_ntv_species` (`:263`) builds the main ions, the quasi-neutral impurity and the optional electron.
  - It errors if profile scale factors ≠ 1 with a multi-ion list.
  - It requires every main-ion `z < zimp`.
  - Zeff and the zpitch closure come from `multi_ion_composition` (`:235`).

### `KineticProfileSplines{S,D,C}` (`Equilibrium/PlasmaEquilibrium.jl:254`)

- Value splines: `ni_spline`, `ne_spline`, `Ti_spline`, `Te_spline` (in J), `omegaE_spline` (rad/s), `loglam_spline`, `nui_spline`, `nue_spline`, `zeff_spline`.
- Derivatives: `ni_deriv`, `ne_deriv`, `Ti_deriv`, `Te_deriv`.
- Also `xs`, `npts`, `npts_minus_1`.
- Built by `kinetic_profile_splines` (`KineticProfiles.jl:~410`): pchip resample to 101 points (`_NKIN=100`), ΔT in eV×`_EV_J`.
  - Krook ν (Logan & Park 2013 Eq. 6): `nui = (zpitch/3.5e17)*zi^2*n_i*lnΛ/(sqrt(mi)*T_keV^1.5)`, and the electron analogue with `sqrt(_ME/_MP)`.
  - Supplied `nu_i`, `nu_e`, `lnLambda` (rates-v1 or HDF5) replace the model's.
- Scaling: with density/temperature/toroidal-rotation factors ≠ 1, `omegaE = toroidal_rotation_factor*wphi - wdian_new - wdiat_new`, then `ExB_rotation_factor` multiplies it.
- `shift_exb_rotation(kp, Δω)` shifts ω_E by a rigid amount.
- Raw data: `KineticProfileData` (`PlasmaEquilibrium.jl:167`), fields `n_i n_e T_i T_e omega_E omega_tor chi_* zeff lnLambda nu_i nu_e species_densities provenance ion_species impurity_charge impurity_mass ...`.
- The tearing/SLAYER path is the only other consumer of `omega_tor`.

### Utils.jl contents

- `check_inner_quadrature` (`:14`).
- `find_sign_change_roots` (`:32`): scan + Brent.
- `_resonance_nodes_from_frequencies` (`:53`) and `kinetic_resonance_psi_nodes` (`:76`):
  - Zeros of Ω_ℓ(x=2.5) per ℓ, using the closed forms `ω_b = (π/4)·√(ε/2)·√(2T/m)/(q·R0)` (`wbhat_f`, `:89`) and `ω_d = q·T/(2ε·R0²·Z·e·B0)·wdfac` (`wdhat_f`, `:90`).
  - These are for panel placement only.
  - The x=2.5 default is the peak of x^2.5·e^{-x}.
- `orbit_widths` (`:104`): `rho = √(2mT)/(|Z|e B0)`, `w_potato = (q²ρ²R0)^(1/3)`, `rho_banana = qρ/√ε`, `rho_theta = qρ/ε`.
- `kinetic_axis_validity_psi` (`:126, 164`), `kinetic_axis_validity_envelope` (`:180`, C² quintic smoothstep over [ψ_c, 2ψ_c]), `kinetic_validity_profiles` (`:202, 389`), `clear_rational_windows` (`:270`), `axis_validity_boundary` (`:346`), `resonance_grid_nodes` (`:364`).

### `CalculatedKineticMatrices.jl`

- `compute_calculated_kinetic_matrices` (`:47`): loops ψ (`Threads.@threads :static`), n, ℓ ∈ -nl:nl and species. Returns `kw_flat` and `kt_flat` of shape (mpsi, np², 6).
- The ψ loop skips where the envelope is 0, and multiplies by `env[ipsi]`.
- It uses `_setup_surface_state` and `_matrix_bounce_geometry` (`Torque.jl:529`), once per ψ and (ψ,n).
- `ForceFreeStates.kinetic_grid_inputs` (`:176`) and `ForceFreeStates.kinetic_model_source` (`:192`) hook the model into the force-balance solve.
- `ForceFreeStates/Kinetic.jl:275-277` adds the matrices to the DCON ones, `amat_kin = amat .+ kwmat[:,:,1] .+ ktmat[:,:,1]`, etc.
- The geometric matrices S,T,X,Y,Z come from `ForceFreeStates.build_kinetic_metric_matrices` (`Fourfit.jl:237`); definitions are the fmodb bands at `Fourfit.jl:307-316` and `:142-143` (y = yband1 + i·singfac²·yband2; z = zband1 + i(m²·zband2 + n·zband3)).

## 7. Examples and regression harness

### Examples that run KineticForces (`examples/`)

Verified. Names match your memory.

**`TJ_kinetic_NTV_example`** (files: `forcing.dat gpec.toml kinetic.dat kinetic_rates.dat run_example.jl`). Post-PE torque. Kinetic section:
```toml
[KineticForces]
fcgl_flag = true                        # Fluid CGL energy alongside the default fgar
axis_validity_suppression = true
```
Other keys: `[Equilibrium] kinetic_file = "kinetic.dat"`, `jac_type="hamada"`, `psilow=0.01`, `psihigh=0.99`, `mpsi=128`, `mtheta=256`. `[TJAnalytic]`: `r0=10.0, a=1.0, q0=1.6, qa=3.2, p0=0.1, B0=1.0`. `[PerturbedEquilibrium] regularization_width = 0.05`; `[ForceFreeStates] nn=1`, `integrator="Forward"`. `kinetic.dat` has columns `psi_n n_i n_e T_i[eV] T_e[eV] omega_E`.

**`TJ_kinetic_calculated_example`** (`forcing.dat gpec.toml kinetic.dat`, no `run_example.jl`). Self-consistent kinetic solve.
```toml
[[Equilibrium.ion_species]]  z=1 m=2 fraction=0.5
[[Equilibrium.ion_species]]  z=1 m=3 fraction=0.5
[ForceFreeStates]  kinetic = "KineticForces"
[PerturbedEquilibrium] compute_torque_response = true
[KineticForces]
axis_validity_suppression = true
```
`mpsi=32`.

**`TJ_kinetic_multiion_example`** (with `run_example.jl`). D-T 50/50, post-PE torque, `impurity_charge=6`, `impurity_mass=12`, `mpsi=128`.
```toml
[KineticForces]
electron = true                         # Add electron NTV in addition to the ion species
```

**`a10_kinetic_example`** (`a10_prof1.txt`, `fix_a100_k10_q2_bn010_prof1`; no PE, no forcing). Stability only, EFIT g-file.
```toml
[ForceFreeStates]  kinetic = "KineticForces"
[KineticForces]
bounce_harmonic_max = 6
electron = false
nutype = "harmonic"
f0type = "maxwellian"
atol_pitch = 1e-9
axis_validity_suppression = true
```
Also `mpsi=16`, `[[Equilibrium.ion_species]] z=1 m=2 fraction=1.0`.

**Other decks with a `[KineticForces]` section:**
- `DIIID-like_ideal_example/gpec.toml:88-95`: post-PE torque with `bounce_harmonic_max=6, electron=false, nutype="harmonic", f0type="maxwellian", atol_pitch=1e-9, axis_validity_suppression=true`. Docs cite this as the template (`docs/src/kinetic_forces.md`).
- `DIIID-like_error_field_example/gpec.toml:246`: the same keys, with `[ErrorFields.NTV] method="fgar"` and a rotation scan.

### Regression-harness cases

**Location:** `regression-harness/cases/*.toml`; runner `regression-harness/regress.jl`. 27 cases in total; `[case]` blocks name `example_dir` and optional `[overrides]`/`remove_sections`; `[quantities.*]` blocks give `h5path`, `type`, `extract`, `noise_threshold`, `order`.

**Kinetic cases:**

| Case | Example dir | Tracked HDF5 paths |
|---|---|---|
| `tj_kinetic_ntv` | `TJ_kinetic_NTV_example` | `KineticForces/fgar/total_torque` (noise 1e-8), `KineticForces/fgar/psi_nsteps`, `KineticForces/fcgl/total_energy` |
| `tj_kinetic_ntv_rates` | same, with overrides `Equilibrium.kinetic_file="kinetic_rates.dat"`, `Equilibrium.kinetic_format="rates-v1"` | `KineticForces/fgar/total_torque`, `psi_nsteps` |
| `tj_kinetic_calculated` | `TJ_kinetic_calculated_example` | `ForceFreeStates/FreeBoundaryStability/eigenmode_energies` family, `PerturbedEquilibrium/TorqueResponse/total_torque`, `PerturbedEquilibrium/Energies/toroidal_torque`, `KineticForces/fgar/total_torque` (1e-10) |
| `tj_kinetic_multiion` | `TJ_kinetic_multiion_example` | `KineticForces/fgar/total_torque`, `KineticForces/PerSpecies/ion_z1_m2/fgar/total_torque`, `.../ion_z1_m3/...`, `.../electron/...` |
| `solovev_kinetic_nuzero` | `test/test_data/regression_solovev_kinetic_nuzero` | `nutype="zero"` matrix-path anchors; overrides `Equilibrium.ion_species` (D-T) and `delta_mlow/mhigh=8` |
| `diiid_kinetic_multi_n` | `DIIID-like_ideal_example` | n=1,2 kinetic matrices; eigenmode_energies, `SingularSurfaces/Kinetic/rational_count`, `rational_psi` |
| `diiid_error_field` | `DIIID-like_error_field_example` | `ErrorFields/NTV/...` |

**Other kinetic-adjacent test data:** `test/test_data/regression_solovev_kinetic_calculated`, `regression_tj_kinetic_calculated`.

**Unit tests:**
- `test/runtests_kinetic.jl`, a single `@testset "KineticForces Unit Tests"`. Its sub-testsets include `powspace`, `energy_integrand_scalar`, `bounce-point enumeration`, `bounce geometry is harmonic-free`, `find_resonance_energies`, `integrate_energy`, `fluid CGL energy`, `KineticForcesControl defaults`, `METHOD_REGISTRY`, `psi_panel_points`, `kinetic profile I/O`, and `multi-n kinetic geometry and F̄ blocks` (line 906).
- Other files: `test/runtests_kinetic_profiles.jl`, `runtests_kinetic_validity.jl`, `runtests_multiion.jl`, `runtests_ntv_limits.jl`, `runtests_torque_response.jl`, `runtests_error_fields.jl`.
- Every `runtests_*.jl` must be listed in `test/suites.jl` (`TEST_SUITES`, checked by `test/runtests.jl:28-33`).

## 8. Equilibrium module: evaluating the field geometry

**The class `PlasmaEquilibrium`** (mutable, `Equilibrium/PlasmaEquilibrium.jl`). Geometry and field interpolants are `FastInterpolations.CubicInterpolantND` objects, called as `f((psi, theta); hint=(Ref(1),Ref(1)), deriv=DerivOp(i,j))`. ψ is normalized, θ is the SFL angle in [0,1], periodic.

| Interpolant | Content |
|---|---|
| `rzphi_rsquared` | r² = (R-ro)² + (Z-zo)² |
| `rzphi_offset` | η/(2π) - θ_new (angle offset) |
| `rzphi_nu` | ν in φ = 2πζ + ν(ψ,θ) |
| `rzphi_jac` | Jacobian |
| `eqfun_B` | \|B\| in T |
| `eqfun_metric1`, `eqfun_metric2` | (e1·e2 + q e3·e1)/(J B²) and (e2·e3 + q e3·e3)/(J B²) |

**R, Z, ψ at arbitrary (ψ,θ).** There are no named `R(ψ,θ)` or `Z(ψ,θ)` functions. They are reconstructed from the primitives, as in `Equilibrium/Derived/FluxSurfaceMetrics.jl`:
- `rzphi_primitives(equil, psi, theta; hint)` -> `(r2, deta, jac, r2_y, deta_y)` (`:9`).
- `metric_from_primitives(ro, theta, p)` (`:26-31`) -> `(rfac, r, jac, delpsi)`, with `rfac = √|r²|`, `r = ro + rfac·cos(2π(θ + deta))`, and `delpsi = |∇ψ_N|` from `w11 = (1+deta_y)(2π)² rfac r / jac`, `w12 = -r2_y π r/(rfac jac)`.
- `flux_surface_metric(equil, psi, theta)` (`:39`) -> `(r, jac, delpsi)`.
- Z: `Z = zo + rfac·sin(2π(θ + deta))`. The code at `Separatrix.jl:222-236` uses exactly `z = zo + rfac*sin(2π(θ+η))`; `Axisymmetric/Direct.jl:473` uses the geometric-angle form.

**B_φ and B_p.**
- `F_spline` stores 2πF with F = R·B_t, so `B_t = F_spline(ψ)/(2π R)`. `bo_axis = |F_spline(0)|/(2π ro)` (`KineticForcesStructs.jl:235`).
- The `eqfun_B` definition (`Direct.jl:711`) is `modB = sqrt(((2π*psio*delpsi_norm)^2 + f_val^2)/(2π r)^2)`, so `B_p = psio·|∇ψ_N|/R` with `delpsi` as above.
- There is no callable `Bφ(ψ,θ)` or `ψ(R,Z)`. `eqfun_B` gives |B| only. The toroidal-field sign is `params.bt_sign` and the current sign `params.ip_sign` (`Config.jl:301-302`).
- Poloidal flux: ψ_N is the coordinate itself; physical ψ per radian is `psio·(…)`. `chi1 = 2π·psio`, as in `ntv_profiles(...; chi1=2π*equil.psio)`.
- `equil.profiles` is `ProfileSplines`: `xs, F_spline (2πF), P_spline (μ0P), dVdpsi_spline, q_spline, F_deriv, P_deriv, dVdpsi_deriv, q_deriv`. `equil.geometry` is `GeometryProfileSplines`: `area_spline, avg_r_spline, avg_R_spline`.
- `equil.params::EquilibriumParameters` has `ro zo psio b0 q0 qmin qmax qa q95 rsep zsep bt0 bt_sign ip_sign ...` (`Config.jl:265+`).
- Related exports: `flux_surface_metric`, `flux_surface_area`, `compute_jac_delpsi`.

**Orbit or guiding-center utilities: none.** `grep -rniE "orbit|guiding|P_phi|Pphi|canonical"` over `src docs test examples CLAUDE.md` finds nothing relevant:
- "orbit" appears only in KF's own bounce-orbit comments and in orbit-**width** validity checks.
- "P_phi" appears only as SLAYER's `P_perp_model` option (`InnerLayer/SLAYER/LayerParameters.jl:125`), unrelated.
- "canonical" appears in Riccati normal-form comments, Vacuum Gauss-Legendre, and `docs/src/kinetic_forces.md:8`, which only mentions canonical angular momentum in prose.
- Bounce integrals are line integrals in the single flux coordinate θ at fixed ψ (zero-orbit-width). There is no ψ-excursion, drift tracing, P_φ or invariant bookkeeping to extend. A guiding-center module would be new code.
- Reusable primitives: `powspace`, the `SurfaceBField` bounce-point root machinery, `find_sign_change_roots`, the `_bounce_samples!` exact-cubic quadrature weights, `equil.eqfun_B` with `DerivOp` derivatives, `flux_surface_metric`.

## 9. `docs/development/architecture.md` and `CLAUDE.md`

### Adding a path or module

- **Module layout** (`architecture.md`; module list at "Module Structure", KineticForces entry item 10).
  - The architecture doc covers KineticForces in 3 lines and does not list its files. File structure is in `src/KineticForces/KineticForces.jl:27-39`, include order: `KineticForcesStructs`, `Utils`, `Output`, `BounceAveraging`, `Torque`, `EnergyIntegration`, `PitchIntegration`, `Compute`, `CalculatedKineticMatrices`.
  - Module conventions elsewhere: "Where code goes" docstring in `Equilibrium/Equilibrium.jl:1-20`.
- **Adding an NTV method:** the `METHOD_REGISTRY` docstring (`KineticForcesStructs.jl:13-14`) says to append a registry entry and add the matching `*_flag` field to `KineticForcesControl`. `method_kind` routes it (`:gar` or `:fcgl`) in `tpsi!`.
- **Adding a deck key:** every `KineticForcesControl` field is a deck key via `section_keywords`. A rename needs `_MOVED_KEYS`/`_DEPRECATED_KF_KEYS` (`Deck.jl:10-52`).
- **Deck and API parity** (`CLAUDE.md`, Important Notes): every capability must work from `gpec.toml` (via `main`) and from the scripting API (`solve`, problem/result types). Every new feature, deck key or API keyword adds a case to the `"deck and API parity"` testset in `test/runtests_solve_api.jl`. The kinetic example there is at `:115` ("calculated kinetic solve matches the TOML-driven run").
- **Docs coverage:** a new module or submodule with public docstrings needs an `@autodocs` block under `docs/src/`. `docs/src/kinetic_forces.md` already ends with `@autodocs Modules = [GeneralizedPerturbedEquilibrium.KineticForces]`. Docstrings are rendered as CommonMark, so unit annotations like `[m]` in plain text break links; use backticks.
- **Struct fields:** document them in the docstring's `## Fields` section, not inline. Do not cite PR or issue numbers in source comments.
- **Comments:** keep them concise; no "Step 1/Step 2"; cite paper equations.
- **Minimal change:** reuse FastInterpolations native `integrate`/`cumulative_integrate`/`deriv1`; do not reimplement spline ops; size the change to the problem.
- **HDF5 output:** CamelCase groups, snake_case datasets, inputs only under `Input/`, no new group names without reading `docs/development/hdf5-conventions.md`. Each writer keeps a `path => (; long_name, units, dims, scale, attach)` table next to it; a new dataset needs its table entry in the same commit. For KF that is `KF_METHOD_H5_ANNOTATIONS` in `Output.jl`.
- **TOML:** inline `# description` on every variable line, sourced from the config struct docstring; `docs/development/toml-conventions.md`.
- **Formatting:** JuliaFormatter 1.0.62 pinned by pre-commit, `margin = 180`.
- **Julia version target:** 1.11. After editing `Project.toml`, regenerate the CI pins with `julia +1.11 ci/manifests/update.jl` and `julia +1.12 ci/manifests/update.jl`.

### Tests

```bash
julia --project=. -e 'using Pkg; Pkg.activate("."); Pkg.instantiate(); include("test/runtests.jl")'
julia --project=. test/runtests.jl test/runtests_kinetic.jl     # one file (ARGS)
GPEC_TEST_GROUP=<suite> ...                                      # group from test/suites.jl
```

CLAUDE.md lists older file names such as `runtests_solovev.jl`; the actual files differ (see `ls test/`).

### Regression harness (mandatory per PR)

```bash
alias regress='julia --project=regression-harness regression-harness/regress.jl'
regress --list-cases
regress --cases diiid_n1 --refs develop,my-branch     # commit first; each ref runs in its own worktree
regress --show et_real --case solovev_n1
regress --cases solovev_n1 --ref-range develop~10..develop
```

- Fetch and fast-forward `develop` first, commit your work, and compare branch refs. `local` runs in the live checkout, so a mid-run edit silently corrupts the report.
- Cases requested in one invocation share one worktree per commit.
- Every PR body carries the `## Release note` block with the regression result and the commit it was run at.
- For a new kinetic quantity, add or extend a case under `regression-harness/cases/` (`example_dir` plus `[quantities.*]`). Suggest new cases to the user, per CLAUDE.md.
- Full reference: `docs/development/regression-harness.md`; developer-notes mirror: `docs/src/developer_notes.md:44-91`.

### Workflow constraints (CLAUDE.md)

- Branch-based GitFlow (`feature/`, `bugfix/`, ...), branch from `develop`, never commit to `develop` or `main`.
- Commit and PR title grammar: `Area[.Submodule] - TAG[!] - Imperative summary` (e.g. `KineticForces - FEAT - ...`); check with `python3 ci/conventions/check_subject.py --title "..."`.
- A PR needs an assignee and a requested human reviewer, or must be a DRAFT.
- CLAUDE.md forbids merging into `develop` without third-party human review.
- Subagent consultations: budget ≤30 tool uses and ≤10 min, stated verbatim in the prompt.

## 10. Physics-document crosswalk (equation -> code name)

| Physics | Code |
|---|---|
| T = T_φ + 2in·δW_k | `MethodResult.total_torque` (complex); Re -> `total_torque` dataset; `total_energy = imag/(2·nlow)` (`Compute.jl:418`) |
| Torque normalization (Logan 2013 Eq. 19) | `tnorm` (`Torque.jl:266-267`) |
| Energy integrand (Logan 2013 Eq. 8) | `_energy_integrand_real` (`EnergyIntegration.jl:122`), `N(x)e^{-x}/(iΩ-ν)` |
| Ω(x) | `leff*wb*√x + n*(we + wd*x)` |
| ℓ_eff | `leff` (`PitchIntegration.jl:~118`); `lnq` in `bounce_harmonic` (`BounceAveraging.jl:393`) |
| ν(x) | `_energy_collision_frequency` (`EnergyIntegration.jl:51`); trapped `nuk/(2*epsr)` |
| ω_E | `welec`/`we` = `omegaE_spline(psi)` |
| ω_*n, ω_*T | `wdian`/`wn`, `wdiat`/`wt` (`Torque.jl:452-453`) |
| ω_b | `wb_arr` = `wbbar*bhat` (`BounceAveraging.jl:359-360`) |
| ω_D | `wd_arr` = `wdbar*dhat` (`BounceAveraging.jl:361`); factor `wdfac` |
| \|δJ\|² (bounce-averaged action) | `dJdJ` = `wbbar*\|bj_integral\|²/2/ro²` |
| δJ integrand | `jvtheta` (`BounceAveraging.jl:1054`); built from `dbob_m` (δB/B) and `divx_m` (∇·ξ⊥) |
| λ = μB0/E, trapped/passing | `lambda`, `sigma`, `bobmax`; `vpar = 1 - λB/bo` |
| x = E/T | `x`; range `[0, X_ENERGY_MAX]` |
| Distribution function | `f0type` -> `f0type_id` -> numerator; no f0 struct |
| W_X, W_Y, W_Z and matrices A-H | `wxmt, wymt, wzmt` -> `wmats_lmda` -> `op_wmats[:,:,1:6]` -> `KineticMatrices/{A..H}` |
| δB/B, ∇·ξ⊥ | `set_perturbation_data!` -> `dbob_m`, `divx_m` (`KineticForcesStructs.jl:254`) |
| ψ-resolved torque density | `dTdpsi`, `dTdpsi_ell` (HDF5); `integrate_psi_quadgk` result `torque_profile` |

