"""
    FieldModel

Assembled magnetic-field model handed to the field-line integrator. Two integration
representations share this struct (selected by `coords`):

  - `"flux"`: straight-field-line `(ψ,θ,ζ)`. Field lines are integrated through the **actual
    perturbed field** via a Hamiltonian map whose background winding is `1/q(ψ)` and whose
    perturbation is the true helical flux `Ω`, built from the real radial-field harmonics
    `R_ψ,m(ψ)` of the selected source (coil Biot-Savart for `vacuum`; the reconstructed
    physical field for `total`/`plasma`). No idealized island model is used — `vacuum` opens
    real islands, `total` is ideal-screened (nested surfaces), exactly as the physics gives.
  - `"real"`: cylindrical `(R,Z,φ)`. The axisymmetric background is evaluated by inverting
    `(R,Z)→(ψ,θ)` inside the control surface; the perturbation is the exact coil
    Biot-Savart field (vacuum) or the reconstructed cylindrical perturbed field.

## Fields

  - `coords::String`: `"flux"` or `"real"`.
  - `equil`: the `Equilibrium.PlasmaEquilibrium`.
  - `nn::Int`, `mlow::Int`, `mpert::Int`: toroidal mode number and poloidal harmonic range.
  - `psio::Float64`: total poloidal flux (per radian) `= equil.psio`.
  - `helicity::Int`: `sign(Bt)·sign(Ip)`, sets the toroidal-angle sense.
  - `psi_control::Float64`: control-surface (outer) ψ_N bounding interior tracing.

Flux-mode field drive (harmonics of `dψ_N/dζ = R_ψ` and its ψ-derivative over the ψ grid):
  - `Rpsi_re/Rpsi_im`: `_series_interp` of the real/imag `R_ψ,m(ψ)` `[npsi, mpert]`.
  - `dRpsi_re/dRpsi_im`: `_series_interp` of `∂R_ψ,m/∂ψ` (finite-differenced over the ψ grid).

Real-mode perturbation:
  - `coil_sets::Vector{CoilSet}`: coil geometry for the exact real-space vacuum field.
  - `bR_re/bR_im/bZ_re/bZ_im/bphi_re/bphi_im`: cylindrical perturbed-field spectra over ψ_N
    (plasma/total source; empty when the coil field is used).
"""
mutable struct FieldModel
    coords::String
    equil::Any
    nn::Int
    mlow::Int
    mpert::Int
    psio::Float64
    helicity::Int
    psi_control::Float64

    # Toroidal harmonics carried by the flux drive. The vacuum coil field is fully 3D, so
    # several n are kept (e.g. n=1,2,3) to capture island bifurcation — at q=m/n both (2,1)
    # and (4,2) resonate. total/plasma are single-n (the perturbed equilibrium is one n).
    n_vals::Vector{Int}
    Rpsi_re::Any
    Rpsi_im::Any
    dRpsi_re::Any
    dRpsi_im::Any

    coil_sets::Vector{ForcingTerms.CoilSet}
    bR_re::Any
    bR_im::Any
    bZ_re::Any
    bZ_im::Any
    bphi_re::Any
    bphi_im::Any
end

"""
    _series_interp(psi_grid, mat) -> CubicSeriesInterpolant

Cubic interpolant of the columns of a real `[npsi, mpert]` matrix over the ψ grid,
returning an `mpert`-vector when evaluated. Extends flat past the grid ends.
"""
_series_interp(psi_grid::AbstractVector{Float64}, mat::AbstractMatrix{Float64}) =
    cubic_interp(collect(Float64, psi_grid), Series(Matrix(mat)); extrap=ExtendExtrap())

# --------------------------------------------------------------------------------------
# Flux-coordinate field drive: project the actual perturbed field onto the radial
# harmonics R_ψ,m(ψ), then integrate the Hamiltonian map. R_ψ = (B_pert·∇ψ_N)/(B0·∇ζ)
# with B0·∇ζ = F/(2πR)² and B_pert·∇ψ_N = |∇ψ_N|·(B_pert·n̂). n̂ is the outward flux-surface
# normal (∇ψ direction), obtained by rotating the poloidal tangent ∂(R,Z)/∂θ by −90°.
# --------------------------------------------------------------------------------------

"""
    _flux_surface_geometry(equil, psi_grid, theta_grid) -> named arrays

Per `(ψ_i, θ_j)` geometry used by the radial projection: cylindrical `(R,Z)`, the poloidal
tangent `(tR,tZ)=∂(R,Z)/∂θ`, the projection prefactor `pref = |∇ψ_N|/|t| · (2π)²R²/F_spline`
(so `R_ψ = pref·(B_R·tZ − B_Z·tR)`), and the toroidal-angle offset `ν`.
"""
function _flux_surface_geometry(equil, psi_grid::AbstractVector{Float64},
                                theta_grid::AbstractVector{Float64})
    np = length(psi_grid); nt = length(theta_grid)
    R = zeros(np, nt); Z = zeros(np, nt); tR = zeros(np, nt); tZ = zeros(np, nt)
    pref = zeros(np, nt); nu = zeros(np, nt)
    dt = 1e-6
    for i in 1:np
        psi = psi_grid[i]
        Fspline = equil.profiles.F_spline(psi)
        for j in 1:nt
            th = theta_grid[j]
            Rc, Zc = _rz_at(equil, psi, th)
            R1, Z1 = _rz_at(equil, psi, th + dt)
            R0, Z0 = _rz_at(equil, psi, th - dt)
            τR = (R1 - R0) / (2dt); τZ = (Z1 - Z0) / (2dt)
            τn = sqrt(τR^2 + τZ^2); τn < 1e-12 && (τn = 1.0)
            m = Equilibrium.flux_surface_metric(equil, psi, th)
            R[i, j] = Rc; Z[i, j] = Zc; tR[i, j] = τR; tZ[i, j] = τZ
            pref[i, j] = m.delpsi / τn * (2π)^2 * Rc^2 / Fspline
            nu[i, j] = equil.rzphi_nu((psi, th))
        end
    end
    return (; R, Z, tR, tZ, pref, nu)
end

"""
    _radial_harmonics(geom, BR, BZ, n_vals, mlow, mpert, theta_grid, zeta_grid) -> Matrix{ComplexF64}

Project the perturbed field `(BR,BZ)` (arrays `[np, nt, nz]`) onto the normalized radial
drive `R_ψ` and Fourier-decompose over `(θ,ζ)` to the complex harmonics `R_ψ,{m,n}(ψ)` for
every `n` in `n_vals`, in the convention `R_ψ(θ,ζ) = Re[Σ_{m,n} R_ψ,{m,n} e^{i·2π(mθ − nζ)}]`.
Returned as `[np, mpert·length(n_vals)]`, column `(nidx-1)·mpert + ip`.
"""
function _radial_harmonics(geom, BR::Array{Float64,3}, BZ::Array{Float64,3},
                           n_vals::Vector{Int}, mlow::Int, mpert::Int,
                           theta_grid::AbstractVector{Float64}, zeta_grid::AbstractVector{Float64})
    np, nt, nz = size(BR)
    nv = length(n_vals)
    Rpsi_m = zeros(ComplexF64, np, mpert * nv)
    scale = 2.0 / (nt * nz)
    # Precompute the projected radial drive R_ψ[i,j,k] once.
    Rψ = Array{Float64,3}(undef, np, nt, nz)
    @inbounds for i in 1:np, j in 1:nt, k in 1:nz
        Rψ[i, j, k] = geom.pref[i, j] * (BR[i, j, k] * geom.tZ[i, j] - BZ[i, j, k] * geom.tR[i, j])
    end
    @inbounds for i in 1:np
        for (nidx, nv_) in enumerate(n_vals)
            for ip in 1:mpert
                m = mlow + ip - 1
                acc = 0.0 + 0.0im
                for j in 1:nt
                    for k in 1:nz
                        acc += Rψ[i, j, k] * cis(-2π * (m * theta_grid[j] - nv_ * zeta_grid[k]))
                    end
                end
                Rpsi_m[i, (nidx - 1) * mpert + ip] = scale * acc
            end
        end
    end
    return Rpsi_m
end

"""
    build_field_drive_flux_model(equil, nn, mlow, mpert, psi_control, helicity, psi_lo, psi_hi;
                                 coil_sets, b_modes, pe_psi_grid, npsi, ntheta, nzeta)

Construct a flux-coordinate `FieldModel` that integrates the actual perturbed field. Samples
the source field on a `(ψ,θ,ζ)` grid over `[psi_lo, psi_hi]`, projects it to the radial
harmonics `R_ψ,m(ψ)`, and stores value/derivative interpolants for the Hamiltonian map.

The source is the exact coil Biot-Savart field when `coil_sets` is non-empty (`vacuum`),
otherwise the reconstructed cylindrical field `b_modes.R/.Z` on `pe_psi_grid` (`total`/`plasma`).
"""
function build_field_drive_flux_model(equil, nn::Int, mlow::Int, mpert::Int,
                                      psi_control::Float64, helicity::Int,
                                      psi_lo::Float64, psi_hi::Float64;
                                      n_vals::Vector{Int}=[nn],
                                      coil_sets::Vector{ForcingTerms.CoilSet}=ForcingTerms.CoilSet[],
                                      b_modes=nothing, pe_psi_grid::Vector{Float64}=Float64[],
                                      npsi::Int=0, ntheta::Int=0, nzeta::Int=0)
    npsi == 0 && (npsi = clamp(round(Int, (psi_hi - psi_lo) / 0.002), 60, 400))
    ntheta == 0 && (ntheta = max(256, 4 * mpert))
    nzeta == 0 && (nzeta = max(16, 4 * maximum(n_vals)))
    psi_grid = collect(range(psi_lo, psi_hi; length=npsi))
    theta_grid = collect(range(0.0; length=ntheta, step=1.0 / ntheta))
    zeta_grid = collect(range(0.0; length=nzeta, step=1.0 / nzeta))
    geom = _flux_surface_geometry(equil, psi_grid, theta_grid)

    BR = zeros(npsi, ntheta, nzeta)
    BZ = zeros(npsi, ntheta, nzeta)
    if !isempty(coil_sets)
        # Batch all observation points through one threaded Biot-Savart call.
        nobs = npsi * ntheta * nzeta
        oR = zeros(nobs); ophi = zeros(nobs); oZ = zeros(nobs)
        idx = 0
        for k in 1:nzeta, j in 1:ntheta, i in 1:npsi
            idx += 1
            oR[idx] = geom.R[i, j]; oZ[idx] = geom.Z[i, j]
            # Machine toroidal angle (matches the forcing convention: φ = −helicity(2πζ + ν)).
            ophi[idx] = -helicity * (2π * zeta_grid[k] + geom.nu[i, j])
        end
        bR = zeros(nobs); bphi = zeros(nobs); bZ = zeros(nobs)
        ForcingTerms.compute_biot_savart_boundary!(bR, bphi, bZ, oR, ophi, oZ, coil_sets)
        idx = 0
        for k in 1:nzeta, j in 1:ntheta, i in 1:npsi
            idx += 1
            BR[i, j, k] = bR[idx]; BZ[i, j, k] = bZ[idx]
        end
    else
        # Reconstructed cylindrical field (total/plasma) — SFL harmonics interpolated over ψ.
        bR_re = _series_interp(pe_psi_grid, real.(b_modes.R))
        bR_im = _series_interp(pe_psi_grid, imag.(b_modes.R))
        bZ_re = _series_interp(pe_psi_grid, real.(b_modes.Z))
        bZ_im = _series_interp(pe_psi_grid, imag.(b_modes.Z))
        for i in 1:npsi
            psi = psi_grid[i]
            rr = bR_re(psi); ri = bR_im(psi); zr = bZ_re(psi); zi = bZ_im(psi)
            for j in 1:ntheta, k in 1:nzeta
                BR[i, j, k] = _mode_sum(rr, ri, theta_grid[j], zeta_grid[k], mlow, nn)
                BZ[i, j, k] = _mode_sum(zr, zi, theta_grid[j], zeta_grid[k], mlow, nn)
            end
        end
    end

    Rpsi_m = _radial_harmonics(geom, BR, BZ, n_vals, mlow, mpert, theta_grid, zeta_grid)
    # ψ-derivative of each harmonic (all m·n columns) by central differences on the ψ grid.
    dRpsi_m = similar(Rpsi_m)
    @inbounds for col in axes(Rpsi_m, 2)
        for i in 1:npsi
            lo = max(i - 1, 1); hi = min(i + 1, npsi)
            dRpsi_m[i, col] = (Rpsi_m[hi, col] - Rpsi_m[lo, col]) / (psi_grid[hi] - psi_grid[lo])
        end
    end

    return FieldModel("flux", equil, nn, mlow, mpert, equil.psio, helicity, psi_control,
                      copy(n_vals),
                      _series_interp(psi_grid, real.(Rpsi_m)), _series_interp(psi_grid, imag.(Rpsi_m)),
                      _series_interp(psi_grid, real.(dRpsi_m)), _series_interp(psi_grid, imag.(dRpsi_m)),
                      ForcingTerms.CoilSet[], nothing, nothing, nothing, nothing, nothing, nothing)
end

"""
    build_real_field_model(equil, nn, mlow, mpert, psi_control, helicity;
                           coil_sets, psi_grid, b_modes)

Construct a real-space `(R,Z,φ)` `FieldModel`. When `coil_sets` is non-empty the
perturbation is the exact coil Biot-Savart field (vacuum source). Otherwise the
reconstructed cylindrical perturbed spectra (`b_modes.R/Z/phi`, flagged beta) are used.
"""
function build_real_field_model(equil, nn::Int, mlow::Int, mpert::Int,
                                psi_control::Float64, helicity::Int;
                                coil_sets::Vector{ForcingTerms.CoilSet}=ForcingTerms.CoilSet[],
                                psi_grid::Vector{Float64}=Float64[],
                                b_modes=nothing)
    use_recon = isempty(coil_sets) && b_modes !== nothing && !isempty(psi_grid)
    return FieldModel("real", equil, nn, mlow, mpert, equil.psio, helicity, psi_control,
                      Int[], nothing, nothing, nothing, nothing,
                      coil_sets,
                      use_recon ? _series_interp(psi_grid, real.(b_modes.R)) : nothing,
                      use_recon ? _series_interp(psi_grid, imag.(b_modes.R)) : nothing,
                      use_recon ? _series_interp(psi_grid, real.(b_modes.Z)) : nothing,
                      use_recon ? _series_interp(psi_grid, imag.(b_modes.Z)) : nothing,
                      use_recon ? _series_interp(psi_grid, real.(b_modes.phi)) : nothing,
                      use_recon ? _series_interp(psi_grid, imag.(b_modes.phi)) : nothing)
end

"""
    fieldline_rhs_flux!(du, u, model, zeta)

Straight-field-line Hamiltonian map, `u = [ψ_N, θ]`, independent variable `ζ`, integrating
the actual perturbed field:

    dψ_N/dζ = Re[Σ_{m,n} R_ψ,{m,n}(ψ) e^{i·2π(mθ − nζ)}]                  (m ≠ 0)
    dθ/dζ   = 1/q(ψ) + Re[Σ_{m,n} (i/(2πm)) ∂R_ψ,{m,n}/∂ψ e^{i·2π(mθ − nζ)}]

summed over the toroidal harmonics `n ∈ model.n_vals` (`vacuum` keeps `n=1…flux_n_max`; the
coil field is fully 3D). The second term is the ψ-derivative of the helical flux
`Ω_{m,n} = i R_ψ,{m,n}/(2πm)`, so the map is the symplectic flow of `H = ∫dψ/q + Ω`. Islands
appear at `n·q = m`; several `n` resonating on one surface (e.g. `(2,1)` and `(4,2)` at `q=2`)
reproduce island bifurcation. Ideal screening (`R_ψ,res → 0`) leaves nested surfaces.
"""
function fieldline_rhs_flux!(du, u, model::FieldModel, zeta)
    psi = clamp(u[1], 1e-6, model.psi_control)
    theta = u[2]
    q = model.equil.profiles.q_spline(psi)

    Rre = model.Rpsi_re(psi); Rim = model.Rpsi_im(psi)
    dRre = model.dRpsi_re(psi); dRim = model.dRpsi_im(psi)

    dpsi = 0.0
    dtheta = 1.0 / q
    mpert = model.mpert
    @inbounds for (nidx, nv) in enumerate(model.n_vals)
        off = (nidx - 1) * mpert
        for ip in 1:mpert
            m = model.mlow + ip - 1
            m == 0 && continue
            col = off + ip
            ph = cis(2π * (m * theta - nv * zeta))
            dpsi += real((Rre[col] + im * Rim[col]) * ph)
            dtheta += real((im / (2π * m)) * (dRre[col] + im * dRim[col]) * ph)
        end
    end
    du[1] = dpsi
    du[2] = dtheta
    return nothing
end

# --------------------------------------------------------------------------------------
# Real-space cylindrical field-line map.
# --------------------------------------------------------------------------------------
@inline function _mode_sum(coeff_re::AbstractVector, coeff_im::AbstractVector,
                           theta::Float64, zeta::Float64, mlow::Int, nn::Int)
    acc = 0.0 + 0.0im
    @inbounds for i in eachindex(coeff_re)
        m = mlow + i - 1
        acc += (coeff_re[i] + im * coeff_im[i]) * cis(2π * m * theta)
    end
    return real(acc * cis(-2π * nn * zeta))
end

"""
    invert_rz_to_flux(equil, R, Z; psi_max) -> (psi, theta, converged)

Recover `(ψ_N, θ)` from `(R,Z)` by Newton iteration on the forward `rzphi` map. Returns
`converged=false` when the point lies outside the mapped flux domain (e.g. beyond the LCFS).
"""
function invert_rz_to_flux(equil, R::Float64, Z::Float64; psi_max::Float64=1.0)
    ro, zo = equil.ro, equil.zo
    dR = R - ro
    dZ = Z - zo
    r_target = sqrt(dR^2 + dZ^2)
    r_target < 1e-9 && return (0.0, 0.0, true)
    eta_target = atan(dZ, dR)

    psi = clamp((r_target / _boundary_minor(equil, eta_target, psi_max))^2, 1e-5, psi_max)
    theta = mod(eta_target / (2π), 1.0)

    for _ in 1:40
        Rc, Zc = _rz_at(equil, psi, theta)
        fR = Rc - R
        fZ = Zc - Z
        (fR^2 + fZ^2 < 1e-16) && return (psi, mod(theta, 1.0), true)
        dp = 1e-6
        dt = 1e-6
        Rp, Zp = _rz_at(equil, min(psi + dp, psi_max), theta)
        Rt, Zt = _rz_at(equil, psi, theta + dt)
        J11 = (Rp - Rc) / dp; J12 = (Rt - Rc) / dt
        J21 = (Zp - Zc) / dp; J22 = (Zt - Zc) / dt
        det = J11 * J22 - J12 * J21
        abs(det) < 1e-20 && break
        dpsi = -(J22 * fR - J12 * fZ) / det
        dtheta = -(-J21 * fR + J11 * fZ) / det
        psi = clamp(psi + dpsi, 1e-5, psi_max)
        theta = theta + dtheta
    end
    Rc, Zc = _rz_at(equil, psi, mod(theta, 1.0))
    conv = (Rc - R)^2 + (Zc - Z)^2 < 1e-8
    return (psi, mod(theta, 1.0), conv)
end

@inline function _rz_at(equil, psi::Float64, theta::Float64)
    rfac = sqrt(abs(equil.rzphi_rsquared((psi, theta))))
    eta = 2π * (theta + equil.rzphi_offset((psi, theta)))
    return equil.ro + rfac * cos(eta), equil.zo + rfac * sin(eta)
end

@inline function _boundary_minor(equil, eta::Float64, psi_max::Float64)
    theta = mod(eta / (2π), 1.0)
    return max(sqrt(abs(equil.rzphi_rsquared((psi_max, theta)))), 1e-3)
end

"""
    equilibrium_field_real(model, R, Z) -> (BR, BZ, Bphi, psi, inside)

Axisymmetric equilibrium field at `(R,Z)`. Inside the control surface the poloidal field is
`|∇ψ|/R` along the flux-surface tangent and `Bφ = F/R`; outside, only the vacuum toroidal
field `Bφ = F_edge/R` is returned (`inside=false`).
"""
function equilibrium_field_real(model::FieldModel, R::Float64, Z::Float64)
    equil = model.equil
    psi, theta, conv = invert_rz_to_flux(equil, R, Z; psi_max=model.psi_control)
    Fedge = equil.profiles.F_spline(model.psi_control) / (2π)
    if !conv || psi >= model.psi_control
        return (0.0, 0.0, Fedge / R, psi, false)
    end
    metric = Equilibrium.flux_surface_metric(equil, psi, theta)
    Bpol = model.psio * metric.delpsi / R
    dt = 1e-6
    R1, Z1 = _rz_at(equil, psi, theta + dt)
    R0, Z0 = _rz_at(equil, psi, theta - dt)
    tR = R1 - R0; tZ = Z1 - Z0
    tnorm = sqrt(tR^2 + tZ^2)
    tnorm < 1e-12 && (tnorm = 1.0)
    sgn = float(model.helicity)
    BR = Bpol * sgn * tR / tnorm
    BZ = Bpol * sgn * tZ / tnorm
    F = equil.profiles.F_spline(psi) / (2π)
    return (BR, BZ, F / R, psi, true)
end

"""
    perturbation_field_real(model, R, Z, phi) -> (dBR, dBZ, dBphi)

Perturbed cylindrical field at `(R,φ,Z)`. Exact coil Biot-Savart when coil sets are present
(vacuum source); otherwise the reconstructed cylindrical spectra at the inverted `(ψ,θ)`.
"""
function perturbation_field_real(model::FieldModel, R::Float64, Z::Float64, phi::Float64)
    if !isempty(model.coil_sets)
        oR = [R]; ophi = [phi]; oZ = [Z]
        BR = [0.0]; Bphi = [0.0]; BZ = [0.0]
        ForcingTerms.compute_biot_savart_boundary!(BR, Bphi, BZ, oR, ophi, oZ, model.coil_sets)
        return (BR[1], BZ[1], Bphi[1])
    end
    # Reconstructed-spectra branch (plasma/total real mode) — beta: the b_modes.R/Z/phi
    # spectra are assumed built with the same e^{i(mθ−nφ)} convention as FieldReconstruction.
    model.bR_re === nothing && return (0.0, 0.0, 0.0)
    psi, theta, conv = invert_rz_to_flux(model.equil, R, Z; psi_max=model.psi_control)
    !conv && return (0.0, 0.0, 0.0)
    zeta = phi / (2π)
    dBR = _mode_sum(model.bR_re(psi), model.bR_im(psi), theta, zeta, model.mlow, model.nn)
    dBZ = _mode_sum(model.bZ_re(psi), model.bZ_im(psi), theta, zeta, model.mlow, model.nn)
    dBphi = _mode_sum(model.bphi_re(psi), model.bphi_im(psi), theta, zeta, model.mlow, model.nn)
    return (dBR, dBZ, dBphi)
end

"""
    fieldline_rhs_real!(du, u, model, phi)

Cylindrical field-line map, `u = [R, Z]`, independent variable `φ`:
`dR/dφ = R·B_R/B_φ`, `dZ/dφ = R·B_Z/B_φ`, with `B = B_equilibrium + B_perturbation`.
"""
function fieldline_rhs_real!(du, u, model::FieldModel, phi)
    R = u[1]; Z = u[2]
    BR, BZ, Bphi, _, _ = equilibrium_field_real(model, R, Z)
    dBR, dBZ, dBphi = perturbation_field_real(model, R, Z, phi)
    BR += dBR; BZ += dBZ; Bphi += dBphi
    abs(Bphi) < 1e-12 && (Bphi = 1e-12)
    du[1] = R * BR / Bphi
    du[2] = R * BZ / Bphi
    return nothing
end
