# GGJSpec.jl
#
# Builds the model-agnostic `WasowGalerkin.SystemSpec` for the single-fluid
# Glasser–Greene–Johnson inner-layer system. This file holds the GGJ-specific
# physics that was previously inlined in `InnerAsymptotics.jl` and `Galerkin.jl`:
# the asymptotic coefficient matrices `A₀/A₁/A₂` (Glasser & Wang 2020, Eqs. 4–5),
# the block-diagonalizing basis `T/Tinv` (Eqs. 7–8), the Mercier Frobenius
# exponents (Eq. 49), the FEM coefficient matrices `(I, U, V)` (inpso_get_uv),
# and the map from the first-order asymptotic state to the physical `(W, N, Θ)`
# variables (inpso_get_ua/dua). The shared engine consumes these via `SystemSpec`.

using StaticArrays
import ..WasowGalerkin: SystemSpec, ParityBC, BCKind, DIRICHLET, NEUMANN
import ..WasowGalerkin: WasowCache, build_wasow, evaluate_wasow, wasow_residual, pick_xmax, solve_galerkin
import ..WasowGalerkin: solve_galerkin_full, FullDomainMatching
import ..InnerLayerModel, ..solve_inner, ..solve_inner_full

# Asymptotic coefficient matrices A₀, A₁, A₂ (Glasser & Wang 2020 Eqs. 4–5).
# Port of the A-matrix block of inps_tjmat (inps.f lines 188–202).
function _ggj_asymptotic_matrices(p::GGJParameters, Q::ComplexF64)
    e = ComplexF64(p.E)
    f = ComplexF64(p.F)
    h = ComplexF64(p.H)
    g = ComplexF64(p.G)
    k = ComplexF64(p.K)
    q = Q
    q2 = q * q

    A0 = zeros(ComplexF64, 6, 6)
    A0[4, 2] = -q
    A0[5, 2] = 1 / q
    A0[6, 3] = 1 / q
    A0[1, 4] = 1
    A0[2, 5] = 1
    A0[3, 6] = 1
    A0[4, 6] = h

    A1 = zeros(ComplexF64, 6, 6)
    A1[4, 1] = q
    A1[5, 1] = -1 / q
    A1[6, 1] = -1 / q
    A1[6, 2] = -(g - k * e) * q
    A1[5, 3] = -(e + f) / q2
    A1[6, 3] = (g + k * f) * q
    A1[4, 4] = 1
    A1[5, 4] = -h / q2
    A1[6, 4] = h * k * q
    A1[5, 5] = -1
    A1[6, 6] = -1

    A2 = zeros(ComplexF64, 6, 6)
    A2[4, 1] = -2
    A2[5, 1] = h / q2
    A2[6, 1] = -h * k * q

    return (A0, A1, A2)
end

# Block-diagonalizing basis T and its inverse (Glasser & Wang 2020 Eqs. 7–8).
# Port of the T/Tinv block of inps_tjmat (inps.f lines 145–187).
function _ggj_eigenbasis(p::GGJParameters, Q::ComplexF64)
    h = ComplexF64(p.H)
    q = Q
    q2 = q * q
    λ = 1 / sqrt(q)

    T = @SMatrix ComplexF64[
        1 0 h*q q2/λ h*q -q2/λ
        0 0 0 -1/λ 0 1/λ
        0 0 -1/λ 0 1/λ 0
        0 1 -h/λ -q2 h/λ -q2
        0 0 0 1 0 1
        0 0 1 0 1 0
    ]
    Tinv = @SMatrix ComplexF64[
        1 q2 0 0 0 -h*q
        0 0 -h 1 q2 0
        0 0 -λ/2 0 0 1/2
        0 -λ/2 0 0 1/2 0
        0 0 λ/2 0 0 1/2
        0 λ/2 0 0 1/2 0
    ]
    return Matrix(T), Matrix(Tinv)
end

# Mercier-shifted Frobenius exponents at infinity (Glasser & Wang 2020 Eq. 49):
# R = (3/2 + √(−D_I), 3/2 − √(−D_I)).
function _ggj_exponents(p::GGJParameters, ::ComplexF64)
    p1v = p1(p)
    return Float64[p1v+1.5, -p1v+1.5]
end

# FEM coefficient matrices (I, U, V) of the second-order physical system
# I·u'' + V·u' + U·u = 0 at coordinate x. Port of inpso_get_uv.
function _ggj_coeffs(p::GGJParameters, Q::ComplexF64, x::Real)
    e = ComplexF64(p.E)
    f = ComplexF64(p.F)
    h = ComplexF64(p.H)
    g = ComplexF64(p.G)
    k = ComplexF64(p.K)
    q = Q
    q2 = q * q
    x2 = x * x

    Imat = @SMatrix ComplexF64[1 0 0; 0 q2 0; 0 0 q]
    U = @SMatrix ComplexF64[
        q (-q*x) 0
        (-x/q)*q2 (x2/q)*q2 (-(e + f)/q2)*q2
        (-x/q)*q (-(g - k * e)*q)*q (x2/q+(g+k*f)*q)*q
    ]
    V = @SMatrix ComplexF64[
        0 0 h
        (-h/q2)*q2 0 0
        (h*k*q)*q 0 0
    ]
    return Imat, U, V
end

# Map the first-order asymptotic state U (6×2) and its derivative to the physical
# (W, N, Θ) representation (3×2 each). Port of inpso_get_ua/dua.
function _ggj_physical(U::AbstractMatrix{ComplexF64}, dU::AbstractMatrix{ComplexF64}, x::Real)
    ua = zeros(ComplexF64, 3, 2)
    dua = zeros(ComplexF64, 3, 2)
    @inbounds for j in 1:2
        ua[1, j] = U[1, j] / x
        ua[2, j] = U[2, j]
        ua[3, j] = U[3, j]
        dua[1, j] = U[4, j] - U[1, j] / (x * x)
        dua[2, j] = U[5, j] * x
        dua[3, j] = U[6, j] * x
    end
    return ua, dua
end

"""
    ggj_system_spec(p::GGJParameters) -> SystemSpec

Construct the [`WasowGalerkin.SystemSpec`](@ref) for the single-fluid GGJ
inner-layer system: a 6th-order first-order asymptotic system (`n=6`) with three
physical components `(W, N, Θ)` (`ncomp=3`), a 2-dimensional algebraic subspace
(`n_alg=2`, indices `1:2`), a 3-term asymptotic series, Hermite-cubic elements
(`np=3`), and the even/odd parity decomposition of the half domain.
"""
function ggj_system_spec(p::GGJParameters)
    bc = ParityBC([
        BCKind[NEUMANN, DIRICHLET, DIRICHLET],   # odd:  W'(0)=0, N(0)=0, Θ(0)=0
        BCKind[DIRICHLET, NEUMANN, NEUMANN]     # even: W(0)=0, N'(0)=0, Θ'(0)=0
    ])
    return SystemSpec(;
        n=6, ncomp=3, np=3, nterms=3, n_alg=2,
        alg_idx=[1, 2], exp_idx=[3, 4, 5, 6],
        asymptotic_matrices=_ggj_asymptotic_matrices,
        eigenbasis=_ggj_eigenbasis,
        exponents=_ggj_exponents,
        coeffs=_ggj_coeffs,
        physical=_ggj_physical,
        rescale=rescale_delta,
        bc=bc,
        domain=:half
    )
end

"""
    solve_inner(::GGJModel{:galerkin}, params::GGJParameters, γ::Number;
                kmax=8, nx=512, nq=4, pfac=1.0, cutoff=5, xfac=1.0, tol_res=1e-5)
                -> SVector{2,ComplexF64}

Solve the GGJ inner-layer matching problem with the shared Wasow–Galerkin engine.
Builds the GGJ [`SystemSpec`](@ref ggj_system_spec) and dispatches to
`WasowGalerkin.solve_galerkin`. Returns `(Δ₁, Δ₂)` with the deltac.f parity-swap
and `rescale_delta` applied, matching the legacy output convention.
"""
function solve_inner(::GGJModel{:galerkin}, params::GGJParameters, γ::Number;
    kmax::Int=8, nx::Int=512, nq::Int=4, pfac::Float64=1.0,
    cutoff::Int=5, xfac::Float64=1.0, tol_res::Float64=1e-5)
    spec = ggj_system_spec(params)
    Q = inner_Q(params, γ)
    Δraw = solve_galerkin(spec, params, Q; kmax=kmax, nx=nx, nq=nq, pfac=pfac,
        cutoff=cutoff, xfac=xfac, tol_res=tol_res)
    # deltac.f swap convention (deltac.f lines 194–196), then physical rescaling.
    Δswapped = SVector{2,ComplexF64}(Δraw[2], Δraw[1])
    return rescale_delta(Δswapped, params)
end

# -----------------------------------------------------------------------
# Backward-compatible names for the GGJ asymptotic kernel. The kernel now lives
# in the model-agnostic engine; these thin wrappers preserve the historical
# `InnerLayer` public surface (`build_asymptotics`, `evaluate_asymptotics`,
# `asymptotic_residual`, `pick_xmax`, `InnerAsymptoticsCache`) for the GGJ system.
# -----------------------------------------------------------------------

const InnerAsymptoticsCache = WasowCache

"""
    build_asymptotics(params::GGJParameters, Q::Number; kmax::Int=8) -> WasowCache

Construct the Wasow asymptotic basis for the GGJ system at growth-rate parameter
`Q`. Thin wrapper over the shared engine using the GGJ [`SystemSpec`](@ref
ggj_system_spec).
"""
build_asymptotics(params::GGJParameters, Q::Number; kmax::Int=8) =
    build_wasow(ggj_system_spec(params), params, ComplexF64(Q); kmax=kmax)

"""
    evaluate_asymptotics(cache::WasowCache, x::Real; derivative::Bool=true, apply_T::Bool=true)

Evaluate the GGJ Wasow asymptotic basis at `x`. Thin wrapper over
`WasowGalerkin.evaluate_wasow`.
"""
evaluate_asymptotics(cache::WasowCache, x::Real; derivative::Bool=true, apply_T::Bool=true) =
    evaluate_wasow(cache, x; derivative=derivative, apply_T=apply_T)

"""
    asymptotic_residual(cache::WasowCache, x::Real)

Per-column residual of the GGJ asymptotic basis at `x`. Thin wrapper over
`WasowGalerkin.wasow_residual`.
"""
asymptotic_residual(cache::WasowCache, x::Real) = wasow_residual(cache, x)

"""
    pick_xmax(params::GGJParameters, Q::Number; kwargs...) -> (Float64, WasowCache)

Adaptive outer matching point for the GGJ system. Thin wrapper over
`WasowGalerkin.pick_xmax` using the GGJ [`SystemSpec`](@ref ggj_system_spec).
"""
pick_xmax(params::GGJParameters, Q::Number; kwargs...) =
    pick_xmax(ggj_system_spec(params), params, ComplexF64(Q); kwargs...)

"""
    ggj_system_spec_full(p::GGJParameters) -> SystemSpec

Full-domain variant of [`ggj_system_spec`](@ref): identical physics with
`domain = :full`, for the parity-coupled full-domain solve. The GGJ system is
parity-symmetric, so its full-domain matching matrix is symmetric and recombines
to the half-domain `(Δ_odd, Δ_even)` pair (the validation oracle); the
half-domain path remains the production route for GGJ.
"""
function ggj_system_spec_full(p::GGJParameters)
    s = ggj_system_spec(p)
    return SystemSpec(; n=s.n, ncomp=s.ncomp, np=s.np, nterms=s.nterms, n_alg=s.n_alg,
        alg_idx=s.alg_idx, exp_idx=s.exp_idx,
        asymptotic_matrices=s.asymptotic_matrices, eigenbasis=s.eigenbasis,
        exponents=s.exponents, coeffs=s.coeffs, physical=s.physical,
        rescale=s.rescale, bc=s.bc, domain=:full)
end

"""
    solve_inner_full(::GGJModel{:galerkin}, params::GGJParameters, γ::Number; kwargs...)
        -> WasowGalerkin.FullDomainMatching

Full-domain GGJ inner-layer solve via the shared engine's two-sided
(parity-coupled) path. Returns the [`FullDomainMatching`](@ref) object whose 2×2
matrix `M` is symmetric for the parity-symmetric GGJ system and recombines (via
`WasowGalerkin.parity_recombine`) to the half-domain `(Δ_odd, Δ_even)` pair.
"""
function solve_inner_full(::GGJModel{:galerkin}, params::GGJParameters, γ::Number;
    kmax::Int=8, nx::Int=512, nq::Int=4, pfac::Float64=1.0,
    cutoff::Int=5, xfac::Float64=1.0, tol_res::Float64=1e-5)
    spec = ggj_system_spec_full(params)
    Q = inner_Q(params, γ)
    return solve_galerkin_full(spec, params, Q; kmax=kmax, nx=nx, nq=nq, pfac=pfac,
        cutoff=cutoff, xfac=xfac, tol_res=tol_res)
end
