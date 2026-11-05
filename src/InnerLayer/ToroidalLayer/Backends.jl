# Backends.jl
#
# The ray and Galerkin solves behind a model's `solve_inner` / `solve_inner_profile`, and the keywords each
# accepts, so a model type only forwards to them.

"""
    ray_response(params, γ::Number; kwargs...) -> InnerLayerResponse

Rotated-ray matching data for a model's `solve_inner`: converts γ via `inner_Q`, forwards keywords to
[`solve_ray`](@ref), and returns (Δ₊, Δ₋) as the `interchange` and `tearing` fields of
[`InnerLayerResponse`](@ref). `real_profiles`, `npc` and `certify_rtol` are profile-only options
(see [`ray_profile`](@ref)) and are ignored here.
"""
function ray_response(params::ToroidalLayerParameters, γ::Number;
    real_profiles=nothing, npc=nothing, certify_rtol=nothing, kwargs...)
    res = solve_ray(params, inner_Q(params, γ); kwargs...)
    return InnerLayerResponse(; interchange=res.Δ[1], tearing=res.Δ[2])
end

"""
    ray_profile(params, γ::Number; real_profiles=false, npc=8, certify_rtol=1e-3, kwargs...)
        -> (; Δ, x, Ψ, Ξ, dψdx, rescale, certΔ)

Rotated-ray implementation of the [`solve_inner_profile`](@ref) interface. `Δ` and the
layer-center fields come from the solve at the model's default contour angle; x = 0 lies on every contour, so
Ψ(0), Ξ(0) are the real-axis values, and by default only they are returned (`x = [0.0]`,
`certΔ = NaN`). With `real_profiles = true` the fields across the layer (x > 0) come from an
extra θ = 0 solve (`npc` points per cell), a diagnostic; the x = 0 row stays the rotated
solve's, and `certΔ` is the largest relative difference of the two solves' Δ₊, Δ₋, warned
above `certify_rtol`. Other keywords forward to [`solve_ray`](@ref), except θ.
"""
function ray_profile(params::ToroidalLayerParameters, γ::Number;
    real_profiles::Bool=false, npc::Int=8, certify_rtol::Float64=1e-3, kwargs...)
    Q = inner_Q(params, γ)
    rr = solve_ray(params, Q; kwargs...)             # Δ and layer-center fields (optimal θ)
    c = solution_profile(rr; npc=1)                  # first point is s = 0
    if !real_profiles
        return (; Δ=rr.Δ, x=[0.0], Ψ=c.Ψ[1:1, :], Ξ=c.Ξ[1:1, :], profile_conversions(params)..., certΔ=NaN)
    end
    r0 = solve_ray(params, Q; θ=0.0, kwargs...)      # real-axis profile solve
    certΔ = maximum(abs.(r0.Δ .- rr.Δ) ./ max.(abs.(rr.Δ), 1e-300))
    certΔ > certify_rtol && @warn "Ray solver: θ=0 profile solve disagrees with certified Δ" Q certΔ
    prof = solution_profile(r0; npc=npc)
    prof.Ψ[1, :] .= c.Ψ[1, :]                        # x = 0 from the rotated solve, as without profiles
    prof.Ξ[1, :] .= c.Ξ[1, :]
    return (; Δ=rr.Δ, x=prof.s, Ψ=prof.Ψ, Ξ=prof.Ξ, profile_conversions(params)..., certΔ=certΔ)
end

"""
    galerkin_profile(params, γ::Number; kwargs...) -> (; Δ, x, Ψ, Ξ, dψdx, rescale)

Hermite-FEM implementation of the [`solve_inner_profile`](@ref) interface: real-axis solve, so `Δ` and the
profiles come from the same solution. Same numerics and keywords as [`galerkin_response`](@ref).
"""
function galerkin_profile(params::ToroidalLayerParameters, γ::Number; kwargs...)
    Δ, ws = _galerkin_solve(params, γ; kwargs...)
    prof = _solution_profile(ws)
    return (; Δ=Δ, x=prof.x, Ψ=prof.Ψ, Ξ=prof.Ξ, profile_conversions(params)...)
end

"""
    galerkin_response(params, γ::Number; kmax::Int=8, nx::Int=512, nq::Int=4, pfac::Float64=1.0,
                      cutoff::Int=5, xfac::Float64=1.0, tol_res::Float64=1e-5) -> InnerLayerResponse

Solve the inner-layer matching problem with the Hermite-cubic finite element (Galerkin) method
(GWP2016 Sec. III), in the "resonant + noexp + inps" configuration.

Returns the parity-projected matching data (GWP2016 Eqs. 34–35), rescaled by
[`rescale_delta`](@ref), as an `InnerLayerResponse`: `interchange` is the even-parity Δ₊
(isol = 1), `tearing` the odd-parity Δ₋ (isol = 2).

Knobs: `kmax` series order; `nx` cell count; `nq` Gauss–Lobatto order; `pfac` grid packing;
`cutoff` extension-cell span (cells nx−cutoff−1 … nx−1); `xfac` scale on the two inner matching radii; `tol_res`
resonant-cell quadrature tolerance.
"""
function galerkin_response(params::ToroidalLayerParameters, γ::Number; kwargs...)
    Δ, _ = _galerkin_solve(params, γ; kwargs...)
    return InnerLayerResponse(; interchange=Δ[1], tearing=Δ[2])
end

# Each backend's option names: the keywords of the functions its solves forward them to.
_kwarg_names(f) = unique!(Symbol[k for m in methods(f) for k in Base.kwarg_decl(m) if !endswith(string(k), "...")])

"""
    backend_options(solver::Symbol) -> Vector{Symbol}

Keywords the shared `:ray` or `:galerkin` backend accepts, for validating a model's construction options.
"""
backend_options(solver::Symbol) = backend_options(Val(solver))
backend_options(::Val{:ray}) = union(_kwarg_names(solve_ray), (:real_profiles, :npc, :certify_rtol))
backend_options(::Val{:galerkin}) = _kwarg_names(_galerkin_solve)
