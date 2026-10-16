# ResonantMatch.jl
#
# Matching result and the 4·msing outer↔inner system (rmatch match_rpec; Wang et al. 2020 PoP 27 122509, Eq. 11).

"""
    MatchResult

Matching coefficients and matched profiles, found in `result.match`. The profile
fields are empty when the solve kept no Galerkin basis; the inner fields are empty with `ideal`.

## Fields
  - `cout`, `cin`: `(2msing, ncoil)` outer and inner coefficients.
  - `deltar`: `(msing, 2)` inner-layer Δ per surface and parity.
  - `rpec_eig`: forced eigenvalue γ = 2πi·n·f per surface.
  - `residual`: relative residual of the linear solve.
  - `bpen`: `(msing, ncoil)` penetrated resonant field at each layer center.
  - `reconnected_flux`: `(2msing, ncoil)` `Δ_coil + Δ_outᵀ·cout`.
  - `xi`, `xi_deriv`: `(mpert, ngrid, ncoil)` matched outer ξ and ξ′, one column per coil.
  - `inner_psi`, `inner_xi`, `inner_b`: per surface, the inner ψ grid and the composite ξ_ψ and b^ψ on it.
  - `inner_params`: per-surface `GGJParameters` used.
  - `model`: the inner-layer model solved, with its backend options.
"""
struct MatchResult
    cout::Matrix{ComplexF64}
    cin::Matrix{ComplexF64}
    deltar::Matrix{ComplexF64}
    rpec_eig::Vector{ComplexF64}
    residual::Float64
    bpen::Matrix{ComplexF64}
    reconnected_flux::Matrix{ComplexF64}
    xi::Array{ComplexF64,3}
    xi_deriv::Array{ComplexF64,3}
    inner_psi::Vector{Vector{Float64}}
    inner_xi::Vector{Matrix{ComplexF64}}
    inner_b::Vector{Matrix{ComplexF64}}
    inner_params::Vector{InnerLayer.GGJParameters}
    model::InnerLayer.InnerLayerModel
end

# Solve the 4·msing system mat·[cout; cin] = rmat (rmatch match_rpec).
function _match_system(dp_raw::AbstractMatrix, dp_coil::AbstractMatrix, deltar::AbstractMatrix)
    msing = size(dp_raw, 1) ÷ 2
    ncoil = size(dp_coil, 2)
    mat = zeros(ComplexF64, 4msing, 4msing)
    rmat = zeros(ComplexF64, 4msing, ncoil)
    @views mat[(2msing+1):4msing, 1:2msing] .= transpose(dp_raw)   # Δ_out
    @views rmat[(2msing+1):4msing, :] .= .-dp_coil                 # −Δ_coil
    for ising in 1:msing
        idx1 = 2ising - 1
        idx2 = 2ising
        idx3 = idx1 + 2msing
        idx4 = idx2 + 2msing
        delta1 = deltar[ising, 1]
        delta2 = deltar[ising, 2]
        mat[idx1, idx1] = 1
        mat[idx2, idx2] = 1
        mat[idx1, idx3] = -1
        mat[idx1, idx4] = 1
        mat[idx2, idx3] = -1
        mat[idx2, idx4] = -1
        # inner-layer Δ block (match.f signs)
        mat[idx3, idx3] = -delta1
        mat[idx3, idx4] = delta2
        mat[idx4, idx3] = -delta1
        mat[idx4, idx4] = -delta2
    end

    cof = mat \ rmat
    residual = norm(mat * cof - rmat) / max(norm(rmat), 1e-300)
    return cof[1:2msing, :], cof[(2msing+1):4msing, :], residual
end

# Write `m` under SingularSurfaces/Match/, annotated per the metadata contract.
function write_match!(out_h5, m::MatchResult)
    g = "SingularSurfaces/Match"
    out_h5["$g/cout"] = m.cout
    out_h5["$g/cin"] = m.cin
    out_h5["$g/Delta_r"] = m.deltar
    out_h5["$g/bpen"] = m.bpen
    out_h5["$g/reconnected_flux"] = m.reconnected_flux
    out_h5["$g/rpec_eig"] = m.rpec_eig
    # Per-surface inner-layer ξ_ψ(ψ) (match.f intotsol); ragged grids → one dataset pair per surface.
    for i in eachindex(m.inner_psi)
        out_h5["$g/Inner/psi_$i"] = m.inner_psi[i]
        out_h5["$g/Inner/xi_$i"] = m.inner_xi[i]
        out_h5["$g/Inner/b_$i"] = m.inner_b[i]
    end
    out_h5["$g/residual"] = m.residual
    if !isempty(m.inner_params)
        for f in (:E, :F, :G, :H, :K, :M)
            out_h5["$g/InnerParams/$(f)"] = [getfield(pp, f) for pp in m.inner_params]
        end
        # Literature names, matching the Tearing PerSurface mapping for the same fields.
        out_h5["$g/InnerParams/tau_A"] = [pp.taua for pp in m.inner_params]
        out_h5["$g/InnerParams/tau_R"] = [pp.taur for pp in m.inner_params]
        out_h5["$g/InnerParams/dVdpsi"] = [pp.v1 for pp in m.inner_params]
    end
    a = attrs(out_h5[g])
    a["inner_model"] = InnerLayer.model_name(m.model)
    m.model isa InnerLayer.GGJModel && (a["inner_model_options"] = string(m.model.options))
    inner = ["$g/Inner/$(v)_$i" => meta for i in eachindex(m.inner_psi) for (v, meta) in _MATCH_INNER_H5_ANNOTATIONS]
    Utilities.HDF5Annotations.annotate!(out_h5, vcat(MATCH_H5_ANNOTATIONS, inner))
    return nothing
end

# Matching quantities are per unit coil-response column (the Δ′ coil block's normalization).
const _MATCH_SURF = (; dims=("surface",))
const MATCH_H5_ANNOTATIONS = [
    "SingularSurfaces/Match/cout" =>
        (; long_name="outer-region matching coefficients per surface parity, per unit coil-response column", dims=("surface_parity", "coil")),
    "SingularSurfaces/Match/cin" =>
        (; long_name="inner-region matching coefficients per surface parity, per unit coil-response column", dims=("surface_parity", "coil")),
    "SingularSurfaces/Match/Delta_r" =>
        (; long_name="inner-layer Δ per surface and parity (deltac convention)", dims=("surface", "parity")),
    "SingularSurfaces/Match/bpen" =>
        (; long_name="penetrated resonant field b^ψ at each layer center, per unit coil-response column", dims=("surface", "coil")),
    "SingularSurfaces/Match/reconnected_flux" =>
        (; long_name="reconnected flux Δ_coil + Δ_outᵀ·cout per surface parity, per unit coil-response column", dims=("surface_parity", "coil")),
    "SingularSurfaces/Match/rpec_eig" => (; long_name="forced eigenvalue γ = 2πi·n·f at each surface", units="rad/s", _MATCH_SURF...),
    "SingularSurfaces/Match/residual" => (; long_name="relative residual of the matching linear solve"),
    "SingularSurfaces/Match/InnerParams/E" => (; long_name="Glasser-Greene-Johnson coefficient E per surface", _MATCH_SURF...),
    "SingularSurfaces/Match/InnerParams/F" => (; long_name="Glasser-Greene-Johnson coefficient F per surface", _MATCH_SURF...),
    "SingularSurfaces/Match/InnerParams/G" => (; long_name="Glasser-Greene-Johnson coefficient G per surface", _MATCH_SURF...),
    "SingularSurfaces/Match/InnerParams/H" => (; long_name="Glasser-Greene-Johnson coefficient H per surface", _MATCH_SURF...),
    "SingularSurfaces/Match/InnerParams/K" => (; long_name="Glasser-Greene-Johnson coefficient K per surface", _MATCH_SURF...),
    "SingularSurfaces/Match/InnerParams/M" => (; long_name="Glasser-Greene-Johnson coefficient M per surface", _MATCH_SURF...),
    "SingularSurfaces/Match/InnerParams/tau_A" => (; long_name="Alfvén time τ_A per surface", units="s", _MATCH_SURF...),
    "SingularSurfaces/Match/InnerParams/tau_R" => (; long_name="resistive diffusion time τ_R per surface", units="s", _MATCH_SURF...),
    "SingularSurfaces/Match/InnerParams/dVdpsi" => (; long_name="dV/dψ_N at each surface", units="m^3", _MATCH_SURF...)
]
const _MATCH_INNER_H5_ANNOTATIONS = [
    "psi" => (; long_name="normalized poloidal flux ψ_N grid across the inner layer of this surface", dims=("inner_psi",)),
    "xi" => (; long_name="composite inner-layer ξ^ψ per unit coil-response column", dims=("inner_psi", "coil")),
    "b" => (; long_name="composite inner-layer b^ψ per unit coil-response column", dims=("inner_psi", "coil"))
]
