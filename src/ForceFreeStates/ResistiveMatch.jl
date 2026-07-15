# ResistiveMatch.jl
#
# Coil-driven outer↔inner resistive asymptotic matching for the STRIDE/Riccati path.
# Implements Wang et al., Phys. Plasmas 27, 122509 (2020), Eq. (11):
#
#     (Δ_out − Δ_in(i2πf)) · C = −Δ_coil,      C = −(Δ_out − Δ_in(i2πf))^{-1} · Δ_coil
#
# Δ_out is the ideal outer-region Δ′ small-solution block, Δ_in(i2πf) is the resistive inner-layer
# matching data (InnerLayer.solve_inner at the rpec forced eigenvalue γ = 2πi·n·f), and Δ_coil is
# the small-solution contribution induced by the external coil perturbations.
#
# Mirrors RDCON's `gal_match_rpec` (Fortran rmatch `match_rpec`, match.f). Key normalization point
# (verified against Wang 2020 + gal_match_rpec): Δ_out and Δ_coil must share the SAME normalization,
# so both use STRIDE's RAW small-solution coefficients —
#   Δ_out  = dp_raw (loop_boundary_conditions, big-solution-driven raw coeffs)
#   Δ_coil = raw edge-driven coeffs (loop_edge_boundary_conditions with snorm = 1)
# The `snorm = |n·q'|^α` used for the delta_coil↔galerkin comparison is not part of the matching
# normalization and is deliberately omitted.

"Result of the coil-driven outer↔inner resistive match (Wang 2020 Eq. 11)."
struct ResonantMatchResult
    cout::Matrix{ComplexF64}      # outer coefficients (2·msing × ncoil)
    cin::Matrix{ComplexF64}       # inner coefficients (2·msing × ncoil)
    deltar::Matrix{ComplexF64}    # per-surface inner-layer Δ (msing × 2), deltac.f convention
    rpec_eig::Vector{ComplexF64}  # forced eigenvalue γ_s = 2πi·n·f per surface
    reconnected_flux::Matrix{ComplexF64}  # matched small-solution (reconnected) amplitude (2·msing × ncoil)
    residual::Float64             # relative solve residual ‖mat·cof − rmat‖ / ‖rmat‖
end

"""
    resonant_match_rpec(delta_out_raw, delta_coil_raw, sings, equil, intr, ctrl) -> ResonantMatchResult

Assemble and solve the 4·msing coil-driven matching system (Wang 2020 Eq. 11; match.f) from STRIDE's
raw outer Δ′ block and raw Δ_coil, plus the per-surface resistive inner-layer Δ (resist_eval →
InnerLayer.solve_inner). `delta_out_raw` is (2msing × 2msing) indexed [driver, surface]; `delta_coil_raw`
is (2msing × ncoil) indexed [surface, coil-mode].
"""
function resonant_match_rpec(delta_out_raw::AbstractMatrix, delta_coil_raw::AbstractMatrix,
    sings::Vector{SingType}, equil::Equilibrium.PlasmaEquilibrium,
    intr::ForceFreeStatesInternal, ctrl::ForceFreeStatesControl)

    msing = length(sings)
    ncoil = size(delta_coil_raw, 2)
    nn = intr.nlow

    if ctrl.gal_ideal_flag                          # ideal limit: skip the inner layer, cout = 0
        # reconnected flux collapses to the bare coil small-solution amplitude (no reconnection)
        return ResonantMatchResult(zeros(ComplexF64, 2msing, ncoil), zeros(ComplexF64, 2msing, ncoil),
            zeros(ComplexF64, msing, 2), zeros(ComplexF64, msing), Matrix{ComplexF64}(delta_coil_raw), 0.0)
    end
    for (nm, v) in (("gal_eta", ctrl.gal_eta), ("gal_rho", ctrl.gal_rho), ("gal_rotation", ctrl.gal_rotation))
        length(v) == msing || error("resonant_match_rpec: $nm has length $(length(v)), expected msing=$msing (one per surface, core→edge)")
    end

    # --- per-surface resistive inner-layer matching data Δ(Q) (Wang 2020; deltac.f) ---
    inner = InnerLayer.GGJModel(; solver=:galerkin)
    deltar = zeros(ComplexF64, msing, 2)
    rpec_eig = zeros(ComplexF64, msing)
    for i in 1:msing
        params = resist_eval(sings[i], equil, intr; eta=ctrl.gal_eta[i], rho=ctrl.gal_rho[i], gamma=ctrl.gal_gamma, ising=i)
        γ = 2π * im * nn * ctrl.gal_rotation[i]     # rpec forced eigenvalue 2πi·n·f
        rpec_eig[i] = γ
        Δ = InnerLayer.solve_inner(inner, params, γ)
        deltar[i, 1] = Δ[1]
        deltar[i, 2] = Δ[2]
    end

    # --- assemble the 4·msing matching system mat·[cout; cin] = rmat (match.f) ---
    mat = zeros(ComplexF64, 4msing, 4msing)
    rmat = zeros(ComplexF64, 4msing, ncoil)
    @views mat[2msing+1:4msing, 1:2msing] .= transpose(delta_out_raw)   # Δ_out (raw dp_raw block)
    @views rmat[2msing+1:4msing, :] .= .-delta_coil_raw                 # −Δ_coil (STRIDE (2msing,ncoil))
    for ising in 1:msing
        idx1 = 2ising - 1; idx2 = 2ising
        idx3 = idx1 + 2msing; idx4 = idx2 + 2msing
        d1 = deltar[ising, 1]; d2 = deltar[ising, 2]
        mat[idx1, idx1] = 1; mat[idx2, idx2] = 1
        mat[idx1, idx3] = -1; mat[idx1, idx4] = 1
        mat[idx2, idx3] = -1; mat[idx2, idx4] = -1
        mat[idx3, idx3] = -d1; mat[idx3, idx4] = d2   # Δ_in blocks
        mat[idx4, idx3] = -d1; mat[idx4, idx4] = -d2
    end

    cof = mat \ rmat
    residual = norm(mat * cof - rmat) / max(norm(rmat), 1e-300)
    return ResonantMatchResult(cof[1:2msing, :], cof[2msing+1:4msing, :], deltar, rpec_eig, residual)
end
