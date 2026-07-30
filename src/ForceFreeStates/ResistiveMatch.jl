# ResistiveMatch.jl
#
# This file does the "matching" step for the resistive (tearing) plasma response.
#
# The plasma has an outer region (ideal MHD, which STRIDE already solves) and a very
# thin inner layer right at each rational surface, where resistivity actually matters. Neither piece
# is complete by itself; they have to agree where they meet. Computing that agreement is what this
# file does, following Wang et al. 2020 (Phys. Plasmas 27, 122509), Eq. (11):
#
#     C = -(Δ_out - Δ_in(i2πf))^{-1} · Δ_coil
#
# In layman's terms:
#   Δ_out  = how the ideal outer plasma responds (comes from STRIDE's Δ' solve)
#   Δ_in   = how the resistive inner layer responds; the "i2πf" is where plasma rotation enters
#   Δ_coil = the push from the external coil (this is what drives everything)
#   C      = the matched coefficients we solve for
#
# This is the same calculation the Fortran code performs in RDCON's gal_match_rpec (match.f), done here in
# Julia on the STRIDE path.
#
# Δ_out and Δ_coil have to be in the SAME units before they can be subtracted, so both use STRIDE's RAW
# coefficients (no extra scaling). A factor called snorm (= |n·q'|^α) appears elsewhere when comparing to the
# Galerkin code, but that is only a unit conversion for plotting, not part of the physics, so it is omitted here.

"What comes out of the match (Wang 2020 Eq. 11)."
struct ResonantMatchResult
    cout::Matrix{ComplexF64}      # the outer coefficients we solved for (2·msing × ncoil)
    cin::Matrix{ComplexF64}       # the inner-layer coefficients (2·msing × ncoil)
    deltar::Matrix{ComplexF64}    # the inner-layer Δ at each surface (msing × 2)
    rpec_eig::Vector{ComplexF64}  # the frequency each layer "sees": γ = 2πi·n·f, one per surface
    reconnected_flux::Matrix{ComplexF64}  # the reconnected (resonant) flux, the physical answer we want
    residual::Float64             # how well the linear solve worked (should be tiny, ~1e-16)
end

"""
    resonant_match_rpec(delta_out_raw, delta_coil_raw, sings, equil, intr, ctrl) -> ResonantMatchResult

Build and solve the matching system (Wang 2020 Eq. 11). It takes STRIDE's raw outer response
`delta_out_raw` and raw coil drive `delta_coil_raw`, gets the resistive inner-layer Δ at each surface
(resist_eval → InnerLayer.solve_inner), then solves for the matched coefficients.
Shapes: `delta_out_raw` is (2msing × 2msing); `delta_coil_raw` is (2msing × ncoil).
"""
function resonant_match_rpec(delta_out_raw::AbstractMatrix, delta_coil_raw::AbstractMatrix,
    sings::Vector{SingType}, equil::Equilibrium.PlasmaEquilibrium,
    intr::ForceFreeStatesInternal, ctrl::ForceFreeStatesControl)

    msing = size(delta_out_raw, 1) ÷ 2
    ncoil = size(delta_coil_raw, 2)
    nn = intr.nlow

    # Safety checks: make sure the surfaces line up with the outer blocks before I trust anything. If
    # they don't match, the wrong inner-layer Δ would quietly get attached to the wrong surface, so I
    # error out loudly instead of getting a silently-wrong answer.
    size(delta_out_raw) == (2msing, 2msing) ||
        error("resonant_match_rpec: delta_out_raw is $(size(delta_out_raw)), expected (2msing, 2msing)")
    length(sings) == msing ||
        error("resonant_match_rpec: sings length $(length(sings)) != msing=$msing (surface-set mismatch)")
    size(delta_coil_raw, 1) == 2msing ||
        error("resonant_match_rpec: delta_coil_raw has $(size(delta_coil_raw, 1)) rows, expected 2msing=$(2msing)")

    if ctrl.gal_ideal_flag                          # "ideal" case: no resistivity, so no reconnection → cout is 0
        # with no inner layer there's nothing to reconnect, so the flux is just the bare coil drive
        return ResonantMatchResult(zeros(ComplexF64, 2msing, ncoil), zeros(ComplexF64, 2msing, ncoil),
            zeros(ComplexF64, msing, 2), zeros(ComplexF64, msing), Matrix{ComplexF64}(delta_coil_raw), 0.0)
    end
    for (nm, v) in (("gal_eta", ctrl.gal_eta), ("gal_rho", ctrl.gal_rho), ("gal_rotation", ctrl.gal_rotation))
        length(v) == msing || error("resonant_match_rpec: $nm has length $(length(v)), expected msing=$msing (one per surface, core→edge)")
    end

    # --- get the resistive inner-layer Δ at each rational surface ---
    inner = InnerLayer.GGJModel(; solver=:galerkin)
    deltar = zeros(ComplexF64, msing, 2)
    rpec_eig = zeros(ComplexF64, msing)
    for i in 1:msing
        params = resist_eval(sings[i], equil, intr; eta=ctrl.gal_eta[i], rho=ctrl.gal_rho[i], gamma=ctrl.gal_gamma, ising=i)
        γ = 2π * im * nn * ctrl.gal_rotation[i]     # the frequency this layer sees (plasma rotation enters here)
        rpec_eig[i] = γ
        Δ = InnerLayer.solve_inner(inner, params, γ)
        deltar[i, 1] = Δ[1]
        deltar[i, 2] = Δ[2]
    end

    # --- build the big linear system  mat · [cout; cin] = rmat  and solve it (this IS Eq. 11 written out) ---
    mat = zeros(ComplexF64, 4msing, 4msing)
    rmat = zeros(ComplexF64, 4msing, ncoil)
    @views mat[2msing+1:4msing, 1:2msing] .= transpose(delta_out_raw)   # the Δ_out part of the system
    @views rmat[2msing+1:4msing, :] .= .-delta_coil_raw                 # the −Δ_coil drive goes on the right-hand side
    for ising in 1:msing
        # fill in one surface at a time: top rows tie cout to cin, bottom rows carry the inner-layer Δ
        idx1 = 2ising - 1; idx2 = 2ising
        idx3 = idx1 + 2msing; idx4 = idx2 + 2msing
        d1 = deltar[ising, 1]; d2 = deltar[ising, 2]
        mat[idx1, idx1] = 1; mat[idx2, idx2] = 1
        mat[idx1, idx3] = -1; mat[idx1, idx4] = 1
        mat[idx2, idx3] = -1; mat[idx2, idx4] = -1
        mat[idx3, idx3] = -d1; mat[idx3, idx4] = d2   # the Δ_in entries for this surface
        mat[idx4, idx3] = -d1; mat[idx4, idx4] = -d2
    end

    cof = mat \ rmat                                              # solve the system
    residual = norm(mat * cof - rmat) / max(norm(rmat), 1e-300)  # double-check the solve is good
    cout = cof[1:2msing, :]
    cin = cof[2msing+1:4msing, :]
    # The reconnected flux is the coil drive plus the plasma's own outer response to the matched cout.
    # It also equals -Δ_in·cin, so the inner-layer side agrees — a useful sanity check.
    reconnected_flux = delta_coil_raw .+ transpose(delta_out_raw) * cout
    return ResonantMatchResult(cout, cin, deltar, rpec_eig, reconnected_flux, residual)
end
