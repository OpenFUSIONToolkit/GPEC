# TorqueBalance.jl
#
# `TorqueBalance` solves the torque balance eqution for each rational surface.
# It follows the derivation found in Cole PopP 2006. For simplicity, the script
# uses equation 62 of Cole. This can be improved by incorporating the true
# outer-layer Δ' from the perturbed equilibrium and then solving equation 61.
#
# The viscous torque is:
#
#   T_v = 2 · P (Q_0 - Q) / ((S kappa^hat) · (b_r(r_s)/B_phi))^2
#
# The electromagnetic torque is:
#
#   T_em = Im[Δ_inner(Q)] / |alpha + Δ_inner(Q)|^2
#
# Now solving for critical resonant field, we have:
#
#   (b_r(r_s)/B_phi))^2_crit = max(2 · P (Q_0 - Q) / ((S kappa^hat) · Im[-Δ_inner(Q)^-1])
#
# Normalizations, definitions, and other conventions are taken from the Cole paper.
# The critical resonant field is found at each rational surface. For each, the script
# scans over a range of Q values and finds the maximum value of the right-hand side of
# the equation above.

"""
    TorqueBalance{M<:InnerLayerModel, P}

Per-surface torque balance data: `(model, params, Q0, P, lu, sval)`.

"""

struct TorqueBalance{M<:InnerLayerModel,P}
    model::M
    params::P
    Q0::Float64
    P::Float64
    lu::Float64
    sval::Float64
end

function torque_balance_value(tb::TorqueBalance, Q::Number)
    Δ = solve_inner(tb.model, tb.params, ComplexF64(Q)).tearing
    alpha = 1e-2 # Equation 62 of Cole PopP 2006 takes limit of alpha << 1. Set at 1e-2 but more accurate method is alpha = S^(-1/3) * (-r_s Δ'_s)
    jxb = -imag(1.0 / (Δ + alpha))
    return 2.0 * tb.P * (tb.Q0 - Q) / jxb, Δ
end

"""
    torque_balance_scan(model::InnerLayerModel, params::InnerLayerParameters, Q0::Real, P::Real,
        lu::Real, sval::Real;Qmin=-10.0, Qmax=10.0, n=200) -> (Qs, bal, Qs_positive, bal_positive,
        Qpeak, br_crit, Qpeak_ind, Δs)

Scan over a range of Q values to find the maximum of the torque balance equation.
Returns the Q values, torque balance values, positive Q values, positive torque balance values,
the Q value at the peak, the critical resonant field, the index of the peak Q value, and the Δ values for each Q.
"""

function torque_balance_scan(tb; Qmin=-10.0, Qmax=10.0, n=20000)
    Qs = range(Qmin, Qmax; length=n)
    torque_out = [torque_balance_value(tb, q) for q in Qs]
    bal = [x[1] for x in torque_out]
    Δs = [x[2] for x in torque_out]
    positive = isfinite.(bal) .& (bal .> 0.0)

    if !any(positive)
        @warn "No positive torque balance found in the specified range. Try increasing the number of Q samples scanned or adjusting the range."
        return Qs, bal, NaN, NaN, NaN, Δs
    end

    # Find the highest point, then remove candidates within ΔQ of it.
    idx_maxima = findall(isfinite.(bal) .& (bal .> 0))
    sort!(idx_maxima; by=i -> bal[i], rev=true)
    selected = Int[]
    min_ΔQ = 0.001 * abs(Qmax - Qmin)

    for i in idx_maxima
        if all(abs(Qs[i] - Qs[j]) > min_ΔQ for j in selected)
            push!(selected, i)
        end
        length(selected) == 2 && break
    end

    if isempty(selected)
        @warn "No usable positive torque-balance maximum found. Try increasing the number of Q samples scanned or adjusting the range."
        return Qs, bal, NaN, NaN, NaN, Δs
    end

    idx_maxima = selected
    maxima = bal[idx_maxima]
    Qs_maxima = Qs[idx_maxima]

    # find index, q val and bal val of the q closest to Q_e
    idx_closest_Q_e = argmin(abs.(Qs .+ tb.params.Q_e)) # + Q_e since Q_e = -omega_e * Qconv
    q_closest_Q_e = Qs[idx_closest_Q_e]
    bal_closest_Q_e = bal[idx_closest_Q_e]

    # find index, q val and bal val of the q closest to Q_i
    idx_closest_Q_i = argmin(abs.(Qs .+ tb.params.Q_i)) # + Q_i since Q_i = -omega_i * Qconv
    q_closest_Q_i = Qs[idx_closest_Q_i]
    bal_closest_Q_i = bal[idx_closest_Q_i]

    println(
        "Local maxima indices: ",
        idx_maxima,
        " with values: ",
        maxima,
        " and corresponding Qs: ",
        Qs_maxima,
        " closest Q to Q_e: ",
        q_closest_Q_e,
        " closest Q to Q_i: ",
        q_closest_Q_i
    )

    # check if 1st q_maxima is same as Q_e or Q_i from p
    if idx_maxima[1] == idx_closest_Q_e || idx_maxima[1] == idx_closest_Q_i
        if length(idx_maxima) < 2
            @warn "The first local maximum corresponds to a pole from Q_e or Q_i, but there is no second local maximum to select. Try increasing the number of Q samples scanned or adjusting the range."
            return Qs, bal, NaN, NaN, NaN, Δs
        end
        println("Warning: The first local maximum may correspond to an electron or ion diamagnetic resonance. Selecting the second local maximum instead.")
        if idx_maxima[2] == idx_closest_Q_e || idx_maxima[2] == idx_closest_Q_i
            @warn "Both the first and second local maxima correspond to poles from Q_e or Q_i. Unable to select a valid local maximum. Try increasing the number of Q samples scanned or adjusting the range."
            return Qs, bal, NaN, NaN, NaN, Δs
        end
        idx_maximum = idx_maxima[2]
        maximum = bal[idx_maximum]
        Qmaximum = Qs[idx_maximum]
    else
        idx_maximum = idx_maxima[1]
        maximum = bal[idx_maximum]
        Qmaximum = Qs[idx_maximum]
    end

    println("Selected local maximum index: ", idx_maximum, " with value: ", maximum, " and corresponding Q: ", Qmaximum)

    br_crit = sqrt(maximum / tb.lu * (tb.sval^2 / 2.0))

    return Qs, bal, Qmaximum, br_crit, idx_maximum, Δs
end
