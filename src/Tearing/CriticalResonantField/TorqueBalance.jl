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
    delta_n_p = 1e-2 # JK hack to avoid singularity at Δ=0.0. Need to fix this in the future.
    jxb = -imag(1.0 / (Δ + delta_n_p))
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

function torque_balance_scan(tb; Qmin=-10.0, Qmax=10.0, n=200)
    Qs = range(Qmin, Qmax; length=n)
    torque_out = [torque_balance_value(tb, q) for q in Qs]
    bal = [x[1] for x in torque_out]
    Δs = [x[2] for x in torque_out]
    positive = isfinite.(bal) .& (bal .> 0.0)

    if !any(positive)
        return Qs, bal, [0.0], [0.0], 0.0, 0.0, NaN, Δs
    end
    i = argmax(bal)
    Qpeak_ind = i
    Qpeak = Qs[i]
    maxbal = bal[i]

    Qs_positive = Qs[positive]
    bal_positive = bal[positive]

    i = argmax(bal_positive)
    #Qpeak = Qs_positive[i]

    #maxbal = bal_positive[i]

    br_crit = sqrt(maxbal / tb.lu * (tb.sval^2 / 2.0))
    return Qs, bal, Qs_positive, bal_positive, Qpeak, br_crit, Qpeak_ind, Δs
end

# Need to fix and improve this function. The positive masking is not helpful.
# Need to avoid poles from Q_e and Q_i.
