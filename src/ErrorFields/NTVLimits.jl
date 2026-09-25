"""
    NTVLimits

How much error field a correction coil can cancel before the neoclassical toroidal viscosity
(NTV) torque of its own field costs the rotation that keeps the penetration threshold up. A
correction coil array cancels the dominant-mode overlap at `C_c` per kilo-ampere-turn, but the
non-resonant remainder of its field drives an NTV torque `T·I²` that does not go away when the
resonant part is cancelled. With a torque budget `T_0` and the threshold taken to fall in
proportion to the torque spent, the current needed to correct an intrinsic overlap `δ_EF` down
to the (reduced) threshold solves

```
δ_EF − C_c·I = s·δ_thresh·(1 − T·I²/T_0)
```

whose real roots exist only up to a largest correctable overlap `δ_max`. The model holds only
while the plasma still rotates, `T·I² < T_0`, so the current is capped at `I_max = √(T_0/T)`; at
that current the threshold has fallen to zero and any remaining overlap locks. This is the model
of Logan et al., Nucl. Fusion (2026), doi:10.1088/1741-4326/ae6086, Eqs. (5)–(7), for SPARC and
of Leuthold et al., J. Plasma Phys. 92, E49 (2026), doi:10.1017/S0022377826101421, Eq. (A2),
for ARC. The torque
coefficients are quadratic forms of the applied spectrum, so each is one plasma-response
evaluation per kilo-ampere-turn: `T_full` for the coil's whole field and `T_residual` for its
field with the dominant mode projected out, the torque that survives a perfect correction.
"""

"""
    NTVControl

Settings of the correction-coil NTV evaluation, the `[ErrorFields.NTV]` TOML table.

## Fields

  - `efc_coils`: coil set names of the correction arrays to evaluate (empty: stage off)
  - `method`: KineticForces torque method to use (`"fgar"` by default; it must be enabled in `[KineticForces]`)
  - `verbose`: log per-array couplings
"""
Base.@kwdef struct NTVControl
    efc_coils::Vector{String} = String[]
    method::String = "fgar"
    verbose::Bool = false
end

"""
    EFCCoupling

The couplings of one correction coil array, per kilo-ampere-turn of its current pattern.

## Fields

  - `coil_name`: the array
  - `delta_per_kat`: dominant-mode overlap `|δ|` per kAt (`C_c`)
  - `overlap_percent`: resonant fraction of the array's field, `100·|Vᴴb̃|/‖b̃‖`
  - `torque_full_per_kat2`: NTV torque of the whole field per kAt², N·m, with its sign
  - `torque_residual_per_kat2`: NTV torque of the field with the dominant mode projected out, per kAt², N·m, with its sign

The sign of an NTV torque depends on the rotation and on conventions; the limits below consume
the budget with the torque's magnitude and never treat a negative torque as no torque.
"""
struct EFCCoupling
    coil_name::String
    delta_per_kat::Float64
    overlap_percent::Float64
    torque_full_per_kat2::Float64
    torque_residual_per_kat2::Float64
end

"""
    residual_spectrum(dom::DominantCoupling, b̃; mode=1) -> Vector{ComplexF64}

The applied root-area-weighted spectrum with singular mode `mode` projected out,
`b̃ − V_k (V_kᴴ b̃)`: what a perfect single-mode correction leaves behind.
"""
function residual_spectrum(dom::DominantCoupling, b̃::AbstractVector{<:Number}; mode::Int=1)
    v = dom.right_singular_vectors[:, mode]
    return Vector{ComplexF64}(b̃) .- v .* dot(v, b̃)
end

"""
    correction_current(δ_ef, c::EFCCoupling; delta_threshold, torque_budget, safety_factor=1.0, ntv=true) -> Float64

Correction current, kAt, that brings an intrinsic overlap `δ_ef` down to
`safety_factor × delta_threshold`. Without NTV (`ntv = false`) that is the linear
`(δ_ef − s·δ_thresh) / C_c`, zero when no correction is needed. With NTV the residual torque
lowers the threshold in proportion to the fraction of `torque_budget` (N·m) it consumes, and the
smaller root of the resulting quadratic is returned. `NaN` when the overlap is beyond
[`max_correctable_overlap`](@ref): either the quadratic has no real root, or its root lies past
`I_max = √(T_0/T_residual)`, where the residual torque has used the whole budget and the plasma
no longer rotates.
"""
function correction_current(δ_ef::Real, c::EFCCoupling; delta_threshold::Real, torque_budget::Real, safety_factor::Real=1.0, ntv::Bool=true)
    target = safety_factor * delta_threshold
    excess = δ_ef - target
    excess <= 0 && return 0.0
    ntv || return excess / c.delta_per_kat
    t_res = abs(c.torque_residual_per_kat2)
    t_res == 0 && return excess / c.delta_per_kat
    a = target * t_res / torque_budget
    if a == 0
        current = excess / c.delta_per_kat
    else
        disc = c.delta_per_kat^2 - 4a * excess
        disc < 0 && return NaN
        current = (c.delta_per_kat - sqrt(disc)) / (2a)
    end
    # Past I_max the rotation would be negative, outside the model (Logan et al. 2026, Eq. 6).
    return current <= sqrt(torque_budget / t_res) ? current : NaN
end

"""
    max_correctable_overlap(c::EFCCoupling; delta_threshold, torque_budget, safety_factor=1.0) -> (; with_ntv, residual_only, torque_only)

The largest intrinsic overlap the array can correct, under three readings of the model:

  - `with_ntv`: the largest `δ_ef` for which [`correction_current`](@ref) has a valid root. The
    correctable overlap `s·δ_thresh·(1 − T_residual I²/T_0) + C_c I` peaks at
    `I* = C_c T_0 / (2 s δ_thresh T_residual)`; when `I*` lies beyond `I_max = √(T_0/T_residual)`
    the plasma stops rotating first, and the limit is the overlap at `I_max`, which is `residual_only`.
  - `residual_only`: `C_c √(T_0/T_residual)`, the overlap cancelled when the residual torque has
    used the whole budget (Logan et al. 2026, Eq. 7, the SPARC limit).
  - `torque_only`: `C_c √(T_0/T_full)`, the same with the whole field's torque, dominant mode
    included (Leuthold et al. 2026, Eq. A2, the ARC limit).

Torques enter with their magnitude; an array with zero torque has infinite limits.
"""
function max_correctable_overlap(c::EFCCoupling; delta_threshold::Real, torque_budget::Real, safety_factor::Real=1.0)
    target = safety_factor * delta_threshold
    t_res, t_full = abs(c.torque_residual_per_kat2), abs(c.torque_full_per_kat2)
    torque_only = t_full > 0 ? c.delta_per_kat * sqrt(torque_budget / t_full) : Inf
    t_res > 0 || return (; with_ntv=Inf, residual_only=Inf, torque_only)
    i_max = sqrt(torque_budget / t_res)
    residual_only = c.delta_per_kat * i_max
    i_peak = target > 0 ? c.delta_per_kat * torque_budget / (2 * target * t_res) : Inf
    with_ntv = i_peak < i_max ? target * (1 - t_res * i_peak^2 / torque_budget) + c.delta_per_kat * i_peak : residual_only
    return (; with_ntv, residual_only, torque_only)
end

"""
    efc_current_curve(c::EFCCoupling; delta_threshold, torque_budget, safety_factor=1.0, delta_max=15, npoints=500,
                      torque_rtol=0.0, budget_rtol=0.0) -> NamedTuple

The correction current against intrinsic overlap, `δ_ef` from `0` to `delta_max × δ_thresh`:
`delta_ef`, the linear `current_linear`, the NTV-limited `current_ntv` (`NaN` past the limit),
and the three limits of [`max_correctable_overlap`](@ref).

`torque_rtol` and `budget_rtol` are fractional uncertainties on the NTV torques and on the
torque budget (Logan et al. 2026 use `0.5` for both: ±50 % on the torque, `T_0 = 4 ± 2` N·m).
The model depends on them only through `T/T_0`, so the band is bounded by two curves:
`current_ntv_pessimistic` (torque high, budget low) and `current_ntv_optimistic` (torque low,
budget high), with their limits in `limits_pessimistic` and `limits_optimistic`. With both
tolerances zero the bounds equal the nominal curve.
"""
function efc_current_curve(c::EFCCoupling; delta_threshold::Real, torque_budget::Real, safety_factor::Real=1.0, delta_max::Real=15, npoints::Int=500,
    torque_rtol::Real=0.0, budget_rtol::Real=0.0)
    (0 <= torque_rtol < 1 && 0 <= budget_rtol < 1) ||
        throw(ArgumentError("torque_rtol and budget_rtol are fractional uncertainties in [0, 1); got $torque_rtol and $budget_rtol"))
    δ = collect(range(0.0, delta_max * delta_threshold; length=npoints))
    lin = [correction_current(d, c; delta_threshold, torque_budget, safety_factor, ntv=false) for d in δ]
    ntv_at(budget) = [correction_current(d, c; delta_threshold, torque_budget=budget, safety_factor, ntv=true) for d in δ]
    # Scaling the torques by (1 ± torque_rtol) is the same as scaling the budget by its inverse.
    pessimistic = torque_budget * (1 - budget_rtol) / (1 + torque_rtol)
    optimistic = torque_budget * (1 + budget_rtol) / (1 - torque_rtol)
    limits = max_correctable_overlap(c; delta_threshold, torque_budget, safety_factor)
    return (; delta_ef=δ, current_linear=lin, current_ntv=ntv_at(torque_budget),
        current_ntv_pessimistic=ntv_at(pessimistic), current_ntv_optimistic=ntv_at(optimistic),
        limits_pessimistic=max_correctable_overlap(c; delta_threshold, torque_budget=pessimistic, safety_factor),
        limits_optimistic=max_correctable_overlap(c; delta_threshold, torque_budget=optimistic, safety_factor),
        limits...)
end
