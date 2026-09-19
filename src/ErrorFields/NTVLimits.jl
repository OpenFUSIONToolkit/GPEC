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

whose real roots exist only up to a largest correctable overlap `δ_max`. The torque
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
smaller root of the resulting quadratic is returned; `NaN` when no real root exists, i.e. the
overlap is beyond [`max_correctable_overlap`](@ref).
"""
function correction_current(δ_ef::Real, c::EFCCoupling; delta_threshold::Real, torque_budget::Real, safety_factor::Real=1.0, ntv::Bool=true)
    target = safety_factor * delta_threshold
    excess = δ_ef - target
    excess <= 0 && return 0.0
    ntv || return excess / c.delta_per_kat
    a = target * abs(c.torque_residual_per_kat2) / torque_budget
    a == 0 && return excess / c.delta_per_kat
    disc = c.delta_per_kat^2 - 4a * excess
    disc < 0 && return NaN
    return (c.delta_per_kat - sqrt(disc)) / (2a)
end

"""
    max_correctable_overlap(c::EFCCoupling; delta_threshold, torque_budget, safety_factor=1.0) -> (; with_ntv, torque_only)

The largest intrinsic overlap the array can correct: `with_ntv`, where the quadratic of
[`correction_current`](@ref) loses its real roots, `s·δ_thresh + C_c² T_0 / (4 s δ_thresh T_residual)`;
and `torque_only`, the overlap cancelled by the current at which the whole field's torque alone
exhausts the budget, `C_c √(T_0 / T_full)`.
"""
function max_correctable_overlap(c::EFCCoupling; delta_threshold::Real, torque_budget::Real, safety_factor::Real=1.0)
    target = safety_factor * delta_threshold
    t_res, t_full = abs(c.torque_residual_per_kat2), abs(c.torque_full_per_kat2)
    with_ntv = t_res > 0 ? target + c.delta_per_kat^2 * torque_budget / (4 * target * t_res) : Inf
    torque_only = t_full > 0 ? c.delta_per_kat * sqrt(torque_budget / t_full) : Inf
    return (; with_ntv, torque_only)
end

"""
    efc_current_curve(c::EFCCoupling; delta_threshold, torque_budget, safety_factor=1.0, delta_max=15, npoints=500) -> NamedTuple

The correction current against intrinsic overlap, `δ_ef` from `0` to `delta_max × δ_thresh`:
`delta_ef`, the linear `current_linear`, the NTV-limited `current_ntv` (`NaN` past the limit),
and the two limits of [`max_correctable_overlap`](@ref).
"""
function efc_current_curve(c::EFCCoupling; delta_threshold::Real, torque_budget::Real, safety_factor::Real=1.0, delta_max::Real=15, npoints::Int=500)
    δ = collect(range(0.0, delta_max * delta_threshold; length=npoints))
    lin = [correction_current(d, c; delta_threshold, torque_budget, safety_factor, ntv=false) for d in δ]
    ntv = [correction_current(d, c; delta_threshold, torque_budget, safety_factor, ntv=true) for d in δ]
    limits = max_correctable_overlap(c; delta_threshold, torque_budget, safety_factor)
    return (; delta_ef=δ, current_linear=lin, current_ntv=ntv, limits...)
end
