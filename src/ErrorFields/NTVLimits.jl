"""
    NTVLimits

How much error field a correction coil can cancel before the neoclassical toroidal viscosity
(NTV) torque of its own field costs the rotation that keeps the penetration threshold up. A
correction coil array cancels the dominant-mode overlap at `C_c` per kilo-ampere-turn, but the
non-resonant remainder of its field drives an NTV torque that does not go away when the
resonant part is cancelled.

The torque of a field is a quadratic form of its spectrum, so per kilo-ampere-turn it is one
plasma-response evaluation followed by kinetic torque evaluations: `T_full` for the coil's
whole field and `T_residual` for its field with the dominant mode projected out, the torque that
survives a perfect correction. Each is tabulated against a rigid shift `Δω` of the E×B rotation
(the diamagnetic frequencies held fixed) on a grid that refines itself where the torque changes
fastest, spanning both senses of rotation past the neoclassical offset, so the table carries the
sign changes and the resonances a single evaluation at the nominal rotation cannot.

The correction current then follows from a zero-dimensional torque balance: with the torque
budget `T_0` the torque that would bring the reference rotation `ω_ref` to rest, the rotation
shift under a current `I` solves `Δω·T_0/ω_ref = T(Δω)·I²`, and the threshold falls as
`δ_thresh·(ω/ω_0)^α = δ_thresh·(1 + Δω/ω_ref)^α`. The current that corrects an intrinsic overlap
`δ_EF` down to `s·δ_thresh` solves `δ_EF − C_c·I = s·δ_thresh·(1 + Δω(I)/ω_ref)^α`. When the
balance loses its root the rotation has bifurcated away and the overlap is not correctable. A
constant braking torque with `α = 1` reduces this to the closed-form quadratic
`δ_EF − C_c·I = s·δ_thresh·(1 − T·I²/T_0)` of the original analysis, kept as the `:linear` model.
"""

"""
    NTVControl

Settings of the correction-coil NTV evaluation, the `[ErrorFields.NTV]` TOML table.

## Fields

  - `efc_coils`: coil set names of the correction arrays to evaluate (empty: stage off)
  - `method`: KineticForces torque method to use (`"fgar"` by default; it must be enabled in `[KineticForces]`)
  - `rotation_scan`: tabulate the torques against a rigid E×B rotation shift (`true`); `false` evaluates only the nominal rotation, which limits the analysis to the `:linear` model
  - `rotation_scan_points`: points of the initial symmetric shift grid (made odd so that the unshifted point is one of them)
  - `rotation_scan_max_points`: cap on the number of kinetic evaluations per spectrum after adaptive refinement
  - `rotation_scan_tolerance`: refinement stops when every interval's midpoint prediction error is below this fraction of the torque range
  - `rotation_span_factor`: the scan covers `±span_factor × max(|ω_E|, offset_factor·|ω_*T|)` evaluated at the innermost kinetic surface
  - `rotation_offset_factor`: rough neoclassical offset as a multiple of the ion temperature-gradient diamagnetic frequency `ω_*T` (about 2 after Park, Boozer and Menard 2009), used only to size the scan
  - `verbose`: log per-array couplings and every scan point
"""
Base.@kwdef struct NTVControl
    efc_coils::Vector{String} = String[]
    method::String = "fgar"
    rotation_scan::Bool = true
    rotation_scan_points::Int = 9
    rotation_scan_max_points::Int = 21
    rotation_scan_tolerance::Float64 = 0.05
    rotation_span_factor::Float64 = 1.0
    rotation_offset_factor::Float64 = 2.0
    verbose::Bool = false
end

"""
Sign relating the kinetic torque to the rotation it acts on: `−1` means a positive `T_φ`
lowers the E×B rotation shift `Δω`, i.e. the kernel reports a braking torque as positive for
the co-rotating profiles of the kinetic file. Fixed empirically on the DIII-D error-field
example, whose tabulated torque is positive at the nominal rotation and rises through zero at a
negative shift, so the balance is restoring only with this sign; [`rotation_shift`](@ref) checks
the slope at every crossing it uses and warns if the convention does not hold for a run.
"""
const TORQUE_ROTATION_SIGN = -1.0

"""
    EFCCoupling

The couplings of one correction coil array, per kilo-ampere-turn of its current pattern.

## Fields

  - `coil_name`: the array
  - `delta_per_kat`: dominant-mode overlap `|δ|` per kAt (`C_c`)
  - `overlap_percent`: resonant fraction of the array's field, `100·|Vᴴb̃|/‖b̃‖`
  - `torque_full_per_kat2`: NTV torque of the whole field per kAt², N·m, with its sign, at the nominal rotation
  - `torque_residual_per_kat2`: NTV torque of the field with the dominant mode projected out, per kAt², N·m, with its sign, at the nominal rotation
  - `rotation_shift`: the scanned rigid E×B rotation shifts `Δω`, rad/s, sorted and containing 0 (a single 0 when no scan was made)
  - `torque_full_scan`, `torque_residual_scan`: the two torques at each shift, N·m per kAt²
  - `omega_reference`: the reference rotation `ω_ref`, rad/s, the ion toroidal rotation weighted by density and volume; `NaN` without a scan
  - `omega_offset_estimate`: the rough neoclassical offset used to size the scan, rad/s; `NaN` without a scan
  - `psi`: kinetic ψ_N grid of the torque profiles (empty without a scan)
  - `torque_full_profile`, `torque_residual_profile`: cumulative torque `T(ψ)` at each shift, `length(psi) × length(rotation_shift)`, N·m per kAt²

The five-argument constructor builds a coupling with no scan, the nominal torques only.
"""
struct EFCCoupling
    coil_name::String
    delta_per_kat::Float64
    overlap_percent::Float64
    torque_full_per_kat2::Float64
    torque_residual_per_kat2::Float64
    rotation_shift::Vector{Float64}
    torque_full_scan::Vector{Float64}
    torque_residual_scan::Vector{Float64}
    omega_reference::Float64
    omega_offset_estimate::Float64
    psi::Vector{Float64}
    torque_full_profile::Matrix{Float64}
    torque_residual_profile::Matrix{Float64}
end

EFCCoupling(coil_name, delta_per_kat, overlap_percent, torque_full_per_kat2, torque_residual_per_kat2) =
    EFCCoupling(coil_name, delta_per_kat, overlap_percent, torque_full_per_kat2, torque_residual_per_kat2,
        [0.0], [Float64(torque_full_per_kat2)], [Float64(torque_residual_per_kat2)], NaN, NaN, Float64[], zeros(0, 1), zeros(0, 1))

# Field-wise equality (the default falls back to identity for the array fields); NaN equals NaN.
Base.:(==)(a::EFCCoupling, b::EFCCoupling) = all(isequal(getfield(a, f), getfield(b, f)) for f in fieldnames(EFCCoupling))

"""
    has_rotation_scan(c::EFCCoupling) -> Bool

Whether the coupling carries a torque-versus-rotation table (more than the nominal point and a finite reference rotation).
"""
has_rotation_scan(c::EFCCoupling) = length(c.rotation_shift) > 1 && isfinite(c.omega_reference)

"""
    rotation_scan_span(omega_e, omega_star_T; span_factor=1.0, offset_factor=2.0) -> (; span, offset_estimate)

Half-width of the symmetric rotation-shift scan, `span_factor × max(|ω_E|, offset_factor·|ω_*T|)`,
from the E×B and ion temperature-gradient diamagnetic frequencies at the innermost kinetic
surface, and the signed rough offset `offset_factor·ω_*T` it was sized against.
"""
function rotation_scan_span(omega_e::Real, omega_star_T::Real; span_factor::Real=1.0, offset_factor::Real=2.0)
    offset_estimate = offset_factor * omega_star_T
    span = span_factor * max(abs(omega_e), abs(offset_estimate))
    span > 0 || throw(ArgumentError("rotation_scan_span: both ω_E and ω_*T vanish at the innermost surface; nothing to scan"))
    return (; span, offset_estimate)
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

_scan_values(c::EFCCoupling, field::Symbol) =
    field === :residual ? c.torque_residual_scan :
    field === :full ? c.torque_full_scan :
    throw(ArgumentError("field must be :residual or :full, got $field"))

"""
    torque_at(c::EFCCoupling, Δω; field=:residual) -> Float64

The tabulated torque, N·m per kAt², at rotation shift `Δω` by linear interpolation of the scan;
`NaN` outside the scanned span. Without a scan the nominal torque, for any shift.
"""
function torque_at(c::EFCCoupling, Δω::Real; field::Symbol=:residual)
    T = _scan_values(c, field)
    x = c.rotation_shift
    length(x) == 1 && return T[1]
    (Δω < x[1] || Δω > x[end]) && return NaN
    i = clamp(searchsortedlast(x, Δω), 1, length(x) - 1)
    t = (Δω - x[i]) / (x[i+1] - x[i])
    return (1 - t) * T[i] + t * T[i+1]
end

"""
    torque_zero_crossings(c::EFCCoupling; field=:residual) -> Vector{Float64}

Rotation shifts at which the tabulated torque changes sign (linear interpolation between scan
points): the neoclassical offsets of this field, as found rather than estimated.
"""
function torque_zero_crossings(c::EFCCoupling; field::Symbol=:residual)
    T = _scan_values(c, field)
    x = c.rotation_shift
    out = Float64[]
    for i in 1:(length(x)-1)
        if T[i] * T[i+1] < 0
            push!(out, x[i] - T[i] * (x[i+1] - x[i]) / (T[i+1] - T[i]))
        elseif T[i] == 0 && (i == 1 || T[i-1] != 0)
            push!(out, x[i])
        end
    end
    return out
end

# Bisection of a continuous g on [a, b] with g(a)·g(b) ≤ 0, to relative width `rtol`.
function _bisect(g, a::Float64, b::Float64; rtol::Float64=1e-10, maxiter::Int=200)
    ga, gb = g(a), g(b)
    ga == 0 && return a
    gb == 0 && return b
    ga * gb < 0 || return NaN
    for _ in 1:maxiter
        m = 0.5 * (a + b)
        gm = g(m)
        (gm == 0 || abs(b - a) <= rtol * max(abs(a), abs(b), 1e-300)) && return m
        if ga * gm < 0
            b, gb = m, gm
        else
            a, ga = m, gm
        end
    end
    return 0.5 * (a + b)
end

"""
    rotation_shift(c::EFCCoupling, current; torque_budget, omega_reference=c.omega_reference, field=:residual) -> Float64

Steady rotation shift `Δω` (rad/s) of the zero-dimensional torque balance
`Δω·T_0/ω_ref = T(Δω)·I²` under `current` kAt of the array, on the branch continuous from the
unshifted state: the first root met walking from `Δω = 0` in the direction the torque pushes.
`NaN` when no root lies inside the scanned span, which is the rotation bifurcating away (or a
span that was too small; the scan settings say which). `torque_budget` is `T_0`, N·m; the
reference rotation may be overridden, e.g. by the rotation at the dominant rational surface.
"""
function rotation_shift(c::EFCCoupling, current::Real; torque_budget::Real, omega_reference::Real=c.omega_reference, field::Symbol=:residual)
    has_rotation_scan(c) || throw(ArgumentError("rotation_shift needs a coupling with a rotation scan (see NTVControl.rotation_scan)"))
    torque_budget > 0 || throw(ArgumentError("torque_budget must be positive"))
    isfinite(omega_reference) && omega_reference != 0 || throw(ArgumentError("omega_reference must be finite and nonzero"))
    current == 0 && return 0.0
    friction = torque_budget / omega_reference
    g(Δ) = Δ * friction - TORQUE_ROTATION_SIGN * torque_at(c, Δ; field) * current^2
    x = c.rotation_shift
    i0 = findfirst(==(0.0), x)
    i0 === nothing && throw(ArgumentError("the rotation scan must contain the unshifted point"))
    T0 = _scan_values(c, field)[i0]
    T0 == 0 && return 0.0
    # Walk outward in the direction the torque pushes the rotation (sign of Δ ≈ ω_ref·T·I²/T_0).
    step = sign(TORQUE_ROTATION_SIGN * T0 * omega_reference) > 0 ? 1 : -1
    i = i0
    while 1 <= i + step <= length(x)
        a, b = x[i], x[i+step]
        if g(a) * g(b) <= 0
            root = _bisect(g, min(a, b), max(a, b))
            # A restoring balance needs the torque falling through the crossing in the sense of the shift.
            slope = (torque_at(c, b; field) - torque_at(c, a; field)) / (b - a)
            TORQUE_ROTATION_SIGN * slope * sign(omega_reference) > 0 &&
                @warn "rotation_shift: the torque of $(c.coil_name) grows with the rotation at the balance point Δω = $(root) rad/s; check TORQUE_ROTATION_SIGN against the kernel's convention" maxlog =
                    1
            return root
        end
        i += step
    end
    return NaN
end

"""
    threshold_factor(c::EFCCoupling, current; torque_budget, rotation_exponent=1.0, omega_reference=c.omega_reference,
                     model=:torque_balance, field=:residual) -> Float64

Factor by which the penetration threshold is reduced under `current` kAt: `(1 + Δω/ω_ref)^α`
with the balance's rotation shift for `model = :torque_balance` (`NaN` past the bifurcation
or once the rotation is brought to rest), or `1 − |T|·I²/T_0` for `model = :linear`; `field`
selects the residual (default) or the whole field's torque.
"""
function threshold_factor(c::EFCCoupling, current::Real; torque_budget::Real, rotation_exponent::Real=1.0,
    omega_reference::Real=c.omega_reference, model::Symbol=:torque_balance, field::Symbol=:residual)
    if model === :linear
        T = field === :residual ? c.torque_residual_per_kat2 : c.torque_full_per_kat2
        return 1 - abs(T) * current^2 / torque_budget
    elseif model === :torque_balance
        Δ = rotation_shift(c, current; torque_budget, omega_reference, field)
        isnan(Δ) && return NaN
        ratio = 1 + Δ / omega_reference
        ratio > 0 || return NaN                     # the rotation is brought to rest: nothing left to protect the threshold
        return ratio^rotation_exponent
    end
    throw(ArgumentError("model must be :torque_balance or :linear, got $model"))
end

_resolve_model(c::EFCCoupling, model::Symbol) = model === :auto ? (has_rotation_scan(c) ? :torque_balance : :linear) : model

"""
    correction_current(δ_ef, c::EFCCoupling; delta_threshold, torque_budget, safety_factor=1.0, ntv=true,
                       model=:auto, rotation_exponent=1.0, omega_reference=c.omega_reference) -> Float64

Correction current, kAt, that brings an intrinsic overlap `δ_ef` down to
`safety_factor × delta_threshold`. Without NTV (`ntv = false`) that is the linear
`(δ_ef − s·δ_thresh) / C_c`, zero when no correction is needed. With NTV the residual torque
lowers the threshold by [`threshold_factor`](@ref): for the `:torque_balance` model (the default
when the coupling carries a rotation scan) the current is the root of
`δ_ef − C_c·I − s·δ_thresh·(1 + Δω(I)/ω_ref)^α`; for `:linear` it is the smaller root of the
closed-form quadratic, provided the residual torque has not exhausted the budget there. `NaN`
when no such root exists, i.e. the overlap is beyond [`max_correctable_overlap`](@ref).
"""
function correction_current(δ_ef::Real, c::EFCCoupling; delta_threshold::Real, torque_budget::Real, safety_factor::Real=1.0, ntv::Bool=true,
    model::Symbol=:auto, rotation_exponent::Real=1.0, omega_reference::Real=c.omega_reference)
    target = safety_factor * delta_threshold
    excess = δ_ef - target
    excess <= 0 && return 0.0
    ntv || return excess / c.delta_per_kat
    model = _resolve_model(c, model)
    if model === :linear
        a = target * abs(c.torque_residual_per_kat2) / torque_budget
        a == 0 && return excess / c.delta_per_kat
        disc = c.delta_per_kat^2 - 4a * excess
        disc < 0 && return NaN
        I = (c.delta_per_kat - sqrt(disc)) / (2a)
        # The budget must not be exhausted at the root: a negative threshold is no correction (OMFIT masked these).
        return abs(c.torque_residual_per_kat2) * I^2 < torque_budget ? I : NaN
    end
    h(I) = δ_ef - c.delta_per_kat * I - target * threshold_factor(c, I; torque_budget, rotation_exponent, omega_reference, model)
    I_lin = excess / c.delta_per_kat
    h_lin = h(I_lin)
    isnan(h_lin) && return NaN                      # the linear current alone already collapses the rotation
    h_lin <= 0 && return _bisect(h, 0.0, I_lin)     # an accelerating torque: less current than the linear one
    # A braking torque: the root lies above the linear current, before the balance fails. Find the minimum of h
    # there (the root is a tangency at the correctable limit) and bisect down to the smaller root.
    I_hi = _largest_finite(I -> threshold_factor(c, I; torque_budget, rotation_exponent, omega_reference, model), I_lin, 1e3 * I_lin)
    isinf(I_hi) && (I_hi = 1e3 * I_lin)
    I_star = _argmin_unimodal(h, I_lin, I_hi)
    h(I_star) > 0 && return NaN
    return _bisect(h, I_lin, I_star)
end

# Golden-section minimum of a unimodal g on [a, b]; NaN values count as +Inf.
function _argmin_unimodal(g, a::Float64, b::Float64; maxiter::Int=200)
    φ = (sqrt(5.0) - 1) / 2
    val(x) = (v = g(x); isnan(v) ? Inf : v)
    c, d = b - φ * (b - a), a + φ * (b - a)
    gc, gd = val(c), val(d)
    for _ in 1:maxiter
        if gc < gd
            b, d, gd = d, c, gc
            c = b - φ * (b - a)
            gc = val(c)
        else
            a, c, gc = c, d, gd
            d = a + φ * (b - a)
            gd = val(d)
        end
        (b - a) <= 1e-12 * max(abs(a), abs(b), 1e-300) && break
    end
    x = 0.5 * (a + b)
    return val(x) <= min(gc, gd) ? x : (gc < gd ? c : d)
end

"""
    max_correctable_overlap(c::EFCCoupling; delta_threshold, torque_budget, safety_factor=1.0, model=:auto,
                            rotation_exponent=1.0, omega_reference=c.omega_reference) -> (; with_ntv, torque_only)

The largest intrinsic overlap the array can correct. `with_ntv`: the overlap beyond which
[`correction_current`](@ref) has no root (for the `:linear` model in closed form: the
quadratic's tangency `s·δ_thresh + C_c² T_0 / (4 s δ_thresh T_residual)`, or
`C_c √(T_0 / T_residual)` when the residual torque exhausts the budget before that; for
`:torque_balance` by bisection).
`torque_only`: the overlap cancelled by the current at which the whole field's torque alone
exhausts the budget (`C_c √(T_0 / T_full)`) or, with the balance, brings the reference rotation
to rest or breaks the balance.
`Inf` when there is no limit inside the model's reach.
"""
function max_correctable_overlap(c::EFCCoupling; delta_threshold::Real, torque_budget::Real, safety_factor::Real=1.0, model::Symbol=:auto,
    rotation_exponent::Real=1.0, omega_reference::Real=c.omega_reference)
    target = safety_factor * delta_threshold
    model = _resolve_model(c, model)
    if model === :linear
        t_res, t_full = abs(c.torque_residual_per_kat2), abs(c.torque_full_per_kat2)
        if t_res > 0
            # The quadratic's tangency, unless the residual torque exhausts the budget first (root at √(T_0/T_res)).
            tangency = target + c.delta_per_kat^2 * torque_budget / (4 * target * t_res)
            exhausted = c.delta_per_kat * sqrt(torque_budget / t_res)
            with_ntv = exhausted <= 2 * target ? tangency : exhausted
        else
            with_ntv = Inf
        end
        torque_only = t_full > 0 ? c.delta_per_kat * sqrt(torque_budget / t_full) : Inf
        return (; with_ntv, torque_only)
    end
    kw = (; delta_threshold, torque_budget, safety_factor, model, rotation_exponent, omega_reference)
    with_ntv = _largest_finite(δ -> correction_current(δ, c; kw...), target, 1e4 * target)
    # The whole field's torque alone brings the rotation to rest (factor 0) or breaks the balance.
    stops(I) = (f = threshold_factor(c, I; torque_budget, rotation_exponent, omega_reference, model, field=:full); isfinite(f) && f > 0 ? f : NaN)
    torque_only = c.delta_per_kat * _largest_finite(stops, 0.0, 1e4 * (target / c.delta_per_kat))
    return (; with_ntv, torque_only)
end

# Largest x in [lo, hi] for which f(x) is finite, assuming finiteness holds on [lo, x*) only; Inf if finite up to hi.
function _largest_finite(f, lo::Real, hi::Real)
    isfinite(f(hi)) && return Inf
    a, b = Float64(lo), Float64(hi)
    for _ in 1:200
        m = 0.5 * (a + b)
        if isfinite(f(m))
            a = m
        else
            b = m
        end
        (b - a) <= 1e-8 * max(b, 1e-300) && break
    end
    return a
end

"""
    efc_current_curve(c::EFCCoupling; delta_threshold, torque_budget, safety_factor=1.0, delta_max=15, npoints=500,
                      model=:auto, rotation_exponent=1.0, omega_reference=c.omega_reference) -> NamedTuple

The correction current against intrinsic overlap, `δ_ef` from `0` to `delta_max × δ_thresh`:
`delta_ef`, the linear `current_linear`, the NTV-limited `current_ntv` (`NaN` past the limit),
the threshold factor `threshold_factor` along it, the `model` used, and the two limits of
[`max_correctable_overlap`](@ref).
"""
function efc_current_curve(c::EFCCoupling; delta_threshold::Real, torque_budget::Real, safety_factor::Real=1.0, delta_max::Real=15, npoints::Int=500,
    model::Symbol=:auto, rotation_exponent::Real=1.0, omega_reference::Real=c.omega_reference)
    model = _resolve_model(c, model)
    kw = (; delta_threshold, torque_budget, safety_factor, model, rotation_exponent, omega_reference)
    δ = collect(range(0.0, delta_max * delta_threshold; length=npoints))
    lin = [correction_current(d, c; delta_threshold, torque_budget, safety_factor, ntv=false) for d in δ]
    ntv = [correction_current(d, c; kw...) for d in δ]
    factor = [isnan(I) ? NaN : threshold_factor(c, I; torque_budget, rotation_exponent, omega_reference, model) for I in ntv]
    limits = max_correctable_overlap(c; kw...)
    return (; delta_ef=δ, current_linear=lin, current_ntv=ntv, threshold_factor=factor, model, limits...)
end
