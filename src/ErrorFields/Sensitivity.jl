"""
    compute_coil_sensitivities(coil_sets, rc, equil, cfg, ctrl=ErrorFieldsControl(); psi, b_t0) -> CoilSensitivities

Linearize the control-surface spectrum of every coil set in `coil_sets` with respect to its six
rigid-body degrees of freedom, by central differences of the Biot-Savart spectrum on the surface
`psi` (the solve's integration limit, `ffs.psilim`) sampled once per toroidal mode of `rc`.
The nominal spectrum is evaluated too, so the sum over sets reproduces the run's forcing when
these are the run's coils.

Shift taps translate the whole set rigidly; tilt taps rotate it about the pivot named by
`ctrl.rotation_center`, in degrees. `cfg` supplies the boundary-grid resolution, `b_t0` the
axis toroidal field (tesla) stored for the δ normalization. Errors on a set carrying no
current, whose linearization would vanish identically.

The spectra are placed on `rc`'s (m, n) ordering and conformed to root-area-weighted field with
[`rootarea_field`](@ref), so [`coupling_overlap`](@ref) applies to them directly. Post-hoc use
from a `gpec.h5` goes through the `compute_coil_sensitivities(h5path, coil_sets; kwargs...)`
method.
"""
function compute_coil_sensitivities(
    coil_sets::Vector{CoilSet},
    rc::ResonantCoupling,
    equil::Equilibrium.PlasmaEquilibrium,
    cfg::CoilConfig,
    ctrl::ErrorFieldsControl=ErrorFieldsControl();
    psi::Float64,
    b_t0::Float64
)
    isempty(coil_sets) && throw(ArgumentError("compute_coil_sensitivities: no coil sets given"))
    ctrl.rotation_center in ("conductor", "set") ||
        throw(ArgumentError("rotation_center must be \"conductor\" or \"set\" (got \"$(ctrl.rotation_center)\")"))
    (ctrl.fd_step_shift_m > 0 && ctrl.fd_step_tilt_deg > 0) ||
        throw(ArgumentError("finite-difference steps must be positive (got $(ctrl.fd_step_shift_m) m, $(ctrl.fd_step_tilt_deg) deg)"))
    b_t0 > 0 || throw(ArgumentError("b_t0 must be a positive field magnitude (got $b_t0)"))
    for cs in coil_sets
        all(iszero, cs.currents) &&
            throw(ArgumentError("coil set \"$(cs.name)\" carries no current; its spectrum and sensitivities would vanish identically"))
    end

    N = length(rc.m_modes)
    nset = length(coil_sets)
    m_low, m_high = extrema(rc.m_modes)
    grids = [(n, CoilForcingGrid(equil, cfg, n; psi)) for n in sort(unique(rc.n_modes))]

    # One Biot-Savart pass per coil-set geometry, every toroidal mode on its own grid, placed
    # and conformed to b̃ on rc's column ordering.
    function spectrum(cs::CoilSet)
        modes = ForcingMode[]
        for (n, grid) in grids
            append!(modes, coil_forcing_modes(cs, grid, n, m_low, m_high))
        end
        return rootarea_field(rc, modes)
    end

    nominal = zeros(ComplexF64, N, nset)
    shift = zeros(ComplexF64, N, 3, nset)
    tilt = zeros(ComplexF64, N, 3, nset)
    shift_resid = zeros(3, nset)
    tilt_resid = zeros(3, nset)
    h_shift = ctrl.fd_step_shift_m
    h_tilt = ctrl.fd_step_tilt_deg

    for (j, cs) in enumerate(coil_sets)
        t_start = time()
        b0 = spectrum(cs)
        nominal[:, j] = b0
        pivot = ctrl.rotation_center == "set" ? _set_center(cs) : nothing

        first_norm = zeros(3, 2)
        second_norm = zeros(3, 2)
        for axis in 1:3, (kind, h, out) in ((:shift, h_shift, shift), (:tilt, h_tilt, tilt))
            plus = spectrum(_rigidly_perturbed(cs, kind, axis, h, pivot))
            minus = spectrum(_rigidly_perturbed(cs, kind, axis, -h, pivot))
            out[:, axis, j] = (plus .- minus) ./ (2h)
            col = kind === :shift ? 1 : 2
            first_norm[axis, col] = norm(plus .- minus)
            second_norm[axis, col] = norm(plus .+ minus .- 2 .* b0)
        end

        # A tap the set cannot feel (a vertical shift of an axisymmetric hoop at n ≥ 1) has a
        # first difference at round-off; measure its curvature against the set's largest tap
        # instead of dividing noise by noise.
        floor = 1e-6 * maximum(first_norm)
        resid = floor > 0 ? second_norm ./ max.(first_norm, floor) : zeros(3, 2)
        shift_resid[:, j] = resid[:, 1]
        tilt_resid[:, j] = resid[:, 2]

        worst = maximum(resid)
        worst > 1e-2 &&
            @warn "Coil set \"$(cs.name)\": finite-difference curvature $(@sprintf("%.2e", worst)) of the first difference; reduce fd_step_shift_m / fd_step_tilt_deg"
        ctrl.verbose &&
            @info "  $(cs.name): |b̃| = $(@sprintf("%.3e", norm(b0))) T, max |∂b̃/∂x| = $(@sprintf("%.3e", maximum(norm, eachslice(shift[:, :, j]; dims=2)))) T/m, " *
                  "max |∂b̃/∂θ| = $(@sprintf("%.3e", maximum(norm, eachslice(tilt[:, :, j]; dims=2)))) T/deg, curvature $(@sprintf("%.1e", worst)) ($(@sprintf("%.2f", time() - t_start)) s)"
    end

    return CoilSensitivities(
        [cs.name for cs in coil_sets], copy(rc.m_modes), copy(rc.n_modes), b_t0,
        nominal, shift, tilt, shift_resid, tilt_resid,
        [maximum(abs, cs.currents) for cs in coil_sets], [cs.nw for cs in coil_sets]
    )
end

"""
    _rigidly_perturbed(cs, kind, axis, h, pivot) -> CoilSet

The coil set moved rigidly by `h` along one degree of freedom: `kind = :shift` translates every
conductor by `h` metres along `axis` (x, y, z); `kind = :tilt` rotates every conductor by `h`
degrees about `axis` through `pivot`, or about its own arc-length centre when `pivot === nothing`.
Both go through `apply_transforms` with the toroidal modulation that makes the motion rigid
(`n_tilt = 0` for the shift, `n_tilt = 1` for the tilt).
"""
function _rigidly_perturbed(cs::CoilSet, kind::Symbol, axis::Int, h::Real, pivot)
    per_conductor = fill(Float64(h), cs.ncoil)
    component(a) = a == axis ? per_conductor : Float64[]
    if kind === :shift
        cfg = CoilSetConfig(; shiftx=component(1), shifty=component(2), shiftz=component(3))
        return apply_transforms(cs, cfg; n_tilt=0)
    end
    nom(i) = pivot === nothing ? Float64[] : fill(pivot[i], cs.ncoil)
    cfg = CoilSetConfig(; tiltx=component(1), tilty=component(2), tiltz=component(3),
        xnom=nom(1), ynom=nom(2), znom=nom(3), tilt_in_meters=false)
    return apply_transforms(cs, cfg; n_tilt=1)
end

"""
    _set_center(cs::CoilSet) -> (x0, y0, z0)

Arc-length-weighted centre of the whole coil set, over every conductor and strand: the pivot of
a rigid tilt of a multi-filament winding pack.
"""
function _set_center(cs::CoilSet)
    total = 0.0
    cx = cy = cz = 0.0
    for j in 1:cs.ncoil, k in 1:cs.s, l in 1:(cs.nsec-1)
        dl = hypot(cs.x[j, k, l+1] - cs.x[j, k, l], cs.y[j, k, l+1] - cs.y[j, k, l], cs.z[j, k, l+1] - cs.z[j, k, l])
        cx += dl * (cs.x[j, k, l] + cs.x[j, k, l+1]) / 2
        cy += dl * (cs.y[j, k, l] + cs.y[j, k, l+1]) / 2
        cz += dl * (cs.z[j, k, l] + cs.z[j, k, l+1]) / 2
        total += dl
    end
    total > 0 || return (0.0, 0.0, 0.0)
    return (cx / total, cy / total, cz / total)
end

"""
    sensitivity_table(sens::CoilSensitivities, dom::DominantCoupling; mode=1) -> SensitivityTable
    sensitivity_table(h5path; psi_low=0.0, psi_high=1.0, mode=1) -> SensitivityTable

Project a coil linearization onto singular mode `mode` of `dom` and normalize by the axis
toroidal field: `δ = dot(V[:, mode], b̃) / B_T0` for the nominal spectrum and each derivative,
the dimensionless overlap and its sensitivities per coil set. The window and mode are analysis
choices, so re-evaluate freely on the same `sens`; the file form rebuilds the coupling from
`gpec.h5`, windows it to `psi_low ≤ ψ_N ≤ psi_high`, and reads the stored linearization.
"""
function sensitivity_table(sens::CoilSensitivities, dom::DominantCoupling; mode::Int=1)
    1 <= mode <= length(dom.singular_values) ||
        throw(ArgumentError("mode $mode is outside the $(length(dom.singular_values)) singular modes of the decomposition"))
    size(dom.right_singular_vectors, 1) == length(sens.m_modes) ||
        throw(DimensionMismatch("the decomposition acts on $(size(dom.right_singular_vectors, 1)) modes but the sensitivities carry $(length(sens.m_modes))"))
    v = dom.right_singular_vectors[:, mode]
    project(b̃) = dot(v, b̃) / sens.b_t0
    nset = length(sens.coil_names)

    delta_nominal = [project(view(sens.nominal_field, :, j)) for j in 1:nset]
    shift = [project(view(sens.shift_sensitivity, :, a, j)) for a in 1:3, j in 1:nset]
    tilt = [project(view(sens.tilt_sensitivity, :, a, j)) for a in 1:3, j in 1:nset]
    inplane_rms(S) = [sqrt((abs2(S[1, j]) + abs2(S[2, j])) / 2) for j in 1:nset]
    cancelling(S) = [cancelling_offset(delta_nominal[j], S[1, j], S[2, j])[i] for i in 1:2, j in 1:nset]

    return SensitivityTable(copy(sens.coil_names), mode, delta_nominal, shift, tilt,
        inplane_rms(shift), inplane_rms(tilt), cancelling(shift), cancelling(tilt))
end

function sensitivity_table(h5path::AbstractString; psi_low::Real=0.0, psi_high::Real=1.0, mode::Int=1)
    rc = ResonantCoupling(h5path)
    dom = dominant_coupling(rc; psi_low, psi_high)
    return sensitivity_table(CoilSensitivities(h5path), dom; mode)
end

"""
    cancelling_offset(δ_nominal, S_x, S_y) -> (Δx, Δy)

The real in-plane displacement that cancels a coil set's nominal overlap to linear order,
`S_x·Δx + S_y·Δy = −δ_nominal`, as the least-squares solution of the 2×2 real system on the
real and imaginary parts. For an axisymmetric set, where `S_y = ±i·S_x`, this is the complex
offset `−δ_nominal / S_x` split into its components. A degenerate pair (`S_x` and `S_y` real
multiples of each other) returns the minimum-norm solution.
"""
function cancelling_offset(δ_nominal::Number, S_x::Number, S_y::Number)
    A = [real(S_x) real(S_y); imag(S_x) imag(S_y)]
    Δ = pinv(A) * [-real(δ_nominal), -imag(δ_nominal)]
    return (Δ[1], Δ[2])
end
