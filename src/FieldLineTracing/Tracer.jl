"""
    Tracer

Field-line integration built on `OrdinaryDiffEq` (`ODEProblem` + `Tsit5`). `Tsit5` (5th
order) is well matched to Poincaré tracing — much cheaper per step than `Vern9` at the
loose tolerances field lines need. Produces Poincaré punctures, connection lengths, and
wall footprints for a `FieldModel`.
"""

"""
    _section_times(n_transits, planes, period) -> Vector{Float64}

Sorted independent-variable values at which to record Poincaré punctures: every
`plane + k·period` for `k = 0 … n_transits`, within `[0, n_transits·period]`.
"""
function _section_times(n_transits::Int, planes::AbstractVector{Float64}, period::Float64)
    times = Float64[]
    for k in 0:n_transits
        for p in planes
            t = k * period + mod(p, period)
            t <= n_transits * period + 1e-9 && push!(times, t)
        end
    end
    return sort!(unique(times))
end

"""
    trace_flux!(state, model, ctrl)

Trace field lines in `(ψ,θ,ζ)` and accumulate Poincaré punctures, connection lengths, and
laminar/stochastic classification. Lines are launched on a ψ_N grid at `θ = 0`.
"""
function trace_flux!(state::FieldLineTracingState, model::FieldModel, ctrl::FieldLineTracingControl)
    planes = Float64.(ctrl.phi_planes)
    section_zeta = [mod(p / (2π), 1.0) for p in planes]
    times = _section_times(ctrl.n_transits, section_zeta, 1.0)
    tspan = (0.0, Float64(ctrl.n_transits))

    psi_hi = min(ctrl.psi_end, model.psi_control - 1e-3)
    psi_launch = range(ctrl.psi_start, psi_hi; length=ctrl.n_lines)

    # Terminate when a line escapes the control surface (open/stochastic ejection).
    escaped = Ref(false)
    condition(u, t, integ) = u[1] >= model.psi_control - 1e-4
    affect!(integ) = (escaped[] = true; terminate!(integ))
    esc_cb = DiscreteCallback(condition, affect!)

    for (line_id, psi0) in enumerate(psi_launch)
        escaped[] = false
        u0 = [Float64(psi0), 0.0]
        prob = ODEProblem(fieldline_rhs_flux!, u0, tspan, model)
        sol = solve(prob, Tsit5(); reltol=ctrl.tol, abstol=1e-9,
                    saveat=times, callback=esc_cb, save_everystep=false)

        last_zeta = 0.0
        for (t, u) in zip(sol.t, sol.u)
            psi = clamp(u[1], 0.0, 1.0)
            theta = mod(u[2], 1.0)
            R, Z = _rz_at(model.equil, psi, theta)
            # Attribute the puncture to the nearest requested plane.
            zeta_frac = mod(t, 1.0)
            plane = planes[argmin([abs(zeta_frac - sz) for sz in section_zeta])]
            push!(state.punc_R, R); push!(state.punc_Z, Z)
            push!(state.punc_psi, psi); push!(state.punc_theta, theta)
            push!(state.punc_line, line_id); push!(state.punc_phi, plane)
            last_zeta = t
        end

        if ctrl.compute_connection_length
            R0, Z0 = _rz_at(model.equil, Float64(psi0), 0.0)
            push!(state.cl_R, R0); push!(state.cl_Z, Z0); push!(state.cl_psi, Float64(psi0))
            push!(state.cl_length, last_zeta)          # toroidal transits completed
            push!(state.cl_class, escaped[] ? 2 : 0)   # 2 = escaped control surface, 0 = confined
        end
    end
    return nothing
end

"""
    trace_real!(state, model, ctrl, wall_R, wall_Z)

Trace field lines in cylindrical `(R,Z,φ)` and accumulate Poincaré punctures, connection
lengths, and wall footprints. Lines are launched along the chord
`R ∈ [R_start, R_end]` at `Z = Z_launch`.
"""
function trace_real!(state::FieldLineTracingState, model::FieldModel, ctrl::FieldLineTracingControl,
                     wall_R::Vector{Float64}, wall_Z::Vector{Float64})
    planes = Float64.(ctrl.phi_planes)
    times = _section_times(ctrl.n_transits, planes, 2π)
    tspan = (0.0, 2π * ctrl.n_transits)

    R_lo = ctrl.R_start; R_hi = ctrl.R_end
    if R_lo == 0.0 && R_hi == 0.0
        # Default launch chord: outboard midplane from just inside the axis to the LCFS.
        R_axis = model.equil.ro
        R_edge, _ = _rz_at(model.equil, model.psi_control, 0.0)
        R_lo = R_axis + 0.05 * (R_edge - R_axis)
        R_hi = R_edge + 0.6 * (R_edge - R_axis)     # extend past LCFS for the SOL/footprint region
    end
    Z0 = isnan(ctrl.Z_launch) ? model.equil.zo : ctrl.Z_launch
    R_launch = range(R_lo, R_hi; length=ctrl.n_lines)

    have_wall = !isempty(wall_R)
    hit = Ref(false)
    strike = Ref((0.0, 0.0, 0.0))
    if have_wall
        wcond(u, t, integ) = _point_in_polygon(u[1], u[2], wall_R, wall_Z) ? 1.0 : -1.0
        function waffect!(integ)
            hit[] = true
            strike[] = (integ.u[1], integ.u[2], mod(integ.t, 2π))
            terminate!(integ)
        end
        wall_cb = ContinuousCallback(wcond, waffect!)
    end

    for (line_id, R0) in enumerate(R_launch)
        hit[] = false
        u0 = [Float64(R0), Z0]
        prob = ODEProblem(fieldline_rhs_real!, u0, tspan, model)
        sol = have_wall ?
              solve(prob, Tsit5(); reltol=ctrl.tol, abstol=1e-9, saveat=times,
                    callback=wall_cb, save_everystep=false) :
              solve(prob, Tsit5(); reltol=ctrl.tol, abstol=1e-9, saveat=times,
                    save_everystep=false)

        last_phi = 0.0
        for (t, u) in zip(sol.t, sol.u)
            R, Z = u[1], u[2]
            plane = planes[argmin([abs(mod(t, 2π) - p) for p in planes])]
            psi, theta, conv = invert_rz_to_flux(model.equil, R, Z; psi_max=model.psi_control)
            push!(state.punc_R, R); push!(state.punc_Z, Z)
            push!(state.punc_psi, conv ? psi : NaN); push!(state.punc_theta, conv ? theta : NaN)
            push!(state.punc_line, line_id); push!(state.punc_phi, plane)
            last_phi = t
        end

        if ctrl.compute_footprints && hit[]
            sR, sZ, sphi = strike[]
            push!(state.fp_R, sR); push!(state.fp_Z, sZ); push!(state.fp_phi, sphi)
        end
        if ctrl.compute_connection_length
            psi0, _, _ = invert_rz_to_flux(model.equil, Float64(R0), Z0; psi_max=model.psi_control)
            push!(state.cl_R, Float64(R0)); push!(state.cl_Z, Z0); push!(state.cl_psi, psi0)
            push!(state.cl_length, last_phi * model.equil.ro)   # ≈ connection length [m]
            push!(state.cl_class, hit[] ? 2 : 0)
        end
    end
    return nothing
end

"""
    _point_in_polygon(x, y, px, py) -> Bool

Ray-casting point-in-polygon test for the wall contour `(px,py)`.
"""
function _point_in_polygon(x::Float64, y::Float64, px::Vector{Float64}, py::Vector{Float64})
    n = length(px)
    inside = false
    j = n
    @inbounds for i in 1:n
        if ((py[i] > y) != (py[j] > y)) &&
           (x < (px[j] - px[i]) * (y - py[i]) / (py[j] - py[i] + eps()) + px[i])
            inside = !inside
        end
        j = i
    end
    return inside
end
