"""
    DirectBField

Internal mutable struct to hold B-field components and their derivatives at a point.
It is used as a temporary workspace to avoid allocations in tight loops.
"""
@kwdef mutable struct DirectBField
    psi::Float64 = 0.0
    psir::Float64 = 0.0   # d(psi)/dr
    psiz::Float64 = 0.0   # d(psi)/dz
    psirz::Float64 = 0.0  # d2(psi)/drdz
    psirr::Float64 = 0.0  # d2(psi)/drdr
    psizz::Float64 = 0.0  # d2(psi)/dzdz
    f::Float64 = 0.0      # F = R*Bt
    f1::Float64 = 0.0     # dF/d(psi_norm)
    p::Float64 = 0.0      # mu0*Pressure
    p1::Float64 = 0.0     # dP/d(psi_norm)
    br::Float64 = 0.0     # Br
    bz::Float64 = 0.0     # Bz
    brr::Float64 = 0.0    # d(Br)/dr
    brz::Float64 = 0.0    # d(Br)/dz
    bzr::Float64 = 0.0    # d(Bz)/dr
    bzz::Float64 = 0.0    # d(Bz)/dz
end

"""
    FieldLineDerivParams

Constant parameters of the field-line ODE, read by `direct_fieldline_der!` and `arclength_fieldline_der!`.
"""
struct FieldLineDerivParams{I2D<:FastInterpolations.CubicInterpolantND,S<:FastInterpolations.CubicSeriesInterpolant,D}
    ro::Float64
    zo::Float64
    psi_in::I2D
    sq_in::S
    sq_in_deriv::D
    psio::Float64
    power_bp::Int
    power_b::Int
    power_r::Int
    power_rc::Int  # minor radius rfac = √((R-R₀)²+(Z-Z₀)²) power exponent
    bfield::DirectBField
end

"""
    direct_get_bfield!(bf_out, r, z, psi_in, sq_in, sq_in_deriv, psio; derivs=0)

Evaluates ψ, the profiles F and P, and the poloidal field with its derivatives at (R, Z), in place in `bf_out`.
Port of Fortran `direct_get_bfield`.

## Arguments

  - `psi_in`: 2D cubic interpolant of the poloidal flux `ψ(R,Z)`
  - `sq_in`, `sq_in_deriv`: profile series `F(ψ_norm)`, `P(ψ_norm)` and its derivative
  - `psio`: ψ at the magnetic axis; ψ_norm = 1 - ψ/psio
  - `derivs`: number of ψ derivatives to evaluate (0, 1, or 2)
"""
@with_pool pool function direct_get_bfield!(
    bf_out::DirectBField,
    r::Float64,
    z::Float64,
    psi_in::FastInterpolations.CubicInterpolantND,
    sq_in::FastInterpolations.CubicSeriesInterpolant,
    sq_in_deriv,
    psio::Float64;
    derivs::Int=0
)
    # Evaluate 2D interpolant for psi(r,z) and its derivatives
    if derivs == 0
        bf_out.psi = psi_in((r, z))
    elseif derivs == 1
        bf_out.psi = psi_in((r, z))
        bf_out.psir = psi_in((r, z); deriv=DerivOp(1, 0))
        bf_out.psiz = psi_in((r, z); deriv=DerivOp(0, 1))
    else # derivs >= 2
        bf_out.psi = psi_in((r, z))
        bf_out.psir = psi_in((r, z); deriv=DerivOp(1, 0))
        bf_out.psiz = psi_in((r, z); deriv=DerivOp(0, 1))
        bf_out.psirr = psi_in((r, z); deriv=DerivOp(2, 0))
        bf_out.psirz = psi_in((r, z); deriv=DerivOp(1, 1))
        bf_out.psizz = psi_in((r, z); deriv=DerivOp(0, 2))
    end

    # Evaluate magnetic fields from equilibrium profiles
    psi_norm = (psio > 1e-12) ? (1.0 - bf_out.psi / psio) : 0.0
    psi_norm = clamp(psi_norm, 0.0, 1.0)

    f_sq = acquire!(pool, eltype(sq_in.y), n_series(sq_in))
    f1_sq = acquire!(pool, eltype(sq_in_deriv.parent.y), n_series(sq_in_deriv.parent))
    sq_in(f_sq, psi_norm)
    sq_in_deriv(f1_sq, psi_norm)
    bf_out.f = f_sq[1]  # F = R*Bt
    bf_out.f1 = f1_sq[1] # dF/dψ
    bf_out.p = f_sq[2]  # μ0*Pressure
    bf_out.p1 = f1_sq[2] # dP/dψ

    (derivs == 0) && return

    # Evaluate B-field derivative components
    bf_out.br = bf_out.psiz / r # Br = (1/R) * ∂ψ/∂Z
    bf_out.bz = -bf_out.psir / r # Bz = -(1/R) * ∂ψ/∂R

    (derivs == 1) && return

    # Evaluate more derivatives of B-field components
    bf_out.brr = (bf_out.psirz - bf_out.br) / r
    bf_out.brz = bf_out.psizz / r
    bf_out.bzr = -(bf_out.psirr + bf_out.bz) / r
    bf_out.bzz = -bf_out.psirz / r
end

# Numerical controls of the magnetic-axis search.
const AXIS_MARCH_STEPS = 200      # midplane march steps allowed before giving up on a B_z sign change
const AXIS_NEWTON_ITERATIONS = 200
const AXIS_SINGULAR_DET = 1e-20   # absolute floor on |det ∂(B_R,B_Z)/∂(R,Z)| in T²/m²; it does not scale with ψ₀ or machine size
const AXIS_STEP_RTOL = 1e-12      # Newton step test, relative to R
const AXIS_RESIDUAL_RTOL = 1e-10  # bound on |B_p| relative to B_scale, applied on the restart path only
const AXIS_SEED_CELL_TOL = 0.1    # the bisection sweep has settled when a pass moves the point by less than this many cells
const AXIS_SEED_PASSES = 50
const AXIS_PSI_RATIO = (0.5, 2.0) # accepted band of ψ(axis)/ψ₀; every reader puts ψ = ψ₀ at the axis
const SEPARATRIX_STEPS = 2000     # the Fortran allows more Newton steps here than for the axis solve

_axis_jacobian_det(bf::DirectBField) = bf.brr * bf.bzz - bf.brz * bf.bzr

# Poloidal-field scale |ψ₀| / (R ΔR), about a sixth of the edge poloidal field.
_axis_field_scale(rp::DirectRunInput, r::Float64) = abs(rp.psio) / (r * (rp.rmax - rp.rmin))

"""
    _midplane_axis_guess(rp, bfield, sq_in_deriv) -> (r, z, step)

Initial guess for the axis: march outward along the grid's midplane `Z`, in steps of a twentieth of the grid width, until
`B_z ≥ 0`.
"""
function _midplane_axis_guess(rp::DirectRunInput, bfield::DirectBField, sq_in_deriv)
    r = (rp.rmax + rp.rmin) / 2.0
    z = (rp.zmax + rp.zmin) / 2.0
    step = (rp.rmax - rp.rmin) / 20.0
    for _ in 1:AXIS_MARCH_STEPS
        direct_get_bfield!(bfield, r, z, rp.psi_in, rp.sq_in, sq_in_deriv, rp.psio; derivs=1)
        bfield.bz >= 0.0 && return r, z, step
        r += step
    end
    error("Took too many iterations to get bz=0.")
end

"""
    _axis_newton(rp, bfield, sq_in_deriv, r, z, cap) -> (r, z, failure)

Newton iteration for the O-point `B_R = B_Z = 0`, each step capped at `cap`. `failure` is `nothing` once a step passes
`|δR|, |δZ| ≤ AXIS_STEP_RTOL × R`, else why the iteration stopped. A singular Jacobian stops it, where the Fortran divides
by the determinant unguarded.
"""
function _axis_newton(rp::DirectRunInput, bfield::DirectBField, sq_in_deriv, r::Float64, z::Float64, cap::Float64)
    for _ in 1:AXIS_NEWTON_ITERATIONS
        direct_get_bfield!(bfield, r, z, rp.psi_in, rp.sq_in, sq_in_deriv, rp.psio; derivs=2)
        det = _axis_jacobian_det(bfield)
        abs(det) < AXIS_SINGULAR_DET && return r, z, "stopped on a singular Jacobian, |det ∂(B_R,B_Z)/∂(R,Z)| < $AXIS_SINGULAR_DET"
        # Δx = -J⁻¹ F
        δr = (bfield.brz * bfield.bz - bfield.bzz * bfield.br) / det
        δz = (bfield.bzr * bfield.br - bfield.brr * bfield.bz) / det
        step = hypot(δr, δz)
        step > cap && ((δr, δz) = (δr * cap / step, δz * cap / step))
        r += δr
        z += δz
        abs(δr) <= AXIS_STEP_RTOL * abs(r) && abs(δz) <= AXIS_STEP_RTOL * abs(r) && return r, z, nothing
    end
    return r, z, "stopped without convergence in $AXIS_NEWTON_ITERATIONS iterations"
end

"""
    _axis_bisection_seed(rp, bfield, sq_in_deriv, r, z, halfwidth; passes) -> (r, z, failure)

Seed for the Newton restart: alternate 1-D bisections on `B_Z` along `R` and on `B_R` along `Z`, each bracketed by doubling
`halfwidth` up to the grid edges. The sweep is a nonlinear Gauss–Seidel iteration that contracts by ψ_RZ²/(ψ_RR ψ_ZZ) per
pass, so stopping once a pass moves the point by less than `AXIS_SEED_CELL_TOL` cells leaves it within about a cell of the
axis for contractions up to 0.9. `failure` names a missing sign change or a sweep that did not settle in `passes` passes.
"""
function _axis_bisection_seed(rp::DirectRunInput, bfield::DirectBField, sq_in_deriv, r::Float64, z::Float64, halfwidth::Float64;
    passes::Int=AXIS_SEED_PASSES)
    _bz(rr, zz) = (direct_get_bfield!(bfield, rr, zz, rp.psi_in, rp.sq_in, sq_in_deriv, rp.psio; derivs=1); bfield.bz)
    _br(rr, zz) = (direct_get_bfield!(bfield, rr, zz, rp.psi_in, rp.sq_in, sq_in_deriv, rp.psio; derivs=1); bfield.br)
    hr = (rp.rmax - rp.rmin) / (length(rp.psi_in_xs) - 1)
    hz = (rp.zmax - rp.zmin) / (length(rp.psi_in_ys) - 1)
    # Nearest sign change of f around x, bisected to a hundredth of a cell h; `nothing` if the whole grid line has none.
    function _bracketed_root(f, x, xmin, xmax, h)
        hw = halfwidth
        while true
            lo, hi = max(x - hw, xmin), min(x + hw, xmax)
            sign(f(lo)) * sign(f(hi)) <= 0 && return find_zero(f, (lo, hi), Roots.Bisection(); xatol=0.01 * h)
            (lo == xmin && hi == xmax) && return nothing
            hw *= 2
        end
    end
    for _ in 1:passes
        r_new = _bracketed_root(rr -> _bz(rr, z), r, rp.rmin, rp.rmax, hr)
        r_new === nothing && return r, z, "B_z has no sign change along Z = $z"
        z_new = _bracketed_root(zz -> _br(r_new, zz), z, rp.zmin, rp.zmax, hz)
        z_new === nothing && return r, z, "B_r has no sign change along R = $r_new"
        moved = max(abs(r_new - r) / hr, abs(z_new - z) / hz)
        r, z = r_new, z_new
        moved < AXIS_SEED_CELL_TOL && return r, z, nothing
    end
    return r, z, "the bisection sweep did not settle in $passes passes"
end

"""
    _axis_failure(rp, bfield, sq_in_deriv, r, z; bound_residual) -> failure

Why `(r, z)` is not an acceptable magnetic axis, or `nothing`. It must lie on the ψ grid (a root off it is an artifact of
spline extrapolation), be an O-point (`det ∂(B_R,B_Z)/∂(R,Z) = det(Hess ψ)/R² > 0`; an X-point has `det < 0`) that is a
maximum of ψ (`ψ_RR < 0` and `ψ > 0`), and carry ψ within `AXIS_PSI_RATIO` of ψ₀, which rejects a secondary maximum
elsewhere in the grid. With `bound_residual`, `|B_p| ≤ AXIS_RESIDUAL_RTOL × B_scale` must hold as well.
"""
function _axis_failure(rp::DirectRunInput, bfield::DirectBField, sq_in_deriv, r::Float64, z::Float64; bound_residual::Bool)
    (rp.rmin <= r <= rp.rmax && rp.zmin <= z <= rp.zmax) ||
        return "converged at (R, Z) = ($r, $z), outside the ψ grid R ∈ [$(rp.rmin), $(rp.rmax)], Z ∈ [$(rp.zmin), $(rp.zmax)]"
    direct_get_bfield!(bfield, r, z, rp.psi_in, rp.sq_in, sq_in_deriv, rp.psio; derivs=2)
    det = _axis_jacobian_det(bfield)
    residual = hypot(bfield.br, bfield.bz)
    det > 0 || return "converged at (R, Z) = ($r, $z), which is not an O-point (det ∂(B_R,B_Z)/∂(R,Z) = $det ≤ 0, |B_p| = $residual T)"
    (bfield.psirr < 0 && bfield.psi > 0) ||
        return "converged at (R, Z) = ($r, $z), an O-point that is not the maximum of ψ (ψ = $(bfield.psi), ψ_RR = $(bfield.psirr))"
    lo, hi = AXIS_PSI_RATIO
    lo < bfield.psi / rp.psio < hi ||
        return "converged at (R, Z) = ($r, $z), a maximum of ψ that is not the main axis (ψ/ψ₀ = $(bfield.psi / rp.psio), outside ($lo, $hi))"
    if bound_residual
        bound = AXIS_RESIDUAL_RTOL * _axis_field_scale(rp, r)
        residual <= bound || return "converged at (R, Z) = ($r, $z) with |B_p| = $residual T, above $AXIS_RESIDUAL_RTOL × B_scale = $bound T"
    end
    return nothing
end

# Field diagnostics at an accepted axis, for the restart warning.
function _axis_diagnostics(rp::DirectRunInput, bfield::DirectBField, sq_in_deriv, r::Float64, z::Float64)
    direct_get_bfield!(bfield, r, z, rp.psi_in, rp.sq_in, sq_in_deriv, rp.psio; derivs=2)
    residual = hypot(bfield.br, bfield.bz)
    return "at (R, Z) = ($r, $z) with |B_p| = $residual T ($(residual / _axis_field_scale(rp, r)) × B_scale) and " *
           "det ∂(B_R,B_Z)/∂(R,Z) = $(_axis_jacobian_det(bfield)) T²/m²"
end

"""
    _find_magnetic_axis(rp, bfield, sq_in_deriv, r0, z0, march_step) -> (r, z)

The magnetic axis, by the first attempt whose Newton result passes `_axis_failure`: plain Newton from the midplane guess
`(r0, z0)`, accepted on the step test alone, then Newton restarted from
a bisection seed with steps capped at one march step and the residual bound added. A restart warns with the field at the
axis; two failures error.
"""
function _find_magnetic_axis(rp::DirectRunInput, bfield::DirectBField, sq_in_deriv, r0::Float64, z0::Float64, march_step::Float64)
    restart_cap = march_step    # Newton step cap on the restart path
    seed_halfwidth = march_step # initial bracket half-width of the bisection seed
    attempts = (
        (label="Newton from the midplane guess", seed=() -> (r0, z0, nothing), cap=Inf, bound_residual=false),
        (label="Newton restarted from a first-derivative bisection seed",
            seed=() -> _axis_bisection_seed(rp, bfield, sq_in_deriv, r0, z0, seed_halfwidth), cap=restart_cap, bound_residual=true)
    )
    failures = String[]
    for a in attempts
        rs, zs, why = a.seed()
        if why !== nothing
            push!(failures, "$(a.label): the seed failed, $why")
            continue
        end
        r, z, why = _axis_newton(rp, bfield, sq_in_deriv, rs, zs, a.cap)
        why === nothing && (why = _axis_failure(rp, bfield, sq_in_deriv, r, z; bound_residual=a.bound_residual))
        if why === nothing
            isempty(failures) || @warn "$(join(failures, "; ")); magnetic axis found by $(a.label) " *
                  "$(_axis_diagnostics(rp, bfield, sq_in_deriv, r, z)). Check ψ near the axis in the equilibrium file."
            return r, z
        end
        push!(failures, "$(a.label) $why")
    end
    error("Failed to find magnetic axis: $(join(failures, "; ")).")
end

"""
    _separatrix_crossing(rp, bfield, sq_in_deriv, ro, zo, r_edge, label) -> r

Midplane separatrix crossing on the `r_edge` side of the axis: unbracketed Newton `δR = −ψ/ψ_R` toward ψ = 0 of the
renormalized spline, as in the Fortran, so that it extrapolates past the grid edge for limited equilibria whose box hugs
the LCFS. Each failed attempt moves the initial guess, a blend of `r_edge` and the axis, inward.
"""
function _separatrix_crossing(rp::DirectRunInput, bfield::DirectBField, sq_in_deriv, ro::Float64, zo::Float64, r_edge::Float64, label::String)
    for restart in 0:5
        r = ((3 - 0.5 * restart) * r_edge + ro) / (4 - 0.5 * restart)
        for _ in 1:SEPARATRIX_STEPS
            direct_get_bfield!(bfield, r, zo, rp.psi_in, rp.sq_in, sq_in_deriv, rp.psio; derivs=1)
            dr = -bfield.psi / bfield.psir
            r += dr
            if abs(dr) <= AXIS_STEP_RTOL * abs(r)
                @info "$label separatrix found at R = $(@sprintf("%.3f", r))"
                return r
            end
        end
    end
    error("Took too many steps to find $label separatrix.")
end

"""
    direct_position!(raw_profile)

Finds the magnetic axis (O-point, see `_find_magnetic_axis`) and the inboard and outboard separatrix crossings on its
midplane, and replaces the `ψ(R, Z)` spline in `raw_profile` by one renormalized so that ψ at the axis equals ψ₀.
A g-file whose header signs disagree with its ψ grid errors here.

## Returns

  - `ro`, `zo`: magnetic axis coordinates, in m
  - `rs1`, `rs2`: R of the inboard and outboard separatrix crossings, in m
"""
function direct_position!(raw_profile::DirectRunInput)
    bfield = DirectBField()
    sq_in_deriv = deriv1(raw_profile.sq_in)

    r0, z0, march_step = _midplane_axis_guess(raw_profile, bfield, sq_in_deriv)
    ro, zo = _find_magnetic_axis(raw_profile, bfield, sq_in_deriv, r0, z0, march_step)
    @info "Magnetic axis found at R = $(@sprintf("%.3f", ro)), Z = $(@sprintf("%.3f", zo))"

    # Renormalize psi based on the value at the magnetic axis
    direct_get_bfield!(bfield, ro, zo, raw_profile.psi_in, raw_profile.sq_in, sq_in_deriv, raw_profile.psio; derivs=0)
    x_coords = raw_profile.psi_in_xs
    y_coords = raw_profile.psi_in_ys
    # partials[1, :, :] holds the nodal ψ values
    new_psi_fs = raw_profile.psi_in.nodal_derivs.partials[1, :, :] .* raw_profile.psio / bfield.psi
    raw_profile.psi_in = cubic_interp((x_coords, y_coords), new_psi_fs; extrap=ExtendExtrap())

    rs1 = _separatrix_crossing(raw_profile, bfield, sq_in_deriv, ro, zo, raw_profile.rmin, "inboard")
    rs2 = _separatrix_crossing(raw_profile, bfield, sq_in_deriv, ro, zo, raw_profile.rmax, "outboard")

    return ro, zo, rs1, rs2
end

"""
    sfl_sample_time(sol, ts, sfl, sfl_index, x) -> (t, lo)

Solver time `t` at which the normalised straight-fieldline angle reaches `x` ∈ [0, 1], and the index
`lo` of the saved step that starts its bracket. `sfl` is the SFL integral ∫jac·dl/Bp at the saved
times `ts`; it is monotone, so it brackets the root to one step, and Brent finishes on component
`sfl_index` of the dense solution.
"""
function sfl_sample_time(sol, ts::Vector{Float64}, sfl::AbstractVector{Float64}, sfl_index::Int, x::Float64)
    n = length(ts)
    x <= 0 && return ts[1], 1
    x >= 1 && return ts[n], n - 1
    target = x * sfl[n]
    hi = clamp(searchsortedfirst(sfl, target), 2, n)
    lo = hi - 1
    flo, fhi = sfl[lo] - target, sfl[hi] - target
    flo == 0 && return ts[lo], lo
    fhi == 0 && return ts[hi], lo
    flo < 0 < fhi || error("sfl_sample_time: SFL integral is not monotone across a solver step (x = $x)")
    return find_zero(t -> sol(t)[sfl_index] - target, (ts[lo], ts[hi]), Roots.Brent()), lo
end

"""
    sample_trace_at_sfl_angles(sol, y_out, sfl_index, theta_nodes, state_row) -> Matrix{Float64}

Sample a traced flux surface at the straight-fieldline angles `theta_nodes` (0 to 1). Row `i` is
`state_row(sol(t), t, lo)` at the time `t` where the normalised SFL angle reaches `theta_nodes[i]`
(see `sfl_sample_time`), in the 5-column `y_out` layout. The last row is the saved closed endpoint,
so the surface totals `y_out[end, :]` are exact. Every surface sharing the same abscissae keeps the
resample error from varying between neighbouring surfaces. Every field-line tracer must return its
surface through this function.
"""
function sample_trace_at_sfl_angles(sol, y_out::Matrix{Float64}, sfl_index::Int, theta_nodes, state_row::F) where {F}
    ts = sol.t::Vector{Float64}
    sfl = view(y_out, :, 5)
    samples = Matrix{Float64}(undef, length(theta_nodes), 5)
    for i in 1:(length(theta_nodes)-1)
        t, lo = sfl_sample_time(sol, ts, sfl, sfl_index, Float64(theta_nodes[i]))
        samples[i, :] .= state_row(sol(t), t, lo)
    end
    samples[end, :] .= @view y_out[end, :]
    return samples
end

"""
    direct_fieldline_int(psifac, raw_profile, ro, zo, rs2, theta_nodes) -> (y_out, bfield)

Traces the flux surface `psifac` (ψ_norm) in the geometric angle η from the outboard midplane, refining onto the
surface after every step.

## Arguments

  - `ro`, `zo`: magnetic axis coordinates, in m
  - `rs2`: R of the outboard separatrix crossing, in m
  - `theta_nodes`: straight-fieldline angles (0 to 1) at which to sample the surface

## Returns

  - `y_out`: rows at `theta_nodes` (see `sample_trace_at_sfl_angles`), columns η, ∫dl/Bp, rfac (distance from the axis),
    ∫dl/(R²Bp), ∫jac·dl/Bp
  - `bfield`: the `DirectBField` at the start point
"""
function direct_fieldline_int(psifac::Float64, raw_profile::DirectRunInput, ro::Float64, zo::Float64, rs2::Float64,
    theta_nodes)::Tuple{Matrix{Float64},DirectBField}

    # Find the starting point on the flux surface (outboard midplane)
    psi0_guess = raw_profile.psio * (1.0 - psifac)
    r = ro + sqrt(psifac) * (rs2 - ro)
    z = zo
    bfield = DirectBField()
    sq_in_deriv = deriv1(raw_profile.sq_in)

    # Refine starting R: find r where ψ(r, zo) = ψ₀. The df closure reads bfield.psir
    # which is populated by the preceding f call — Newton guarantees f is evaluated first.
    r = find_zero(
        (r -> (direct_get_bfield!(bfield, r, z, raw_profile.psi_in, raw_profile.sq_in, sq_in_deriv, raw_profile.psio; derivs=1); bfield.psi - psi0_guess),
            _ -> bfield.psir),
        r, Roots.Newton()
    )

    direct_get_bfield!(bfield, r, z, raw_profile.psi_in, raw_profile.sq_in, sq_in_deriv, raw_profile.psio; derivs=2)
    psi0 = bfield.psi

    # Initial state [∫dl/Bp, rfac, ∫dl/(R²Bp), ∫jac·dl/Bp]
    u0 = zeros(Float64, 4)
    u0[2] = sqrt((r - ro)^2 + (z - zo)^2)

    # The ODE right-hand side overwrites its own field buffer, so keep `bfield` at the start point (F, P at ψ₀).
    bfield_ode = DirectBField()
    equil_config = raw_profile.config
    params = FieldLineDerivParams(ro, zo, raw_profile.psi_in, raw_profile.sq_in, sq_in_deriv, raw_profile.psio,
        equil_config.power_bp, equil_config.power_b, equil_config.power_r, equil_config.power_rc, bfield_ode)

    # Use a callback to refine the solution at each step to stay on the flux surface
    function refine_affect!(integrator)
        integrator.u[2] = direct_refine(integrator.u[2], integrator.t, psi0, params)
    end

    # Save each step before refinement, as the Fortran does
    callback = DiscreteCallback((u, t, i) -> true, refine_affect!; save_positions=(true, false))

    prob = ODEProblem{true}(direct_fieldline_der!, u0, (0.0, 2π), params)
    # Dense output so the surface can be sampled at prescribed SFL angles between solver steps.
    rtol = equil_config.flux_surface_rtol
    sol = solve(prob, Vern9(); callback=callback, reltol=rtol, abstol=equil_abstol(rtol), dt=2π / 200, adaptive=true, dense=true)

    # A failed solve returns a truncated solution instead of throwing; check both the retcode and
    # that the field line reached η = 2π, since a callback can end it early and still report Success.
    if sol.retcode != ReturnCode.Success
        error(
            "direct_fieldline_int: field-line integration failed at psifac = " *
            "$(@sprintf("%.6f", psifac)) (retcode $(sol.retcode)); the flux surface did not " *
            "close. This usually means psihigh is too close to the separatrix for the " *
            "equilibrium grid to resolve."
        )
    end
    if !isapprox(sol.t[end], 2π; atol=1e-8)
        error(
            "direct_fieldline_int: field-line integration at psifac = " *
            "$(@sprintf("%.6f", psifac)) stopped at eta = $(@sprintf("%.6f", sol.t[end])) " *
            "instead of 2*pi; the flux surface did not close."
        )
    end

    sol_matrix = reduce(hcat, sol.u::Vector{Vector{Float64}})'
    y_out = hcat(sol.t::Vector{Float64}, sol_matrix)
    # State is [∫dl/Bp, rfac, ∫dl/(R²Bp), ∫jac·dl/Bp] and the integration variable is η itself.
    return sample_trace_at_sfl_angles(sol, y_out, 4, theta_nodes, (u, eta, _) -> (eta, u[1], u[2], u[3], u[4])), bfield
end

"""
    direct_fieldline_der!(dy, y, params, eta)

Right-hand side of the geometric-angle field-line ODE, with state `y = [∫dl/Bp, rfac, ∫dl/(R²Bp), ∫jac·dl/Bp]` and
independent variable the geometric angle `eta`. Port of Fortran `direct_fl_der`, guarded against a vanishing denominator.
"""
function direct_fieldline_der!(dy, y, params::FieldLineDerivParams, eta)

    cos_eta, sin_eta = cos(eta), sin(eta)
    r = params.ro + y[2] * cos_eta
    z = params.zo + y[2] * sin_eta
    direct_get_bfield!(params.bfield, r, z, params.psi_in, params.sq_in, params.sq_in_deriv, params.psio; derivs=1)

    bp = sqrt(params.bfield.br^2 + params.bfield.bz^2)
    bt = params.bfield.f / r
    b = sqrt(bp^2 + bt^2)
    rfac = y[2]
    jac = (bp^params.power_bp) * (b^params.power_b) / (r^params.power_r * rfac^params.power_rc)

    # Denominator for d(l_pol)/d(eta) = rfac |B_pol|/denominator
    denominator = params.bfield.bz * cos_eta - params.bfield.br * sin_eta
    if abs(denominator) < 1e-14
        fill!(dy, 1e20) # Return large derivatives for solver to handle stiffness
        @warn "Denominator in direct_fieldline_der! near zero at eta=$eta."
        return
    end

    # d/dη [∫(dl/Bp)] = 1/|B_P| dl/d(eta) = rfac/denominator
    dy[1] = y[2] / denominator
    # d(rfac)/d(eta) = rfac/denom *( Br cos(eta) + Bz sin(eta) )
    dy[2] = dy[1] * (params.bfield.br * cos_eta + params.bfield.bz * sin_eta)
    # d/dη [∫(dl/(R²Bp))]
    dy[3] = dy[1] / (r^2)
    # d/dη [∫(jac*dl/Bp)]
    dy[4] = dy[1] * jac
end

"""
    direct_refine(rfac, eta, psi0, params) -> Float64

Newton-refines the distance `rfac` from the axis at geometric angle `eta` so the point lies on the surface ψ = `psi0`.
"""
function direct_refine(rfac::Float64, eta::Float64, psi0::Float64, params::FieldLineDerivParams)::Float64
    cos_eta, sin_eta = cos(eta), sin(eta)

    function f(rfac_inner)
        r = params.ro + rfac_inner * cos_eta
        z = params.zo + rfac_inner * sin_eta
        direct_get_bfield!(params.bfield, r, z, params.psi_in, params.sq_in,
            params.sq_in_deriv, params.psio; derivs=0)
        return params.bfield.psi - psi0
    end

    function fp(rfac_inner)
        r = params.ro + rfac_inner * cos_eta
        z = params.zo + rfac_inner * sin_eta
        direct_get_bfield!(params.bfield, r, z, params.psi_in, params.sq_in,
            params.sq_in_deriv, params.psio; derivs=1)
        return params.bfield.psir * cos_eta + params.bfield.psiz * sin_eta
    end

    return find_zero((f, fp), rfac, Roots.Newton();
        atol=1e-12 * abs(psi0), rtol=1e-12, maxevals=50)
end

"""
    equilibrium_solver(raw_profile, fieldline_int=direct_fieldline_int; override_psi_nodes=nothing) -> PlasmaEquilibrium

Direct equilibrium driver: locates the axis, traces each flux surface with `fieldline_int`, and builds the profile,
`rzphi` coordinate and `eqfun` splines in straight-fieldline coordinates.

## Arguments

  - `fieldline_int`: flux-surface tracer, called as `fieldline_int(psifac, raw_profile, ro, zo, rs2, theta_nodes) -> (y_out, bfield)`
    with `y_out` sampled at `theta_nodes` through `sample_trace_at_sfl_angles`
  - `override_psi_nodes`: explicit ψ grid used instead of the configured one
"""
@with_pool pool function equilibrium_solver(raw_profile::DirectRunInput, fieldline_int=direct_fieldline_int;
    override_psi_nodes::Union{Nothing,Vector{Float64}}=nothing)

    equil_params = raw_profile.config
    psio = raw_profile.psio
    mtheta = equil_params.mtheta
    psilow = equil_params.psilow
    psihigh = raw_profile.psihigh_resolved

    # Locate the magnetic axis and separatrix for the field-line integrations
    ro, zo, _, rs2 = direct_position!(raw_profile)

    psi_nodes = override_psi_nodes === nothing ?
                _build_psi_grid(equil_params, psilow, psihigh) :
                _validate_psi_nodes(override_psi_nodes, psilow, psihigh)
    mpsi = length(psi_nodes) - 1
    theta_nodes = range(0.0, 1.0; length=mtheta + 1)

    ff_x_nodes = collect(theta_nodes)
    sq_nodes = zeros!(pool, Float64, mpsi + 1, 4)
    rzphi_nodes = zeros!(pool, Float64, mpsi + 1, mtheta + 1, 4)
    ff_val = zeros!(pool, Float64, 4)
    ff_deriv_val = zeros!(pool, Float64, 4)

    for ipsi in (mpsi+1):-1:1  # outermost to innermost
        # Every tracer returns the surface sampled at theta_nodes, so all surfaces share SFL abscissae.
        y_out, bfield = fieldline_int(psi_nodes[ipsi], raw_profile, ro, zo, rs2, theta_nodes)
        checkpoint!(pool, Float64)

        ff_fs_nodes = acquire!(pool, Float64, mtheta + 1, 4)
        @. ff_fs_nodes[:, 1] = @view(y_out[:, 3])^2
        @. ff_fs_nodes[:, 2] = @view(y_out[:, 1]) / (2π) - ff_x_nodes
        @. ff_fs_nodes[:, 3] = bfield.f * (@view(y_out[:, 4]) - ff_x_nodes * y_out[end, 4])
        @. ff_fs_nodes[:, 4] = @view(y_out[:, 2]) / y_out[end, 2] - ff_x_nodes

        ff_fs_nodes[end, :] .= ff_fs_nodes[1, :]  # enforce periodic endpoint

        ff_interp = cubic_interp(ff_x_nodes, Series(ff_fs_nodes); bc=PeriodicBC())
        ff_deriv = deriv1(ff_interp)

        # Evaluate the θ-fit and its derivative at the nodes (the Jacobian needs the derivative)
        for itheta in 1:(mtheta+1)
            theta = theta_nodes[itheta]
            ff_interp(ff_val, theta)
            ff_deriv(ff_deriv_val, theta)

            rzphi_nodes[ipsi, itheta, 1] = ff_val[1]
            rzphi_nodes[ipsi, itheta, 2] = ff_val[2]
            rzphi_nodes[ipsi, itheta, 3] = ff_val[3]
            rzphi_nodes[ipsi, itheta, 4] = (1.0 + ff_deriv_val[4]) * y_out[end, 2] * 2π * psio
        end

        sq_nodes[ipsi, 1] = bfield.f * 2π
        sq_nodes[ipsi, 2] = bfield.p
        sq_nodes[ipsi, 3] = y_out[end, 2] * 2π * psio
        sq_nodes[ipsi, 4] = y_out[end, 4] * bfield.f / (2π)
        rewind!(pool, Float64)
    end

    # Temporary splines for q0 extrapolation and optional newq0 revision
    profiles = ProfileSplines(
        psi_nodes,
        sq_nodes[:, 1],  # F * 2π
        sq_nodes[:, 2],  # P * μ₀
        sq_nodes[:, 3],  # dV/dψ
        sq_nodes[:, 4]   # q
    )
    # q(0) by linear extrapolation from innermost surface
    q0 = profiles.q_spline.y[1] - profiles.q_deriv(psi_nodes[1]; hint=Ref(1)) * psi_nodes[1]
    if q0 <= 0.0
        @warn "q0 extrapolation to axis gives q0 = $(@sprintf("%.3f", q0)) ≤ 0 — likely a spline artifact from psilow being too large; check psilow or use newq0 to override."
    end
    # newq0 = -1 flips the extrapolated q0; resolved locally so the config keeps the user's request.
    newq0 = equil_params.newq0
    if newq0 == -1
        newq0 = -q0
    end
    if newq0 != 0.0
        @info "Revising q-profile for newq0 = $(@sprintf("%.3f", newq0))"
        f0 = profiles.F_spline.y[1] - profiles.F_deriv(psi_nodes[1]; hint=Ref(1)) * psi_nodes[1]
        f0fac = f0^2 * ((newq0 / q0)^2 - 1.0)
        for i in 1:(mpsi+1)
            ffac = sqrt(1.0 + f0fac / profiles.F_spline.y[i]^2) * sign(newq0)
            sq_nodes[i, 1] *= ffac
            sq_nodes[i, 4] *= ffac
            rzphi_nodes[i, :, 3] .*= ffac
        end
        profiles = ProfileSplines(
            psi_nodes,
            sq_nodes[:, 1],  # F * 2π
            sq_nodes[:, 2],  # P * μ₀
            sq_nodes[:, 3],  # dV/dψ
            sq_nodes[:, 4]   # q
        )
    end

    rzphi_xs = psi_nodes
    # rzphi_ys is a materialized Vector (not the Range) so PlasmaEquilibrium can index it directly
    rzphi_ys = collect(theta_nodes)

    grid2d = (rzphi_xs, theta_nodes)
    opts2d = (bc=(CubicFit(), PeriodicBC()), extrap=(ExtendExtrap(), WrapExtrap()))

    # Snap periodic endpoint: ff_interp evaluation at theta_nodes[end] may drift by machine eps from theta_nodes[1]
    @views rzphi_nodes[:, end, :] .= rzphi_nodes[:, 1, :]

    rzphi_rsquared = cubic_interp(grid2d, rzphi_nodes[:, :, 1]; opts2d...)
    rzphi_offset = cubic_interp(grid2d, rzphi_nodes[:, :, 2]; opts2d...)
    rzphi_nu = cubic_interp(grid2d, rzphi_nodes[:, :, 3]; opts2d...)
    rzphi_jac = cubic_interp(grid2d, rzphi_nodes[:, :, 4]; opts2d...)

    eqfun_B, eqfun_metric1, eqfun_metric2 = _eqfun_from_geometry(
        profiles, rzphi_rsquared, rzphi_offset, rzphi_nu, rzphi_jac, ro, psio, grid2d, opts2d)

    geometry = compute_geometry_profiles(rzphi_xs, rzphi_ys,
        rzphi_rsquared, rzphi_offset, rzphi_jac, ro)

    params = EquilibriumParameters()
    params.bt_sign = raw_profile.bt_sign
    params.ip_sign = raw_profile.ip_sign
    params.psihigh_resolved = psihigh

    return PlasmaEquilibrium(raw_profile.config, params, profiles, geometry,
        rzphi_xs, rzphi_ys,
        rzphi_rsquared, rzphi_offset, rzphi_nu, rzphi_jac,
        eqfun_B, eqfun_metric1, eqfun_metric2,
        ro, zo, psio)
end

# Reuse the processed geometry when restoring a frozen equilibrium snapshot.
function _eqfun_from_geometry(profiles, rzphi_rsquared, rzphi_offset, rzphi_nu, rzphi_jac, ro, psio, grid2d, opts2d)
    mpsi = length(grid2d[1]) - 1
    theta_nodes = grid2d[2]
    mtheta = length(theta_nodes) - 1
    eqfun_fs_nodes = zeros(Float64, mpsi + 1, mtheta + 1, 3)
    v = @MMatrix zeros(Float64, 2, 3)
    for ipsi in 1:(mpsi+1)
        q = profiles.q_spline.y[ipsi]
        f_val = profiles.F_spline.y[ipsi]
        for itheta in 1:(mtheta+1)
            theta_norm = theta_nodes[itheta]
            # partials indexing: [1,:,:] = f, [2,:,:] = ∂f/∂x, [3,:,:] = ∂f/∂y, [4,:,:] = ∂²f/∂x∂y
            f = (
                rzphi_rsquared.nodal_derivs.partials[1, ipsi, itheta],
                rzphi_offset.nodal_derivs.partials[1, ipsi, itheta],
                rzphi_nu.nodal_derivs.partials[1, ipsi, itheta],
                rzphi_jac.nodal_derivs.partials[1, ipsi, itheta]
            )
            fx = (
                rzphi_rsquared.nodal_derivs.partials[2, ipsi, itheta],
                rzphi_offset.nodal_derivs.partials[2, ipsi, itheta],
                rzphi_nu.nodal_derivs.partials[2, ipsi, itheta],
                rzphi_jac.nodal_derivs.partials[2, ipsi, itheta]
            )
            fy = (
                rzphi_rsquared.nodal_derivs.partials[3, ipsi, itheta],
                rzphi_offset.nodal_derivs.partials[3, ipsi, itheta],
                rzphi_nu.nodal_derivs.partials[3, ipsi, itheta],
                rzphi_jac.nodal_derivs.partials[3, ipsi, itheta]
            )
            rfac = sqrt(max(0.0, f[1]))  # guard against spline overshoot near separatrix
            eta = 2π * (theta_norm + f[2])
            r = ro + rfac * cos(eta)
            jacfac = f[4]

            v[1, 1] = (rfac > 0) ? fx[1] / (2.0 * rfac) : 0.0  # 1/(2rfac) * d(rfac)/d(psi_norm)
            v[1, 2] = fx[2] * 2π * rfac                          # 2π*rfac * d(eta)/d(psi_norm)
            v[1, 3] = fx[3] * r                                   # r * d(phi_s)/d(psi_norm)
            v[2, 1] = (rfac > 0) ? fy[1] / (2.0 * rfac) : 0.0  # 1/(2rfac) d(rfac)/d(theta_new)
            v[2, 2] = (1.0 + fy[2]) * 2π * rfac                  # 2π*rfac * d(eta)/d(theta_new)
            v[2, 3] = fy[3] * r                                   # r * d(phi_s)/d(theta_new)
            v33 = 2π * r
            w11 = (jacfac != 0) ? (1.0 + fy[2]) * (2π)^2 * rfac * r / jacfac : 0.0
            w12 = (jacfac * rfac != 0) ? -fy[1] * π * r / (rfac * jacfac) : 0.0
            delpsi_norm = sqrt(w11^2 + w12^2)
            modB = sqrt(((2π * psio * delpsi_norm)^2 + f_val^2) / (2π * r)^2)

            eqfun_fs_nodes[ipsi, itheta, 1] = modB
            denom = jacfac * modB^2
            if abs(denom) > 1e-20
                numerator_2 = dot(v[1, :], v[2, :]) + q * v33 * v[1, 3]  # gyrokinetic C1
                eqfun_fs_nodes[ipsi, itheta, 2] = numerator_2 / denom
                numerator_3 = v[2, 3] * v33 + q * v33^2                  # gyrokinetic C2
                eqfun_fs_nodes[ipsi, itheta, 3] = numerator_3 / denom
            else
                eqfun_fs_nodes[ipsi, itheta, 2] = 0.0
                eqfun_fs_nodes[ipsi, itheta, 3] = 0.0
            end
        end
    end

    @views eqfun_fs_nodes[:, end, :] .= eqfun_fs_nodes[:, 1, :]

    eqfun_B = cubic_interp(grid2d, eqfun_fs_nodes[:, :, 1]; opts2d...)
    eqfun_metric1 = cubic_interp(grid2d, eqfun_fs_nodes[:, :, 2]; opts2d...)
    eqfun_metric2 = cubic_interp(grid2d, eqfun_fs_nodes[:, :, 3]; opts2d...)

    return eqfun_B, eqfun_metric1, eqfun_metric2
end
