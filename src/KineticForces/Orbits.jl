"""
Orbits.jl - guiding-center orbits in the constants of motion (E, μ, P_φ).

Integrates Littlejohn's guiding-center equations with H = ½mv∥² + μB + ZeΦ(ψ) in the axisymmetric
equilibrium, in (ψ_N, θ, φ) with θ the straight-field-line angle and φ the geometric toroidal angle.
The curvature, its ∇p part and the parallel current enter through ∇×b computed from the covariant
b_i, so there is no low-β drift formula. P_φ = m v∥ F/B − Z e s psio ψ_N is conserved exactly by the
equations and monitored numerically.
"""

"""
    GuidingCenterField

Equilibrium field tabulated for guiding-center orbits on the `rzphi` grid of (ψ_N, θ).

## Fields

  - `B`: |B| (`equil.eqfun_B`)
  - `bcov_psi`, `bcov_theta`: covariant b_ψ = g_ψθ B^θ/B and b_θ = g_θθ B^θ/B (m per unit ψ_N, θ)
  - `jac`: signed Jacobian 𝒥 = R(R_θ Z_ψ − R_ψ Z_θ) of (ψ_N, θ, φ) in m³
  - `R`, `Z`: cylindrical coordinates in m
  - `nu`: toroidal offset ν in φ = 2πζ + ν(ψ_N, θ), ζ the straight-field-line toroidal angle in turns
  - `F`, `dFdpsi`: R B_φ in T·m and its ψ_N derivative
  - `phi`: electrostatic potential Φ(ψ_N) in V, or `nothing` for no potential
  - `s`: sign in B^θ = s psio/𝒥, chosen so that q > 0
  - `psio`, `bo`, `ro`: flux per radian, axis field, axis major radius
  - `psi_min`, `psi_max`: the radial grid's first ψ and the loss boundary; orbits reaching them are classified axis or lost
"""
struct GuidingCenterField{I,S,D,P}
    B::I
    bcov_psi::I
    bcov_theta::I
    jac::I
    R::I
    Z::I
    nu::I
    F::S
    dFdpsi::D
    phi::P
    s::Float64
    psio::Float64
    bo::Float64
    ro::Float64
    psi_min::Float64
    psi_max::Float64
end

"""
    guiding_center_field(equil; omegaE=nothing, psi_max=Inf) → GuidingCenterField

Tabulate the guiding-center field of `equil`. `omegaE(ψ_N)` (rad/s, the input E×B frequency) builds
the potential from dΦ/dψ_N = −psio ω_E; `nothing` leaves Φ out of the orbit. Orbits are lost at
`psi_max` (the control surface `psilim` for `forb`, where the perturbation ends) or at the grid's last ψ if lower.
"""
function guiding_center_field(equil; omegaE=nothing, psi_max::Float64=Inf)
    xs, ys = equil.rzphi_xs, equil.rzphi_ys
    nx, ny = length(xs), length(ys)
    F(ψ; hint=nothing) = equil.profiles.F_spline(ψ; hint) / twopi
    dF(ψ; hint=nothing) = equil.profiles.F_deriv(ψ; hint) / twopi
    jac, R, Z, nu, gpt, gtt = (zeros(nx, ny) for _ in 1:6)
    for (j, θ) in enumerate(ys), (i, ψ) in enumerate(xs)
        pt = (ψ, θ)
        rsq = equil.rzphi_rsquared(pt)
        rsq_x = equil.rzphi_rsquared(pt; deriv=DerivOp(1, 0))
        rsq_y = equil.rzphi_rsquared(pt; deriv=DerivOp(0, 1))
        off = equil.rzphi_offset(pt)
        off_x = equil.rzphi_offset(pt; deriv=DerivOp(1, 0))
        off_y = equil.rzphi_offset(pt; deriv=DerivOp(0, 1))
        r = sqrt(rsq)
        α = twopi * (θ + off)
        cα, sα = cos(α), sin(α)
        r_x, r_y = rsq_x / (2r), rsq_y / (2r)
        a_x, a_y = twopi * off_x, twopi * (1 + off_y)
        R_x, Z_x = r_x * cα - r * sα * a_x, r_x * sα + r * cα * a_x
        R_y, Z_y = r_y * cα - r * sα * a_y, r_y * sα + r * cα * a_y
        R[i, j] = equil.ro + r * cα
        nu[i, j] = equil.rzphi_nu(pt)
        Z[i, j] = equil.zo + r * sα
        jac[i, j] = R[i, j] * (R_y * Z_x - R_x * Z_y)
        gpt[i, j] = R_x * R_y + Z_x * Z_y
        gtt[i, j] = R_y^2 + Z_y^2
    end
    # The equilibrium's own Jacobian, as the surface path uses; the one rebuilt from R, Z derivatives is rough near the edge.
    jeq = [equil.rzphi_jac((ψ, θ)) / twopi for ψ in xs, θ in ys]
    ratio = sum(jac[2:end, :] ./ jeq[2:end, :]) / length(jeq[2:end, :])
    abs(abs(ratio) - 1) < 1e-3 || error("guiding_center_field: rzphi_jac/2π and the R, Z Jacobian differ by $(ratio)")
    jac .= sign(ratio) .* jeq
    # q = (1/2π)∮ B^φ/B^θ dθ > 0 fixes the sign of the poloidal field relative to F.
    imid = nx ÷ 2
    s = sign(F(xs[imid]) * sum(jac[imid, 1:end-1] ./ R[imid, 1:end-1] .^ 2) / equil.psio)
    Bt = [equil.eqfun_B((ψ, θ)) for ψ in xs, θ in ys]
    Btheta = s * equil.psio ./ jac
    bpsi = gpt .* Btheta ./ Bt
    btheta = gtt .* Btheta ./ Bt
    for A in (jac, R, Z, nu, bpsi, btheta, Bt)
        A[:, end] .= A[:, 1]
    end
    opts = (bc=(CubicFit(), PeriodicBC()), extrap=(ExtendExtrap(), WrapExtrap()))
    itp(A) = cubic_interp((xs, ys), A; opts...)
    phi = nothing
    if omegaE !== nothing
        dphi = [-equil.psio * omegaE(ψ) for ψ in xs]
        phi = cubic_interp(xs, FastInterpolations.cumulative_integrate(cubic_interp(xs, dphi)); extrap=ExtendExtrap())
    end
    return GuidingCenterField(itp(Bt), itp(bpsi), itp(btheta), itp(jac), itp(R), itp(Z), itp(nu),
        F, dF, phi, s, equil.psio, abs(F(0.0)) / equil.ro, equil.ro, first(xs), min(psi_max, last(xs)))
end

_phi(g::GuidingCenterField, ψ; hint=nothing) = g.phi === nothing ? 0.0 : g.phi(ψ; hint)
_dphi(g::GuidingCenterField, ψ; hint=nothing) = g.phi === nothing ? 0.0 : g.phi(ψ; deriv=DerivOp(1), hint)

"""
Parameters of one orbit integration: species, μ, reference speed, the event bookkeeping, F(ψ₀), and spline cell hints for the
orbit point (`h`, `h1`) and for the start surface (`h0`).
"""
mutable struct _OrbitParams{G}
    g::G
    mass::Float64
    chrg::Float64
    mu::Float64
    v::Float64
    theta0::Float64
    psi0::Float64
    dir0::Float64
    ncross::Int
    winding::Int
    status::Symbol
    F0::Float64
    h::NTuple{2,Base.RefValue{Int}}
    h0::NTuple{2,Base.RefValue{Int}}
    h1::Base.RefValue{Int}
end

# Field-line pitch B^φ/B^θ on the start surface ψ₀ at θ.
_pitch0(g, p, θ) = (pt = (p.psi0, θ); p.F0 * g.jac(pt; hint=p.h0) / (g.s * g.psio * g.R(pt; hint=p.h0)^2))

"""
Right-hand side in normalized time τ = t v/R₀; state (ψ, θ, φ, v∥/v, ∫ψ dτ, χ), with χ = φ − ∫ (B^φ/B^θ)(ψ₀, θ) dθ the
toroidal angle relative to the field line of the start surface. Its rate is formed with the parallel streaming already
cancelled, so the precession of a trapped orbit is not the small remainder of two large toroidal legs.
"""
function _gc_rhs!(du, u, p::_OrbitParams, t)
    g = p.g
    ψ, θ, w = u[1], u[2], u[4]
    pt, h, h1 = (ψ, θ), p.h, p.h1
    B = g.B(pt; hint=h)
    B_x, B_y = g.B(pt; deriv=DerivOp(1, 0), hint=h), g.B(pt; deriv=DerivOp(0, 1), hint=h)
    J = g.jac(pt; hint=h)
    bψ, bθ = g.bcov_psi(pt; hint=h), g.bcov_theta(pt; hint=h)
    bθ_x = g.bcov_theta(pt; deriv=DerivOp(1, 0), hint=h)
    bψ_y = g.bcov_psi(pt; deriv=DerivOp(0, 1), hint=h)
    R = g.R(pt; hint=h)
    F, F_x = g.F(ψ; hint=h1), g.dFdpsi(ψ; hint=h1)
    bφ = F / B
    bφ_x, bφ_y = F_x / B - F * B_x / B^2, -F * B_y / B^2
    # Contravariant ∇×b and B
    cψ, cθ, cφ = bφ_y / J, -bφ_x / J, (bθ_x - bψ_y) / J
    Bθ, Bφ = g.s * g.psio / J, F / R^2
    vpar = p.v * w
    k = p.mass * vpar / p.chrg
    Bsψ, Bsθ, Bsφ = k * cψ, Bθ + k * cθ, Bφ + k * cφ
    Bspar = B + k * (bψ * cψ + bθ * cθ + bφ * cφ)
    Hx = p.mu * B_x + p.chrg * _dphi(g, ψ; hint=h1)
    Hy = p.mu * B_y
    sc = g.ro / p.v
    du[1] = sc * (vpar * Bsψ - bφ * Hy / (J * p.chrg)) / Bspar
    du[2] = sc * (vpar * Bsθ + bφ * Hx / (J * p.chrg)) / Bspar
    du[3] = sc * (vpar * Bsφ + (bψ * Hy - bθ * Hx) / (J * p.chrg)) / Bspar
    du[4] = -sc * (Bsψ * Hx + Bsθ * Hy) / (p.mass * Bspar * p.v)
    du[5] = ψ
    f0 = _pitch0(g, p, θ)
    du[6] = sc * (vpar * (Bφ - Bθ * f0 + k * (cφ - cθ * f0)) + (bψ * Hy - (bθ + f0 * bφ) * Hx) / (J * p.chrg)) / Bspar
    return nothing
end

"""
    GuidingCenterOrbit

One closed (or lost) guiding-center orbit.

## Fields

  - `status`: `:trapped`, `:passing`, `:lost` (reached `psi_max`), `:axis` (reached `psi_min`) or `:unclosed`
  - `tau`: period (full bounce for trapped, one poloidal transit for passing) in s
  - `wb`: 2π/τ in rad/s
  - `wphi`: secular toroidal rate Δφ/τ in rad/s (parallel advance, precession and E×B), from Δχ plus k turns of the
    start surface's field line, k the net winding
  - `wdrift`: toroidal rate relative to the field line of the start surface, Δχ/τ in rad/s
  - `psibar`: time-averaged ψ_N over the period
  - `psi_range`: (min, max) of ψ_N along the orbit
  - `dir_theta`: poloidal direction: the net winding (±1) of a passing orbit, the sign of dθ/dt at the start otherwise
  - `ncross`: crossings of the start section θ ≡ θ₀ (mod 1) per period (2 for a banana, 1 for a passing orbit)
  - `energy_error`, `pphi_error`: relative drift of H and P_φ over the integration
  - `sol`: the ODE solution (dense), state (ψ, θ, φ, v∥/v, ∫ψ dτ, χ) in τ = t v/R₀
"""
struct GuidingCenterOrbit{S}
    status::Symbol
    tau::Float64
    wb::Float64
    wphi::Float64
    wdrift::Float64
    psibar::Float64
    psi_range::Tuple{Float64,Float64}
    dir_theta::Float64
    ncross::Int
    energy_error::Float64
    pphi_error::Float64
    sol::S
end

_energy(g, p, u) = 0.5 * p.mass * (p.v * u[4])^2 + p.mu * g.B((u[1], u[2])) + p.chrg * _phi(g, u[1])
_pphi(g, p, u) = p.mass * p.v * u[4] * g.F(u[1]) / g.B((u[1], u[2])) - p.chrg * g.s * g.psio * u[1]

# E, μ, P_φ fix v∥ at each (ψ, θ), so an orbit closes on its first return to θ ≡ θ₀ (mod 1) at ψ₀ in the starting
# direction; its net winding is the class (0 trapped, ±1 passing), whether or not v∥ reverses (E×B-modified orbits).
const ORBIT_CLOSE_PSI_TOL = 1e-6

function _orbit_condition!(out, u, t, integ)
    p = integ.p
    out[1] = sinpi(u[2] - p.theta0)
    out[2] = u[1] - p.g.psi_max
    out[3] = u[1] - p.g.psi_min
    return nothing
end

function _orbit_event!(integ, idx, up::Bool)
    p = integ.p
    if idx == 1 && integ.t > 1e-6
        k = round(Int, integ.u[2] - p.theta0)
        # d/dt sinpi(θ − θ₀) = π(−1)^k θ̇ at the section, so the crossing direction follows from `up` and k.
        dir = (up ? 1.0 : -1.0) * (isodd(k) ? -1.0 : 1.0)
        p.ncross += 1
        if dir == p.dir0 && abs(integ.u[1] - p.psi0) < ORBIT_CLOSE_PSI_TOL
            p.winding = k
            p.status = k == 0 ? :trapped : abs(k) == 1 ? :passing : :unclosed
            terminate!(integ)
        end
    elseif idx == 2 && up
        p.status = :lost
        terminate!(integ)
    elseif idx == 3 && !up
        p.status = :axis
        terminate!(integ)
    end
    return nothing
end

"""
    guiding_center_orbit(g, psi0, lambda, energy, sigma; mass, chrg, theta0=0.0, rtol=1e-10, tau_max=2e4, dense=true)
        → GuidingCenterOrbit

Integrate the orbit through (`psi0`, `theta0`) with kinetic energy `energy` (J) there, pitch
`lambda` = μB₀/E and sign `sigma` of v∥, until it closes, is lost, reaches the axis grid edge, or
`tau_max` (in R₀/v) passes. Returns `nothing` if the start point is forbidden (λB > B₀).
With `dense=false` only the end state is saved: the status, period and frequencies are kept, but the ψ range
holds the end points only and the solution cannot be sampled along the orbit.
"""
function guiding_center_orbit(g::GuidingCenterField, psi0::Real, lambda::Real, energy::Real, sigma::Real;
    mass::Real, chrg::Real, theta0::Real=0.0, rtol::Real=1e-10, tau_max::Real=2e4, dense::Bool=true)
    B0 = g.B((psi0, theta0))
    w2 = 1 - lambda * B0 / g.bo
    w2 < 0 && return nothing
    v = sqrt(2 * energy / mass)
    mu = lambda * energy / g.bo
    u0 = [Float64(psi0), Float64(theta0), 0.0, sign(sigma) * sqrt(w2), 0.0, 0.0]
    p = _OrbitParams(g, Float64(mass), Float64(chrg), mu, v, Float64(theta0), Float64(psi0), 0.0, 0, 0, :unclosed, g.F(psi0),
        (Ref(1), Ref(1)), (Ref(1), Ref(1)), Ref(1))
    du = similar(u0)
    _gc_rhs!(du, u0, p, 0.0)
    p.dir0 = sign(du[2])
    H0, P0 = _energy(g, p, u0), _pphi(g, p, u0)
    # Events are found from sign changes at step ends: interior checks would run Vern9's extra interpolation stages every step.
    cb = VectorContinuousCallback(_orbit_condition!, (i, idx) -> _orbit_event!(i, idx, true),
        (i, idx) -> _orbit_event!(i, idx, false), 3; save_positions=(false, false), interp_points=0)
    prob = ODEProblem(_gc_rhs!, u0, (0.0, Float64(tau_max)), p)
    sol = solve(prob, Vern9(); reltol=rtol, abstol=rtol * 1e-2, callback=cb, dense, save_everystep=dense)
    uend, τn = sol.u[end], sol.t[end]
    tau = τn * g.ro / v
    psis = [u[1] for u in sol.u]
    Pscale = abs(chrg * g.psio)
    turn = p.winding == 0 ? 0.0 : p.winding * sum(θ -> _pitch0(g, p, θ), range(0, 1; length=513)[1:end-1]) / 512
    return GuidingCenterOrbit(p.status, tau, twopi / tau, (uend[6] + turn) / tau, uend[6] / tau, uend[5] / τn, (minimum(psis), maximum(psis)),
        p.status == :passing ? Float64(p.winding) : p.dir0, p.ncross, abs(_energy(g, p, uend) / H0 - 1), abs(_pphi(g, p, uend) - P0) / Pscale, sol)
end

const ORBIT_STATUS_CODES = Dict(:unclosed => Int8(0), :trapped => Int8(1), :passing => Int8(2), :lost => Int8(3), :axis => Int8(4))

"""
    write_orbits!(h5file, ctrl, kinetic_profiles, equil; psi=0.1:0.1:0.9,
                  pitch=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99], x=[0.1, 1.0, 10.0])

Write `KineticForces/Orbits`: main-ion guiding-center orbits started at the outboard midplane (θ = 0) of each
`psi`, with λ = `pitch`·B₀/B(ψ, 0) (the fraction of the largest λ allowed there), kinetic energy `x`·T_i(ψ) and
both signs σ of v∥. The potential is in the orbit when `ctrl.orbit_potential` is set.
"""
function write_orbits!(h5file::Union{HDF5.File,HDF5.Group}, ctrl::KineticForcesControl, kinetic_profiles, equil;
    psi=collect(0.1:0.1:0.9), pitch=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99], x=[0.1, 1.0, 10.0])
    kinetic_profiles === nothing && return nothing
    root = ensure_group(h5file, "KineticForces")
    haskey(root, "Orbits") && return nothing
    zi, mi = main_ion(equil)
    g = guiding_center_field(equil; omegaE=ctrl.orbit_potential ? kinetic_profiles.omegaE_spline : nothing)
    sigma = [1, -1]
    dims = (length(psi), length(pitch), length(x), length(sigma))
    lambda = zeros(dims[1], dims[2])
    status = zeros(Int8, dims)
    wb, wphi, psibar, psimin, psimax = (fill(NaN, dims) for _ in 1:5)
    for (i, ψ) in enumerate(psi), (j, p) in enumerate(pitch)
        lambda[i, j] = p * g.bo / g.B((ψ, 0.0))
        for (k, xk) in enumerate(x), (l, sv) in enumerate(sigma)
            o = guiding_center_orbit(g, ψ, lambda[i, j], xk * kinetic_profiles.Ti_spline(ψ), sv; mass=mi * mp, chrg=zi * e)
            o === nothing && continue
            status[i, j, k, l] = ORBIT_STATUS_CODES[o.status]
            o.status in (:trapped, :passing) || continue
            wb[i, j, k, l], wphi[i, j, k, l], psibar[i, j, k, l] = o.wb, o.wphi, o.psibar
            psimin[i, j, k, l], psimax[i, j, k, l] = o.psi_range
        end
    end
    og = create_group(root, "Orbits")
    og["psi"], og["pitch"], og["x"], og["sigma"] = collect(Float64, psi), collect(Float64, pitch), collect(Float64, x), sigma
    og["lambda"], og["status"], og["wb"], og["wphi"] = lambda, status, wb, wphi
    og["psibar"], og["psi_min"], og["psi_max"] = psibar, psimin, psimax
    Utilities.HDF5Annotations.annotate!(og, KF_ORBITS_H5_ANNOTATIONS)
    return nothing
end

const _ORBIT_DIMS = ("psi", "pitch", "x", "sigma")
const _ORBIT_ATTACH = (1 => "psi", 2 => "pitch", 3 => "x", 4 => "sigma")
const KF_ORBITS_H5_ANNOTATIONS = [
    "psi" => (; long_name="normalized poloidal flux ψ_N of the orbit start at the outboard midplane", scale="psi"),
    "pitch" => (; long_name="λ as a fraction of its largest allowed value B₀/B(ψ, θ=0) at the start", scale="pitch"),
    "x" => (; long_name="kinetic energy at the start in units of the local ion temperature", scale="x"),
    "sigma" => (; long_name="sign of v∥ at the start", scale="sigma"),
    "lambda" => (; long_name="pitch λ = μB₀/E", units="1", dims=("psi", "pitch"), attach=(1 => "psi", 2 => "pitch")),
    "status" => (; long_name="orbit class: 0 unclosed, 1 trapped, 2 passing, 3 lost, 4 reached the axis grid edge", units="1", dims=_ORBIT_DIMS,
        attach=_ORBIT_ATTACH),
    "wb" => (; long_name="bounce or transit frequency 2π/τ (trapped and passing orbits)", units="rad/s", dims=_ORBIT_DIMS, attach=_ORBIT_ATTACH),
    "wphi" => (; long_name="secular toroidal rate Δφ/τ in the geometric toroidal angle", units="rad/s", dims=_ORBIT_DIMS, attach=_ORBIT_ATTACH),
    "psibar" => (; long_name="time-averaged ψ_N over one orbit period", units="1", dims=_ORBIT_DIMS, attach=_ORBIT_ATTACH),
    "psi_min" => (; long_name="smallest ψ_N along the orbit", units="1", dims=_ORBIT_DIMS, attach=_ORBIT_ATTACH),
    "psi_max" => (; long_name="largest ψ_N along the orbit", units="1", dims=_ORBIT_DIMS, attach=_ORBIT_ATTACH)
]
