"""
    SolovevConfig(...)

A mutable struct holding parameters for the Solev'ev (SOL) plasma equilibrium model.

## Fields:

  - `mr`: number of radial grid zones
  - `mz`: number of axial grid zones
  - `ma`: number of flux grid zones
  - `elongation`: elongation
  - `a`: minor radius
  - `r0`: major radius
  - `q0`: safety factor at the o-point
  - `p0fac`: scale on-axis pressure (P-> P+P0*p0fac. beta changes. Phi,q constant)
  - `b0fac`: scale toroidal field at constant beta (s*Phi,s*f,s^2*P. bt changes. Shape,beta constant)
  - `f0fac`: scale toroidal field at constant pressure (s*f. beta,q changes. Phi,p,bp constant)
"""
@kwdef mutable struct SolovevConfig <: AnalyticEquilibriumConfig
    mr::Int = 128      # number of radial grid zones
    mz::Int = 128      # number of axial grid zones
    ma::Int = 128      # number of flux grid zones
    elongation::Float64 = 1.6 # elongation
    a::Float64 = 0.33      # minor radius
    r0::Float64 = 1.0      # major radius
    q0::Float64 = 1.9      # safety factor at the o-point
    p0fac::Float64 = 1       # scale on-axis pressure (P-> P+P0*p0fac. beta changes. Phi,q constant)
    b0fac::Float64 = 1       # scale toroidal field at constant beta (s*Phi,s*f,s^2*P. bt changes. Shape,beta constant)
    f0fac::Float64 = 1       # scale toroidal field at constant pressure (s*f. beta,q changes. Phi,p,bp constant)
end

"""
Build a `SolovevConfig` from a parsed `[SOL_INPUT]` TOML table.
"""
function SolovevConfig(input_dict::Dict{String,Any})
    return SolovevConfig(; symbolize_keys(input_dict)...)
end

"""
This function handles the Solovev analytical equilibrium model, transforming the input parameters
into the necessary splines and scalar values for equilibrium construction. This is a Julia version
of the Fortran code in sol.f, with no major differences except for arrays going from 0:n to 1:n+1.

## Arguments:

  - `mr`: Number of radial grid zones
  - `mz`: Number of axial grid zones
  - `ma`: Number of flux grid zones
  - `e`: Elongation
  - `a`: Minor radius
  - `r0`: Major radius
  - `q0`: Safety factor at the o-point
  - `p0fac`: Scales on axis pressure (s*P. beta changes. Phi,q constant)
  - `b0fac`: Scales on toroidal field (s*Phi,s*f,s^2*P. bt changes. Shape,beta constant)
  - `f0fac`: Scales on toroidal field (s*f. bt,q changes. Phi,p,bp constant)

## Returns:

  - `DirectRunInput` object
"""
function sol_run(equil_inputs::EquilibriumConfig, sol_inputs::SolovevConfig)

    mr = sol_inputs.mr
    mz = sol_inputs.mz
    ma = sol_inputs.ma
    e = sol_inputs.elongation
    a = sol_inputs.a
    r0 = sol_inputs.r0
    q0 = sol_inputs.q0
    p0fac = sol_inputs.p0fac
    b0fac = sol_inputs.b0fac
    f0fac = sol_inputs.f0fac

    # Validate inputs
    if p0fac < 1.0
        @warn "Forcing p0fac ≥ 1 (no negative pressure)"
        p0fac = 1.0
    end

    # Compute scalar data
    f0 = r0 * b0fac
    psio = e * f0 * a * a / (2 * q0 * r0)
    psifac = psio / (a * r0)^2
    efac = 1 / (e * e)
    pfac = 2 * psio^2 * (e * e + 1) / (a * r0 * e)^2
    rmin = r0 - 1.5 * a
    rmax = r0 + 1.5 * a
    zmax = 1.5 * e * a
    zmin = -zmax

    # zeros, not undef: only columns 1-3 are filled, so column 4 must stay a deterministic 0.
    sqfs = zeros(ma + 1, 4)
    psis = [(ia / (ma + 1))^2 for ia in 1:(ma+1)]
    sqfs[:, 1] .= f0 .* f0fac
    sqfs[:, 2] .= pfac .* (1 .* p0fac .- psis)
    sqfs[:, 3] .= 0.0
    sq_in = cubic_interp(psis, Series(sqfs); extrap=ExtendExtrap())

    # Compute 2D data and spline
    r = [rmin + i * (rmax - rmin) / mr for i in 0:mr]
    z = [zmin + j * (zmax - zmin) / mz for j in 0:mz]
    psifs = Array{Float64}(undef, mr + 1, mz + 1)
    for iz in 1:(mz+1)
        for ir in 1:(mr+1)
            psifs[ir, iz] = psio - psifac * (efac * (r[ir] * z[iz])^2 + (r[ir]^2 - r0^2)^2 / 4)
        end
    end
    psi_in_xs = r
    psi_in_ys = z
    psi_in = cubic_interp((psi_in_xs, psi_in_ys), psifs; extrap=ExtendExtrap())

    # Print out equilibrium info
    @info "Generating Solovev equilibrium: mr=$mr, mz=$mz, ma=$ma, e=$(@sprintf("%.3f", e)), a=$(@sprintf("%.3f", a)), r0=$(@sprintf("%.3f", r0)), q0=$(@sprintf("%.3f", q0))"

    # 1 is bt_sign=+1: Solovev has positive Bt by construction (f0 = r0 * b0fac > 0).
    # Analytic equilibria carry positive field and current; ingest=nothing — replay regenerates from the [SOL_INPUT] TOML section.
    return DirectRunInput(equil_inputs, sq_in, psi_in, psi_in_xs, psi_in_ys, rmin, rmax, zmin, zmax, psio, 1, 1, nothing)
end
