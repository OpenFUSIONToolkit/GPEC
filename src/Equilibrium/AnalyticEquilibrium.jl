
"""
    lar_init_conditions(rmin, sigma_type, params)

Initializes the starting radius and state vector for solving the LAR ODE system.
Also evaluates the initial derivative using the analytic model.

## Arguments:

  - `rmin`: Normalized starting radius (as a fraction of `lar_a`).
  - `lar_input`: A `LargeAspectRatioConfig` object containing equilibrium parameters.

## Returns:

  - `r`: Physical radius corresponding to `rmin * lar_a`.
  - `y`: Initial state vector of length 5.
"""
function lar_init_conditions(rmin::Float64, lar_input::LargeAspectRatioConfig)
    lar_a = lar_input.lar_a
    lar_r0 = lar_input.lar_r0
    q0 = lar_input.q0

    # Ensure rmin is not too small to avoid numerical issues in LAR equations
    # Use a physically meaningful minimum (0.001% of minor radius)
    if rmin < 1e-6
        @warn "Setting rmin to at least 1e-6 to avoid singularities in LAR equations."
        rmin_safe = 1e-6
    else
        rmin_safe = rmin
    end

    r = rmin_safe * lar_a
    y = zeros(5)

    y[1] = r^2 / (lar_r0 * q0)
    y[2] = 1.0
    y[3] = y[1] * lar_r0 / 2

    dy = zeros(5)
    lar_der(dy, r, y, lar_input)

    y[5] = dy[5] * r / 4

    q = r^2 * y[2] / (lar_r0 * y[1])
    y[4] = (q / r)^2 * y[5] / 2

    return r, y
end

"""
    lar_der(dy, r, y, lar_input)

Evaluates the spatial derivatives of the LAR (Large Aspect Ratio) equilibrium ODE system
at a given radius `r`, using the current state vector `y` and equilibrium parameters.

## Arguments:
- `dy`: A preallocated vector where the computed derivatives will be stored (in-place).
- `r`: The radial position at which the derivative is evaluated.
- `y`: The current state vector.
- `lar_input`: A `LargeAspectRatioConfig` object containing equilibrium shaping parameters.

## Returns:
- 0. The result is stored in-place in `dy`.
"""

function lar_der(dy::Vector{Float64}, r::Float64, y::Vector{Float64}, lar_input::LargeAspectRatioConfig)
    lar_a = lar_input.lar_a
    lar_r0 = lar_input.lar_r0

    q0 = lar_input.q0

    beta0 = lar_input.beta0
    p00 = beta0 / 2.0
    p_pres = lar_input.p_pres
    p_sig = lar_input.p_sig

    sigma_type = lar_input.sigma_type
    x = r / lar_a
    xfac = 1 - x^2
    pp = -2 * p_pres * p00 * x * xfac^(p_pres - 1)  # p'
    sigma0 = 2.0 / (q0 * lar_r0)

    sigma = if sigma_type == "wesson"
        sigma0 * xfac^p_sig
    else
        sigma0 / (1 + x^(2 * p_sig))^(1 + 1 / p_sig)
    end

    # LAR equations have singularities at r=0; ensure we never evaluate there
    @assert r > 0.0 "LAR equations require r > 0 (called with r=$r)"

    bsq = (y[1] / r)^2 + y[2]^2     # B²
    q = r^2 * y[2] / (lar_r0 * y[1])

    dy[1] = -pp / bsq * y[1] + sigma * y[2] * r
    dy[2] = -pp / bsq * y[2] - sigma * y[1] / r
    dy[3] = y[1] * lar_r0 / r
    dy[4] = (q / r)^2 * y[5] / r
    dy[5] = y[2] * (r / q)^2 * (r / lar_r0) * (1 - 2 * (lar_r0 * q / y[2])^2 * pp)

    return 0
end

"""
    lar_run(lar_input)

Solves the LAR (Large Aspect Ratio) plasma equilibrium ODE system using analytic profiles
defined by `lar_input`, and returns the full solution table including derived quantities.

## Arguments:
- `lar_input`: A `LargeAspectRatioConfig` object containing profile and geometric parameters.

## Returns:
 - working on it not yet implemented
"""

function lar_run(equil_input::EquilibriumConfig, lar_input::LargeAspectRatioConfig)
    rmin = 1e-4
    lar_a = lar_input.lar_a
    lar_r0 = lar_input.lar_r0
    q0 = lar_input.q0
    beta0 = lar_input.beta0
    sigma_type = lar_input.sigma_type

    p00 = beta0 / 2.0
    lar_input.p_pres = max(lar_input.p_pres, 1.001)

    sigma0 = 2.0 / (q0 * lar_r0)

    ma = lar_input.ma
    mtau = lar_input.mtau

    function dydr(du, u, p, r)
        lar_input = p
        return lar_der(du, r, u, lar_input)
    end

    r0, y0 = lar_init_conditions(rmin, lar_input)
    tspan = (r0, lar_a)
    p = lar_input

    prob = ODEProblem(dydr, y0, tspan, p)

    sol = solve(prob, Rosenbrock23(; autodiff=false); reltol=1e-6, abstol=1e-8, maxiters=10000, dense=false)

    r_arr = sol.t
    y_mat = reduce(hcat, sol.u)'
    steps = length(r_arr)

    temp = zeros(steps, 9)
    for i in 1:steps
        r = r_arr[i]
        @views y = y_mat[i, :]
        x = r / lar_a
        xfac = 1 - x^2
        pval = p00 * xfac^lar_input.p_pres
        sigma = (sigma_type == "wesson") ?
                sigma0 * xfac^lar_input.p_sig :
                sigma0 / (1 + x^(2 * lar_input.p_sig))^(1 + 1 / lar_input.p_sig)
        q = r^2 * y[2] / (lar_r0 * y[1])
        temp[i, 1] = r
        temp[i, 2] = y[1]
        temp[i, 3] = y[2]
        temp[i, 4] = y[3]
        temp[i, 5] = y[4]
        temp[i, 6] = y[5]
        temp[i, 7] = pval
        temp[i, 8] = sigma
        temp[i, 9] = q
    end

    xs_r = temp[:, 1]
    fs_r = temp[:, 2:9]
    spl = cubic_interp(xs_r, fs_r; bc=CubicFit(), search=LinearBinary(), extrap=:extension)
    spl_deriv = deriv1(spl)

    dr = lar_a / (ma + 1)
    r = 0.0
    psio = temp[end, 4]  # ψ_edge

    sq_xs = zeros(ma + 1)
    sq_fs = zeros(ma + 1, 3)
    r_nodes = zeros(ma + 1)
    rzphi_y_nodes = range(0.0, 2π; length=mtau + 1)
    rzphi_fs_nodes = zeros(ma + 1, mtau + 1, 2)

    hint = Ref(1)
    for ia in 1:(ma+1)
        r += dr
        r_nodes[ia] = r
        f = spl(r; hint=hint)
        f1 = spl_deriv(r; hint=hint)
        dψdr = f1[3]

        # Fill spline data for sq_in
        sq_xs[ia] = f[3] / psio  # ψ / psio
        sq_fs[ia, 1] = lar_r0 * f[2]  # F = R0 * Bphi
        sq_fs[ia, 2] = f[6]  # P
        sq_fs[ia, 3] = f[8]  # q

        # Compute Shafranov shift and fill rzphi grid
        r2 = -(f[4] * r / f[8]) / dψdr
        if lar_input.zeroth
            r2 = 0.0
        end
        for itau in 1:(mtau+1)
            θ = 2π * (itau - 1) / mtau
            cosθ, sinθ = cos(θ), sin(θ)
            rfac = r + r2 * cosθ
            rzphi_fs_nodes[ia, itau, 1] = lar_r0 + rfac * cosθ
            rzphi_fs_nodes[ia, itau, 2] = rfac * sinθ
        end
    end

    sq_in = cubic_interp(sq_xs, sq_fs; bc=CubicFit(), search=LinearBinary(), extrap=:extension)
    # Create separate interpolants for R and Z coordinates
    rz_in_xs = r_nodes
    rz_in_ys = collect(rzphi_y_nodes)
    rz_in_R = cubic_interp((rz_in_xs, rz_in_ys), rzphi_fs_nodes[:, :, 1]; search=LinearBinary(),
        bc=(CubicFit(), PeriodicBC()), extrap=(:extension, :wrap))
    rz_in_Z = cubic_interp((rz_in_xs, rz_in_ys), rzphi_fs_nodes[:, :, 2]; search=LinearBinary(),
        bc=(CubicFit(), PeriodicBC()), extrap=(:extension, :wrap))

    # LAR equilibrium has midplane symmetry, so magnetic axis is at Z = 0
    return InverseRunInput(equil_input, sq_in, rz_in_xs, rz_in_ys, rz_in_R, rz_in_Z, lar_r0, 0.0, psio)
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
    e = sol_inputs.e
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

    # Compute 1D data and spline
    sqfs = Array{Float64}(undef, ma + 1, 4)
    psis = [(ia / (ma + 1))^2 for ia in 1:(ma+1)]
    sqfs[:, 1] .= f0 .* f0fac
    sqfs[:, 2] .= pfac .* (1 .* p0fac .- psis)
    sqfs[:, 3] .= 0.0
    sq_in = cubic_interp(psis, sqfs; bc=CubicFit(), extrap=:extension)

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
    psi_in = cubic_interp((psi_in_xs, psi_in_ys), psifs; search=LinearBinary(),
        bc=CubicFit(), extrap=:extension)

    # Print out equilibrium info
    @info "Generating Solovev equilibrium: mr=$mr, mz=$mz, ma=$ma, e=$(@sprintf("%.3f", e)), a=$(@sprintf("%.3f", a)), r0=$(@sprintf("%.3f", r0)), q0=$(@sprintf("%.3f", q0))"

    return DirectRunInput(equil_inputs, sq_in, psi_in, psi_in_xs, psi_in_ys, rmin, rmax, zmin, zmax, psio)
end
