# ======================================================================
#   Main Driver for Ballooning Stability Analysis
#   Computes ballooning stability criterion over all flux surfaces
# ======================================================================
using LinearAlgebra

"""
    compute_ballooning_stability!(ctrl, locstab_fs, plasma_eq)

Main driver routine for local high-n stability analysis. Iterates over all
magnetic flux surfaces, prepares ballooning coefficients, stores `det(d0bar)`
as the local `Di` diagnostic, and optionally integrates the ballooning equation
to compute Delta Prime.

## Arguments

  - `ctrl::ForceFreeStatesControl`: Control parameters for the analysis.
  - `locstab_fs::Matrix{Float64}`: Local stability matrix to store results (modified in place).
  - `plasma_eq::Equilibrium.PlasmaEquilibrium`: Plasma equilibrium data.

This function modifies `locstab_fs` in place with:

  - Column 1: `det(d0bar) * ψ` (Mercier interchange `D_I`)
  - Column 2: resistive interchange `D_R * ψ` (see [`resistive_interchange`](@ref))
  - Column 4: Delta Prime (Δ')
"""
function compute_ballooning_stability!(
    ctrl::ForceFreeStatesControl,
    locstab_fs::Matrix{Float64},
    plasma_eq::Equilibrium.PlasmaEquilibrium;
    theta_k::Float64=0.0,
    compute_delta_prime::Bool=true
)

    if ctrl.verbose
        println("Evaluating local high-n ballooning stability...")
    end

    num_psi = length(plasma_eq.profiles.xs)

    # Loop over flux surfaces
    for flux_surface_index in 1:num_psi

        psi = plasma_eq.profiles.xs[flux_surface_index]
        coeff_data = prepare_ballooning_coefficients(flux_surface_index, plasma_eq; theta_k=theta_k)
        locstab_fs[flux_surface_index, 1] = coeff_data.di * psi
        locstab_fs[flux_surface_index, 2] = resistive_interchange(flux_surface_index, plasma_eq).dr * psi

        if compute_delta_prime && plasma_eq.profiles.xs[flux_surface_index] <= 1.0
            result = integrate_ballooning_ode(
                coeff_data.ode_coefficient_spline;
                theta_k=theta_k
            )
            locstab_fs[flux_surface_index, 4] = result.value
        end
    end

    if ctrl.verbose
        println("Ballooning analysis complete.")
    end

end

"""
    resistive_interchange(flux_surface_index, plasma_eq)

Resistive interchange criterion `D_R = D_I + (H - 1/2)²` at a single flux surface
[Glasser-Greene-Johnson; Glasser Phys. Plasmas 23, 112506 (2016)]. The Mercier
`D_I` and the intermediate `H` are formed from flux-surface averages of the field
and metric quantities. Returns a NamedTuple `(di, dr, h)`, all unscaled by ψ.

Note: this evaluates the Mercier `D_I` from the surface-average formulation, which
is a distinct numerical route from the `det(d0bar)` value reported as `locstab/di`
by [`compute_ballooning_stability!`](@ref).
"""
function resistive_interchange(flux_surface_index::Int, plasma_eq::Equilibrium.PlasmaEquilibrium)
    profiles = plasma_eq.profiles
    ntheta = length(plasma_eq.rzphi_ys)
    ff_fs = zeros(ntheta, 5)

    psi = profiles.xs[flux_surface_index]
    twopif = profiles.F_spline.y[flux_surface_index]
    p1 = profiles.P_deriv(psi)
    v1 = profiles.dVdpsi_spline.y[flux_surface_index]
    v2 = profiles.dVdpsi_deriv(psi)
    q = profiles.q_spline.y[flux_surface_index]
    q1 = profiles.q_deriv(psi)
    chi1 = 2π * plasma_eq.psio

    for itheta in 1:ntheta
        theta = plasma_eq.rzphi_ys[itheta]

        f1 = plasma_eq.rzphi_rsquared.nodal_derivs.partials[1, flux_surface_index, itheta]
        f2 = plasma_eq.rzphi_offset.nodal_derivs.partials[1, flux_surface_index, itheta]
        jac = plasma_eq.rzphi_jac.nodal_derivs.partials[1, flux_surface_index, itheta]
        fy1 = plasma_eq.rzphi_rsquared.nodal_derivs.partials[3, flux_surface_index, itheta]
        fy2 = plasma_eq.rzphi_offset.nodal_derivs.partials[3, flux_surface_index, itheta]
        fy3 = plasma_eq.rzphi_nu.nodal_derivs.partials[3, flux_surface_index, itheta]

        rfac = sqrt(f1)
        eta = 2π * (theta + f2)
        r = plasma_eq.ro + rfac * cos(eta)

        v21 = fy1 / (2.0 * rfac * jac)
        v22 = (1.0 + fy2) * 2π * rfac / jac
        v23 = fy3 * r / jac
        v33 = 2π * r / jac
        bsq = chi1^2 * (v21^2 + v22^2 + (v23 + q * v33)^2)
        dpsisq = (2π * r)^2 * (v21^2 + v22^2)

        ff_fs[itheta, 1] = bsq / dpsisq
        ff_fs[itheta, 2] = 1.0 / dpsisq
        ff_fs[itheta, 3] = 1.0 / bsq
        ff_fs[itheta, 4] = 1.0 / (bsq * dpsisq)
        ff_fs[itheta, 5] = bsq
        @views ff_fs[itheta, :] .*= jac / v1
    end

    avg = FastInterpolations.integrate(cubic_interp(plasma_eq.rzphi_ys, Series(ff_fs); bc=PeriodicBC()))

    term = twopif * p1 * v1 / (q1 * chi1^3) * avg[2]
    di = -0.25 + term * (1 - term) +
         p1 * (v1 / (q1 * chi1^2))^2 * avg[1] *
         (p1 * (avg[3] + (twopif / chi1)^2 * avg[4]) - v2 / v1)
    h = twopif * p1 * v1 / (q1 * chi1^3) * (avg[2] - avg[1] / avg[5])
    return (di=di, dr=di + (h - 0.5)^2, h=h)
end





function _collect_surface_response_background(
    flux_surface_index::Int,
    plasma_eq::Equilibrium.PlasmaEquilibrium,
    theta_grid::AbstractVector{<:Real}
)::NamedTuple
    profiles = plasma_eq.profiles

    theta_vals = Float64.(theta_grid)
    ntheta = length(theta_vals)

    psi0 = profiles.xs[flux_surface_index]
    q0 = Float64(profiles.q_spline.y[flux_surface_index])
    q0prime = Float64(profiles.q_deriv(psi0))
    p0prime = Float64(profiles.P_deriv(psi0))
    two_pi_f0 = Float64(profiles.F_spline.y[flux_surface_index])
    chi_prime0 = Float64(2pi * plasma_eq.psio)

    f0 = zeros(ntheta, 4)
    fx0 = zeros(ntheta, 4)
    fy0 = zeros(ntheta, 4)
    fxy0 = zeros(ntheta, 4)
    rho0 = zeros(ntheta)
    eta0 = zeros(ntheta)
    r0 = zeros(ntheta)
    gradpsi_sq0 = zeros(ntheta)
    h_theta0 = zeros(ntheta)
    C_theta = zeros(ntheta)
    Iper0 = zeros(ntheta)
    kappas0 = zeros(ntheta)
    bsq0 = zeros(ntheta)
    jac0 = zeros(ntheta)
    bdot_theta_phi0 = zeros(ntheta)
    v0 = zeros(ntheta, 3, 3)
    v_zeta_psi0 = zeros(ntheta, 3)
    v_psi_theta0 = zeros(ntheta, 3)
    rz = (
        plasma_eq.rzphi_rsquared,
        plasma_eq.rzphi_offset,
        plasma_eq.rzphi_nu,
        plasma_eq.rzphi_jac
    )

    for i in eachindex(theta_vals)
        theta = theta_vals[i]
        f = ntuple(k -> rz[k].nodal_derivs.partials[1, flux_surface_index, i], 4)
        fx = ntuple(k -> rz[k].nodal_derivs.partials[2, flux_surface_index, i], 4)
        fy = ntuple(k -> rz[k].nodal_derivs.partials[3, flux_surface_index, i], 4)
        fxy = ntuple(k -> rz[k].nodal_derivs.partials[4, flux_surface_index, i], 4)

        rho = sqrt(f[1])
        eta = 2pi * (theta + f[2])
        ceta = cos(eta)
        seta = sin(eta)
        r = plasma_eq.ro + rho * ceta
        jac = f[4]

        rho_psi = fx[1] / (2 * rho)
        rho_theta = fy[1] / (2 * rho)
        rho_psitheta = fxy[1] / (2 * rho) - fx[1] * fy[1] / (4 * f[1] * rho)

        eta_psi = 2pi * fx[2]
        eta_theta = 2pi * (1 + fy[2])
        eta_psitheta = 2pi * fxy[2]

        Rpsi = rho_psi * ceta - rho * seta * eta_psi
        Rtheta = rho_theta * ceta - rho * seta * eta_theta
        Zpsi = rho_psi * seta + rho * ceta * eta_psi
        Ztheta = rho_theta * seta + rho * ceta * eta_theta

        Rpsitheta =
            rho_psitheta * ceta -
            rho_psi * seta * eta_theta -
            rho_theta * seta * eta_psi -
            rho * ceta * eta_psi * eta_theta -
            rho * seta * eta_psitheta

        Zpsitheta =
            rho_psitheta * seta +
            rho_psi * ceta * eta_theta +
            rho_theta * ceta * eta_psi -
            rho * seta * eta_psi * eta_theta +
            rho * ceta * eta_psitheta

        v = zeros(3, 3)
        v[1, 1] = fx[1] / (2 * rho * jac)
        v[1, 2] = fx[2] * 2pi * rho / jac
        v[1, 3] = fx[3] * r / jac
        v[2, 1] = fy[1] / (2 * rho * jac)
        v[2, 2] = (1 + fy[2]) * 2pi * rho / jac
        v[2, 3] = fy[3] * r / jac
        v[3, 3] = 2pi * r / jac

        w = zeros(3, 3)
        w[1, 1] = (1 + fy[2]) * (2pi)^2 * rho * r / jac
        w[1, 2] = -fy[1] * pi * r / (rho * jac)
        w[2, 1] = -fx[2] * (2pi)^2 * r * rho / jac
        w[2, 2] = fx[1] * pi * r / (rho * jac)
        w[3, 1] = (fx[2] * fy[3] - fx[3] * (1 + fy[2])) * 2pi * r * rho / jac
        w[3, 2] = (fx[3] * fy[1] - fx[1] * fy[3]) * r / (2 * rho * jac)
        w[3, 3] = 1 / (2pi * r)

        grad_psi = Vector(@view w[1, :])
        grad_theta = Vector(@view w[2, :])
        grad_zeta = Vector(@view w[3, :])

        v_theta_zeta = Vector(@view v[1, :])
        v_zeta_psi = Vector(@view v[2, :])
        v_psi_theta = Vector(@view v[3, :])

        B_vec = (v_zeta_psi + q0 .* v_psi_theta) * chi_prime0
        bsq = dot(B_vec, B_vec)
        gradpsi_sq = dot(grad_psi, grad_psi)

        f0[i, :] .= f
        fx0[i, :] .= fx
        fy0[i, :] .= fy
        fxy0[i, :] .= fxy
        rho0[i] = rho
        eta0[i] = eta
        r0[i] = r
        gradpsi_sq0[i] = gradpsi_sq
        h_theta0[i] = fy[3] / (2pi)
        C_theta[i] = (Rpsitheta * Ztheta - Rtheta * Zpsitheta) / (Rpsi * Ztheta - Rtheta * Zpsi)
        Iper0[i] = dot(q0 .* grad_theta .- grad_zeta, grad_psi) / gradpsi_sq
        bsq0[i] = bsq
        jac0[i] = jac
        bdot_theta_phi0[i] = dot(B_vec, v_theta_zeta)
        v0[i, :, :] .= v
        v_zeta_psi0[i, :] .= v_zeta_psi
        v_psi_theta0[i, :] .= v_psi_theta
    end

    dbsq_dtheta = _periodic_fft_lowpass_derivative(bsq0, theta_vals; nkeep=16)
    dinvbsq_dtheta = .-dbsq_dtheta ./ (bsq0 .^ 2)

    kappas0 .= -dinvbsq_dtheta .* two_pi_f0 ./ (2 .* jac0)

    int_fs = hcat(1 ./ gradpsi_sq0, r0 .* r0 ./ gradpsi_sq0)
    @views int_fs[end, :] .= int_fs[1, :]
    int_vals = FastInterpolations.integrate(cubic_interp(theta_vals, Series(int_fs); bc=PeriodicBC()))
    I0 = Float64(int_vals[1])
    IR = Float64(int_vals[2])
    Delta_tilde = Float64(I0 * two_pi_f0^2 + chi_prime0^2)

    return (
        theta_grid=theta_vals,
        q0=q0,
        q0prime=q0prime,
        p0prime=p0prime,
        two_pi_f0=two_pi_f0,
        chi_prime0=chi_prime0,
        r0=r0,
        rho0=rho0,
        eta0=eta0,
        f0=f0,
        fx0=fx0,
        fy0=fy0,
        fxy0=fxy0,
        gradpsi_sq0=gradpsi_sq0,
        h_theta0=h_theta0,
        C_theta=C_theta,
        Iper0=Iper0,
        bsq0=bsq0,
        jac0=jac0,
        bdot_theta_phi0=bdot_theta_phi0,
        v0=v0,
        v_zeta_psi0=v_zeta_psi0,
        v_psi_theta0=v_psi_theta0,
        kappas0=kappas0,
        I0=I0,
        IR=IR,
        Delta_tilde=Delta_tilde
    )
end

function _periodic_fft_lowpass_derivative(
    vals::AbstractVector{<:Real},
    xs::AbstractVector{<:Real};
    nkeep::Int=32
)
    nfull = length(vals)
    n = nfull - 1
    if n < 2
        throw(ArgumentError("Need at least 3 periodic samples for FFT derivative"))
    end

    period = Float64(xs[end] - xs[1])
    if !(period > 0.0)
        throw(ArgumentError("Periodic grid must span a positive interval"))
    end

    coeffs = FFTW.fft(ComplexF64.(vals[1:n]))
    deriv_coeffs = zeros(ComplexF64, n)
    maxmode = min(nkeep, fld(n, 2))

    @inbounds for j in 1:n
        mode = j <= div(n, 2) + 1 ? j - 1 : j - 1 - n
        if abs(mode) <= maxmode
            deriv_coeffs[j] = (im * 2pi * mode / period) * coeffs[j]
        end
    end

    deriv = real.(FFTW.ifft(deriv_coeffs))
    return vcat(deriv, deriv[1])
end

function _calculate_i_per_perturbed(
    bg::NamedTuple,
    theta_grid::AbstractVector,
    corr_qprime::Float64,
    corr_pprime::Float64;
    theta_k::Float64=0.0
)
    H = bg.q0 .+ bg.h_theta0
    A = 4pi^2 .* bg.r0 .* bg.r0 ./ (bg.chi_prime0^2 .* bg.gradpsi_sq0)
    B = bg.two_pi_f0^2 ./ (bg.chi_prime0^2 .* bg.gradpsi_sq0)

    weight_fs = hcat(H .* A, H .* B)
    @views weight_fs[end, :] .= weight_fs[1, :]
    weight_int = FastInterpolations.integrate(cubic_interp(theta_grid, Series(weight_fs); bc=PeriodicBC()))
    Abar_w = weight_int[1] / bg.q0
    Bbar_w = weight_int[2] / bg.q0

    denom = 1.0 + Bbar_w

    AperP = H .* ((A .- Abar_w) .- (Abar_w / denom) .* (B .- Bbar_w))
    Aperq = bg.h_theta0 ./ bg.q0 .+ (H ./ bg.q0) .* ((B .- Bbar_w) ./ denom)

    dI_dtheta = AperP .* corr_pprime .+ Aperq .* corr_qprime

    dI_dtheta_periodic = Vector{Float64}(dI_dtheta)
    dI_dtheta_periodic[end] = dI_dtheta_periodic[1]
    i_per_primitive = vec(FastInterpolations.cumulative_integrate(cubic_interp(theta_grid, dI_dtheta_periodic; bc=PeriodicBC())))
    i_per_spl = cubic_interp(Float64.(theta_grid), i_per_primitive; bc=CubicFit())

    theta_min = theta_grid[1]
    theta_period = theta_grid[end] - theta_min
    theta_k_wrapped = mod(theta_k - theta_min, theta_period) + theta_min
    i_per_theta_k = i_per_spl(theta_k_wrapped)
    i_per_perturbed = copy(i_per_primitive .- i_per_theta_k)


    return i_per_perturbed
end

function prepare_ballooning_coefficients(
    flux_surface_index::Int,
    plasma_eq::Equilibrium.PlasmaEquilibrium;
    corr_qprime::Float64=0.0,
    corr_pprime::Float64=0.0,
    theta_k::Float64=0.0
)
    mtheta = length(plasma_eq.rzphi_ys) - 1
    theta_grid = Vector(plasma_eq.rzphi_ys)

    bg = _collect_surface_response_background(flux_surface_index, plasma_eq, theta_grid)

    theta_min = theta_grid[1]
    theta_period = theta_grid[end] - theta_min
    theta_k_wrapped = mod(theta_k - theta_min, theta_period) + theta_min
    theta_k_reference = Float64(theta_k)

    i_per0_periodic = Vector{Float64}(bg.Iper0)
    i_per0_periodic[end] = i_per0_periodic[1]
    i_per0_spl = cubic_interp(theta_grid, i_per0_periodic; bc=PeriodicBC())
    i_per0_theta_k = i_per0_spl(theta_k_wrapped)
    i_per0_theta_ref = i_per0_spl(theta_min)
    i_per0_thetak_shift = i_per0_theta_k - i_per0_theta_ref
    i_per0_shifted = bg.Iper0 .- i_per0_theta_k

    two_pi_f = bg.two_pi_f0
    pressure_gradient = bg.p0prime
    q = bg.q0
    q_derivative = bg.q0prime
    chi_prime = bg.chi_prime0

    bsq = bg.bsq0
    jac_arr_loop1 = bg.jac0
    jac_psi_loop1 = zeros(mtheta + 1)
    bsq_psi_loop1 = zeros(mtheta + 1)
    bdot_theta_phi_loop1 = bg.bdot_theta_phi0

    for itheta in 0:mtheta
        idx = itheta + 1
        jac_psi_loop1[idx] = bg.fx0[idx, 4]
        drfac_dp = bg.fx0[idx, 1] / (2 * bg.rho0[idx])
        eta_dp = 2pi * bg.fx0[idx, 2]
        dr_dp = drfac_dp * cos(bg.eta0[idx]) - bg.rho0[idx] * sin(bg.eta0[idx]) * eta_dp

        dv21_dp =
            (bg.fxy0[idx, 1] / (2 * bg.rho0[idx]) - bg.fx0[idx, 1] * bg.fy0[idx, 1] / (4 * bg.f0[idx, 1] * bg.rho0[idx])) / bg.jac0[idx] -
            bg.v0[idx, 2, 1] * jac_psi_loop1[idx] / bg.jac0[idx]
        dv22_dp = (bg.fxy0[idx, 2] * 2pi * bg.rho0[idx] + (1 + bg.fy0[idx, 2]) * 2pi * drfac_dp) / bg.jac0[idx] - bg.v0[idx, 2, 2] * jac_psi_loop1[idx] / bg.jac0[idx]
        dv23_dp = (bg.fxy0[idx, 3] * bg.r0[idx] + bg.fy0[idx, 3] * dr_dp) / bg.jac0[idx] - bg.v0[idx, 2, 3] * jac_psi_loop1[idx] / bg.jac0[idx]
        dv33_dp = (2pi * dr_dp) / bg.jac0[idx] - bg.v0[idx, 3, 3] * jac_psi_loop1[idx] / bg.jac0[idx]

        grad_theta_dp = [dv21_dp, dv22_dp, dv23_dp]
        grad_zeta_dp = [0.0, 0.0, dv33_dp]
        dbase_dp = grad_theta_dp + q * grad_zeta_dp + q_derivative * (@view bg.v0[idx, 3, :])
        dB_dp = dbase_dp * chi_prime
        bsq_psi_loop1[idx] = 2 * dot((@view bg.v_zeta_psi0[idx, :]) .+ q .* (@view bg.v_psi_theta0[idx, :]), dB_dp) * chi_prime
    end

    bmag = sqrt.(bsq)
    kappaw_periodic = zeros(mtheta + 1)

    spl1_vals = jac_arr_loop1 .* bdot_theta_phi_loop1 ./ bmag
    spl1_deriv = _periodic_fft_lowpass_derivative(spl1_vals, theta_grid; nkeep=32)

    cell1 = -(jac_psi_loop1 ./ jac_arr_loop1 .+ 0.5 .* bsq_psi_loop1 ./ bsq) ./ chi_prime
    cell2 = spl1_deriv ./ (jac_arr_loop1 .* bmag)
    cell3 = two_pi_f .* q_derivative ./ (bsq .* jac_arr_loop1)

    kappaw_periodic .= cell1 .+ cell2 .+ cell3

    pressure_gradient += corr_pprime
    q_derivative += corr_qprime
    i_per_perturbed = _calculate_i_per_perturbed(
        bg,
        theta_grid,
        corr_qprime,
        corr_pprime;
        theta_k=theta_k_wrapped
    )

    nabla_beta_sq_b_sq_periodic = zeros(mtheta + 1)
    nabla_beta_sq_b_sq_peculiar_1st = zeros(mtheta + 1)
    nabla_beta_sq_b_sq_peculiar_2nd = zeros(mtheta + 1)
    jac_chiprime = zeros(mtheta + 1)
    pprime_chiprime = fill(pressure_gradient / chi_prime, mtheta + 1)

    for itheta in 0:mtheta
        grad_psi_sq = bg.gradpsi_sq0[itheta+1]
        i_per_val = i_per0_shifted[itheta+1]
        i_per_total = i_per_val + i_per_perturbed[itheta+1]

        term1 = 1.0 / (chi_prime^2 * grad_psi_sq)
        term2_factor = grad_psi_sq / bsq[itheta+1]

        nabla_beta_sq_b_sq_periodic[itheta+1] = term1 + term2_factor * (i_per_total^2 - 2.0 * theta_k_reference * q_derivative * i_per_total + (theta_k_reference * q_derivative)^2)
        nabla_beta_sq_b_sq_peculiar_1st[itheta+1] = term2_factor * (2.0 * q_derivative * i_per_total - 2.0 * theta_k_reference * q_derivative^2)
        nabla_beta_sq_b_sq_peculiar_2nd[itheta+1] = term2_factor * (q_derivative^2)

        jac_chiprime[itheta+1] = jac_arr_loop1[itheta+1] / chi_prime
    end

    kappas = bg.kappas0
    kappaw_periodic .+= -kappas .* i_per_perturbed .+ (theta_k_reference .* q_derivative .+ i_per0_thetak_shift) .* kappas
    kappaw_secular = q_derivative .* kappas

    bf_fs = zeros(mtheta + 1, 9)
    bf_fs[:, 1] = nabla_beta_sq_b_sq_periodic
    bf_fs[:, 2] = nabla_beta_sq_b_sq_peculiar_1st
    bf_fs[:, 3] = nabla_beta_sq_b_sq_peculiar_2nd
    bf_fs[:, 4] = kappaw_periodic
    bf_fs[:, 5] = kappaw_secular
    bf_fs[:, 6] = jac_chiprime
    bf_fs[:, 7] = pprime_chiprime
    bf_fs[:, 8] = bsq

    sigma = .-(two_pi_f .* pressure_gradient .* q_derivative) ./ (chi_prime^2 .* bsq)
    bf_fs[:, 9] = sigma

    m0_12 = jac_chiprime ./ nabla_beta_sq_b_sq_peculiar_2nd
    m0_21 = -2.0 .* jac_chiprime .* pprime_chiprime .* kappaw_periodic

    n0_fs = zeros(mtheta + 1, 4)
    n0_fs[:, 1] = 0.5 .+ m0_12 .* sigma
    n0_fs[:, 2] = m0_12
    n0_fs[:, 3] = m0_21 .- sigma .- m0_12 .* sigma .^ 2
    n0_fs[:, 4] = -0.5 .- m0_12 .* sigma

    @views n0_fs[end, :] .= n0_fs[1, :]
    n0_int = FastInterpolations.integrate(cubic_interp(theta_grid, Series(n0_fs); bc=PeriodicBC()))

    d0bar = zeros(2, 2)
    d0bar[1, 1] = n0_int[1]
    d0bar[1, 2] = n0_int[2]
    d0bar[2, 1] = n0_int[3]
    d0bar[2, 2] = n0_int[4]

    @views bf_fs[end, :] .= bf_fs[1, :]
    ode_coefficient_spline = (xs=theta_grid, itp=cubic_interp(theta_grid, Series(bf_fs); bc=PeriodicBC()))

    return (
        ode_coefficient_spline=ode_coefficient_spline,
        di=det(d0bar),
        theta_grid=theta_grid
    )
end


"""
    salpha_reference(psi_idx, plasma_eq)

Compute local s-alpha reference values and physical profile derivatives at one flux surface.
"""
function salpha_reference(
    psi_idx::Int,
    plasma_eq::Equilibrium.PlasmaEquilibrium
)
    profiles = plasma_eq.profiles
    psio = plasma_eq.psio
    mu0 = Equilibrium.mu0

    npsi = length(profiles.xs)
    if psi_idx < 1 || psi_idx > npsi
        throw(ArgumentError("psi_idx=$psi_idx is out of bounds for npsi=$npsi"))
    end
    if abs(psio) <= Base.eps(Float64)
        throw(ArgumentError("plasma_eq.psio is too small to compute physical derivatives"))
    end

    dVdpsi_int = FastInterpolations.cumulative_integrate(profiles.dVdpsi_spline)
    volume = max(Float64(dVdpsi_int[psi_idx]), 0.0)
    psi = profiles.xs[psi_idx]
    vprime_phys = profiles.dVdpsi_spline.y[psi_idx] / psio
    q_ref = profiles.q_spline.y[psi_idx]
    qprime_norm_ref = profiles.q_deriv(psi)
    pprime_norm_ref = profiles.P_deriv(psi)
    qprime_ref = qprime_norm_ref / psio
    pprime_ref = (pprime_norm_ref / mu0) / psio

    denom = q_ref * vprime_phys
    s_ref = abs(denom) > Base.eps(Float64) ? (2.0 * volume * qprime_ref / denom) : NaN

    major_radius = max(plasma_eq.ro, Base.eps(Float64))
    alpha_geom = max(volume / (2π * major_radius), 0.0)
    alpha_ref = -2.0 * mu0 * pprime_ref * vprime_phys * sqrt(alpha_geom)

    return (
        s_ref=s_ref,
        alpha_ref=alpha_ref,
        volume_ref=volume,
        vprime_ref=vprime_phys,
        q_ref=q_ref,
        qprime_ref=qprime_ref,
        pprime_ref=pprime_ref,
        qprime_norm_ref=qprime_norm_ref,
        pprime_norm_ref=pprime_norm_ref
    )
end

"""
    ballooning_delta_prime(psi_idx, plasma_eq; corr_qprime=0.0, corr_pprime=0.0, theta_k=0.0)

Evaluate one corrected local ballooning point. Corrections use the same
normalized units as `prepare_ballooning_coefficients`: add `corr_qprime` to
`dq/dpsi_norm` and `corr_pprime` to `d(mu0*p)/dpsi_norm`. `di` is the
`det(d0bar)` diagnostic assembled from the same corrected coefficients.
"""
function ballooning_delta_prime(
    psi_idx::Int,
    plasma_eq::Equilibrium.PlasmaEquilibrium;
    corr_qprime::Float64=0.0,
    corr_pprime::Float64=0.0,
    theta_k::Float64=0.0
)
    coeff_data = prepare_ballooning_coefficients(
        psi_idx,
        plasma_eq;
        corr_qprime=corr_qprime,
        corr_pprime=corr_pprime,
        theta_k=theta_k
    )
    delta_prime = integrate_ballooning_ode(
        coeff_data.ode_coefficient_spline;
        theta_k=theta_k
    ).value

    return (delta_prime=delta_prime, di=coeff_data.di)
end

"""
    scan_delta_prime_map(psi_idx, plasma_eq; s_scales, alpha_scales, ...)

Run a local s-alpha style scan at one flux surface using the `Bal.jl`
makefromZero ballooning path. The scan perturbs physical `dq/dpsi` and
`dp/dpsi`, converts them to the normalized units expected by the local
ballooning evaluator, and returns delta-prime and `det(d0bar)` `Di` maps.
"""
function scan_delta_prime_map(
    psi_idx::Int,
    plasma_eq::Equilibrium.PlasmaEquilibrium;
    ctrl::ForceFreeStatesControl=ForceFreeStatesControl(; verbose=false),
    theta_k::Float64=0.0,
    s_scales::AbstractVector{<:Real},
    alpha_scales::AbstractVector{<:Real}
)
    ref = salpha_reference(psi_idx, plasma_eq)

    s_scales_f = Float64.(s_scales)
    alpha_scales_f = Float64.(alpha_scales)
    isempty(s_scales_f) && throw(ArgumentError("s_scales must not be empty"))
    isempty(alpha_scales_f) && throw(ArgumentError("alpha_scales must not be empty"))

    delta_prime_map = fill(NaN, length(s_scales_f), length(alpha_scales_f))
    di_map = fill(NaN, length(s_scales_f), length(alpha_scales_f))

    for is in eachindex(s_scales_f), ia in eachindex(alpha_scales_f)
        corr_qprime = ref.qprime_norm_ref * (s_scales_f[is] - 1.0)
        corr_pprime = ref.pprime_norm_ref * (alpha_scales_f[ia] - 1.0)

        result = try
            ballooning_delta_prime(
                psi_idx,
                plasma_eq;
                corr_qprime=corr_qprime,
                corr_pprime=corr_pprime,
                theta_k=theta_k
            )
        catch err
            if ctrl.verbose
                @warn "s-alpha scan point failed" psi_idx is ia err
            end
            nothing
        end

        if result !== nothing
            delta_prime_map[is, ia] = result.delta_prime
            di_map[is, ia] = result.di
        end
    end

    s_values = ref.s_ref .* s_scales_f
    alpha_values = ref.alpha_ref .* alpha_scales_f

    return (
        values=delta_prime_map,
        delta_prime=delta_prime_map,
        di=di_map,
        di_values=di_map,
        s_values=s_values,
        alpha_values=alpha_values,
        s_scales=s_scales_f,
        alpha_scales=alpha_scales_f,
        dqdpsi_ref=ref.qprime_ref,
        pprime_ref=ref.pprime_ref,
        theta_k=theta_k,
        reference=ref
    )
end

"""
    critical_ballooning_alpha(psi_idx, plasma_eq; theta_k=0.0, max_alpha_scale=8.0, n_scan=24, tol=1e-3)

First infinite-n ballooning stability boundary at one flux surface: holding the
equilibrium magnetic shear fixed (`corr_qprime = 0`), find the critical pressure
gradient `α` at which the marginal ballooning Δ' first crosses zero as `α` increases
from zero. `α` is linear in `dp/dψ`, so the search is over a scaling of the reference
gradient and the boundary maps directly to `α_crit = α_ref * scale_crit`.

The `α = 0` (zero pressure gradient) point is ballooning stable; its Δ' sign defines
the stable side, and the first scaling at which Δ' flips away from it is the boundary.

Returns `(alpha_crit, alpha_scale_crit, found)`. When no crossing exists within
`max_alpha_scale` (always stable, or second-stability access) `alpha_crit = NaN` and
`found = false`.
"""
function critical_ballooning_alpha(
    psi_idx::Int,
    plasma_eq::Equilibrium.PlasmaEquilibrium;
    theta_k::Float64=0.0,
    max_alpha_scale::Float64=8.0,
    n_scan::Int=24,
    tol::Float64=1e-3
)
    ref = salpha_reference(psi_idx, plasma_eq)

    # Δ' as a function of the α scaling at fixed magnetic shear.
    delta_at(scale) = ballooning_delta_prime(
        psi_idx,
        plasma_eq;
        corr_qprime=0.0,
        corr_pprime=ref.pprime_norm_ref * (scale - 1.0),
        theta_k=theta_k
    ).delta_prime

    stable_sign = sign(delta_at(0.0))
    if stable_sign == 0.0
        return (alpha_crit=NaN, alpha_scale_crit=NaN, found=false)
    end

    # Scan α upward and bracket the first scaling where Δ' leaves the stable side.
    prev_scale = 0.0
    lo = NaN
    hi = NaN
    for k in 1:n_scan
        scale = max_alpha_scale * k / n_scan
        d = delta_at(scale)
        if isfinite(d) && sign(d) != stable_sign
            lo = prev_scale
            hi = scale
            break
        end
        prev_scale = scale
    end

    isnan(hi) && return (alpha_crit=NaN, alpha_scale_crit=NaN, found=false)

    # Bisect the bracket to the requested tolerance.
    while (hi - lo) > tol
        mid = 0.5 * (lo + hi)
        d = delta_at(mid)
        if !isfinite(d) || sign(d) == stable_sign
            lo = mid
        else
            hi = mid
        end
    end
    scale_crit = 0.5 * (lo + hi)

    return (alpha_crit=ref.alpha_ref * scale_crit, alpha_scale_crit=scale_crit, found=true)
end

"""
    ballooning_alpha_boundary(ctrl, plasma_eq; theta_k=0.0)

Profile driver for the BALOO-style ballooning stability diagram. Loops over flux
surfaces returning the experimental pressure gradient `alpha` (from
[`salpha_reference`](@ref)) and the first stability boundary `alpha_critical` (from
[`critical_ballooning_alpha`](@ref)) versus normalized flux `psi`. Surfaces whose
experimental `alpha` exceeds `alpha_critical` are ballooning-unstable.

Per-surface failures and surfaces with no boundary within range are returned as `NaN`.
"""
function ballooning_alpha_boundary(
    ctrl::ForceFreeStatesControl,
    plasma_eq::Equilibrium.PlasmaEquilibrium;
    theta_k::Float64=0.0
)
    xs = plasma_eq.profiles.xs
    npsi = length(xs)
    psi = Vector{Float64}(xs)
    alpha = fill(NaN, npsi)
    alpha_critical = fill(NaN, npsi)

    for i in 1:npsi
        xs[i] > 1.0 && continue
        try
            alpha[i] = salpha_reference(i, plasma_eq).alpha_ref
            alpha_critical[i] = critical_ballooning_alpha(i, plasma_eq; theta_k=theta_k).alpha_crit
        catch err
            if ctrl.verbose
                @warn "ballooning alpha boundary failed" psi_idx = i exception = err
            end
        end
    end

    return (psi=psi, alpha=alpha, alpha_critical=alpha_critical)
end

# ======================================================================
#   ODE Integration
#   Integrates the ideal marginal ballooning equations
# ======================================================================
"""
    integrate_ballooning_ode(ode_coefficient_spline; theta_k=0.0)

Integrates the ballooning ODE from `-Infinity` to `-1e-3` and `+Infinity` to `+1e-3`.
Computes Delta' = (y'/y)_right - (y'/y)_left.

## Returns

  - `delta_prime`: The stability index Δ'.
"""
function integrate_ballooning_ode(ode_coefficient_spline; theta_k::Float64=0.0)
    TOLERANCE = 1e-8
    MINIMUM_STEP = 1e-10
    MATCHING_POINT = 1e-3
    THETA_MAX0 = 16.5

    theta_start_left = -THETA_MAX0
    theta_end_left = -MATCHING_POINT
    theta_start_right = THETA_MAX0
    theta_end_right = MATCHING_POINT

    initial_condition_left = [0.0, 1.0]
    initial_condition_right = [0.0, -1.0]

    problem_left = ODEProblem(
        compute_ballooning_ode!,
        initial_condition_left,
        (theta_start_left, theta_end_left),
        (ode_coefficient_spline, theta_k)
    )
    sol_left = solve(
        problem_left,
        DP5();
        reltol=TOLERANCE,
        abstol=TOLERANCE^2,
        dtmin=MINIMUM_STEP,
        adaptive=true
    )
    if sol_left.retcode != ReturnCode.Success
        return (value=NaN,)
    end

    problem_right = ODEProblem(
        compute_ballooning_ode!,
        initial_condition_right,
        (theta_start_right, theta_end_right),
        (ode_coefficient_spline, theta_k)
    )
    sol_right = solve(
        problem_right,
        DP5();
        reltol=TOLERANCE,
        abstol=TOLERANCE^2,
        dtmin=MINIMUM_STEP,
        adaptive=true
    )
    if sol_right.retcode != ReturnCode.Success
        return (value=NaN,)
    end

    log_deriv_left = sol_left.u[end][2] / sol_left.u[end][1]
    log_deriv_right = sol_right.u[end][2] / sol_right.u[end][1]

    return (value=log_deriv_right - log_deriv_left,)
end



# ======================================================================
#   ODE System Definition
#   Differential equations for marginal ballooning stability
# ======================================================================
"""
    compute_ballooning_ode!(derivatives, solution, parameters, poloidal_angle)

Defines the RHS of the first-order ODE system for ideal marginal ballooning modes.
Used by the DifferentialEquations.jl ODE solver.

The second-order equation `d/dθ(f·dy/dθ) + g·y = 0` is rewritten as a 2x1 system:

  - `dy₁/dθ = y₂ / f`
  - `dy₂/dθ = -g · y₁`

where y₁ is the solution and y₂ = f·dy/dθ.

## Arguments

  - `derivatives::Vector{Float64}`: Output vector [dy₁/dθ, dy₂/dθ].
  - `solution::Vector{Float64}`: Current state vector [y₁, y₂].
  - `parameters::Tuple`: (ode_coefficient_spline, reference_angle).
  - `poloidal_angle::Float64`: Current poloidal angle θ (independent variable).
"""
function compute_ballooning_ode!(derivatives, solution, parameters, poloidal_angle)
    ode_coefficient_spline = parameters[1]
    theta_min = first(ode_coefficient_spline.xs)
    theta_period = last(ode_coefficient_spline.xs) - theta_min
    theta_wrapped = mod(Float64(poloidal_angle) - theta_min, theta_period) + theta_min
    coeffs = ode_coefficient_spline.itp(theta_wrapped)

    periodic = coeffs[1]
    peculiar_1st = coeffs[2]
    peculiar_2nd = coeffs[3]
    kappaw_periodic = coeffs[4]
    kappaw_secular = coeffs[5]
    jac_chiprime = coeffs[6]
    pprime_chiprime = coeffs[7]

    f = (periodic + poloidal_angle * peculiar_1st + poloidal_angle^2 * peculiar_2nd) / jac_chiprime
    kappaw = kappaw_periodic - poloidal_angle * kappaw_secular
    g = -2.0 * jac_chiprime * pprime_chiprime * kappaw

    derivatives[1] = solution[2] / f
    derivatives[2] = g * solution[1]
end
