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

  - `ctrl::DconControl`: Control parameters for the analysis.
  - `locstab_fs::Matrix{Float64}`: Local stability matrix to store results (modified in place).
  - `plasma_eq::Equilibrium.PlasmaEquilibrium`: Plasma equilibrium data.

This function modifies `locstab_fs` in place with:

  - Column 1: `det(d0bar) * ψ`
  - Column 4: Delta Prime (Δ')
"""
function compute_ballooning_stability!(
    ctrl::DconControl,
    locstab_fs::Matrix{Float64},
    plasma_eq::Equilibrium.PlasmaEquilibrium;
    theta_k::Float64=0.0,
    compute_delta_prime::Bool=true
)

    if ctrl.verbose
        println("Evaluating local high-n ballooning stability...")
    end

    num_psi = length(plasma_eq.sq.xs)

    # Loop over flux surfaces
    for flux_surface_index in 1:num_psi

        coeff_data = prepare_ballooning_coefficients(flux_surface_index, plasma_eq; theta_k=theta_k)
        locstab_fs[flux_surface_index, 1] = coeff_data.di * plasma_eq.sq.xs[flux_surface_index]

        if compute_delta_prime && plasma_eq.sq.xs[flux_surface_index] <= 1.0
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





function _collect_surface_response_background(
    flux_surface_index::Int,
    plasma_eq::Equilibrium.PlasmaEquilibrium,
    theta_grid::AbstractVector{<:Real}
)::NamedTuple
    sq = plasma_eq.sq

    theta_vals = Float64.(theta_grid)
    ntheta = length(theta_vals)

    q0 = Float64(sq.fs[flux_surface_index, 4])
    q0prime = Float64(sq.fs1[flux_surface_index, 4])
    p0prime = Float64(sq.fs1[flux_surface_index, 2])
    two_pi_f0 = Float64(sq.fs[flux_surface_index, 1])
    chi_prime0 = Float64(2pi * plasma_eq.psio)
    psi0 = plasma_eq.rzphi.xs[flux_surface_index]

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

    for i in eachindex(theta_vals)
        theta = theta_vals[i]
        f, fx, fy, _, fxy, _ = Spl.bicube_deriv2!(plasma_eq.rzphi, psi0, theta)

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

    int_spl = Spl.CubicSpline(
        theta_vals,
        hcat(1 ./ gradpsi_sq0, r0 .* r0 ./ gradpsi_sq0);
        bctype="periodic"
    )
    Spl.spline_integrate!(int_spl)

    I0 = Float64(int_spl.fsi[end, 1])
    IR = Float64(int_spl.fsi[end, 2])
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

    weight_spl = Spl.CubicSpline(theta_grid, hcat(H .* A, H .* B); bctype="periodic")
    Spl.spline_integrate!(weight_spl)
    Abar_w = weight_spl.fsi[end, 1] / bg.q0
    Bbar_w = weight_spl.fsi[end, 2] / bg.q0

    denom = 1.0 + Bbar_w

    AperP = H .* ((A .- Abar_w) .- (Abar_w / denom) .* (B .- Bbar_w))
    Aperq = bg.h_theta0 ./ bg.q0 .+ (H ./ bg.q0) .* ((B .- Bbar_w) ./ denom)

    dI_dtheta = AperP .* corr_pprime .+ Aperq .* corr_qprime

    dI_spl = Spl.CubicSpline(theta_grid, reshape(dI_dtheta, :, 1); bctype="periodic")
    Spl.spline_integrate!(dI_spl)
    i_per_primitive = copy(dI_spl.fsi[:, 1])
    i_per_spl = Spl.CubicSpline(theta_grid, reshape(i_per_primitive, :, 1); bctype="extrap")

    theta_min = theta_grid[1]
    theta_period = theta_grid[end] - theta_min
    theta_k_wrapped = mod(theta_k - theta_min, theta_period) + theta_min
    i_per_theta_k = Spl.spline_eval!(i_per_spl, theta_k_wrapped)[1]
    i_per_perturbed = copy(i_per_spl.fs[:, 1] .- i_per_theta_k)


    return i_per_perturbed
end

function prepare_ballooning_coefficients(
    flux_surface_index::Int,
    plasma_eq::Equilibrium.PlasmaEquilibrium;
    corr_qprime::Float64=0.0,
    corr_pprime::Float64=0.0,
    theta_k::Float64=0.0
)
    sq = plasma_eq.sq
    rzphi = plasma_eq.rzphi
    mtheta = length(rzphi.ys) - 1
    theta_grid = Vector(rzphi.ys)

    bg = _collect_surface_response_background(flux_surface_index, plasma_eq, theta_grid)

    theta_min = theta_grid[1]
    theta_period = theta_grid[end] - theta_min
    theta_k_wrapped = mod(theta_k - theta_min, theta_period) + theta_min
    theta_k_reference = Float64(theta_k)

    i_per0_spl = Spl.CubicSpline(theta_grid, reshape(bg.Iper0, :, 1); bctype="periodic")
    i_per0_theta_k = Spl.spline_eval!(i_per0_spl, theta_k_wrapped)[1]
    i_per0_theta_ref = Spl.spline_eval!(i_per0_spl, theta_min)[1]
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

        dv21_dp = (bg.fxy0[idx, 1] / (2 * bg.rho0[idx]) - bg.fx0[idx, 1] * bg.fy0[idx, 1] / (4 * bg.f0[idx, 1] * bg.rho0[idx])) / bg.jac0[idx] - bg.v0[idx, 2, 1] * jac_psi_loop1[idx] / bg.jac0[idx]
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

    spl1 = Spl.CubicSpline(theta_grid, jac_arr_loop1 .* bdot_theta_phi_loop1 ./ bmag; bctype="periodic")

    cell1 = -(jac_psi_loop1 ./ jac_arr_loop1 .+ 0.5 .* bsq_psi_loop1 ./ bsq) ./ chi_prime
    cell2 = spl1.fs1[:, 1] ./ (jac_arr_loop1 .* bmag)
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

    n0_spline = Spl.CubicSpline(theta_grid, n0_fs; bctype="periodic")
    Spl.spline_integrate!(n0_spline)

    d0bar = zeros(2, 2)
    d0bar[1, 1] = n0_spline.fsi[end, 1]
    d0bar[1, 2] = n0_spline.fsi[end, 2]
    d0bar[2, 1] = n0_spline.fsi[end, 3]
    d0bar[2, 2] = n0_spline.fsi[end, 4]

    ode_coefficient_spline = Spl.CubicSpline(theta_grid, bf_fs; bctype="periodic")

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
    sq = plasma_eq.sq
    psio = plasma_eq.psio
    mu0 = Equilibrium.mu0

    npsi = length(sq.xs)
    if psi_idx < 1 || psi_idx > npsi
        throw(ArgumentError("psi_idx=$psi_idx is out of bounds for npsi=$npsi"))
    end
    if abs(psio) <= Base.eps(Float64)
        throw(ArgumentError("plasma_eq.psio is too small to compute physical derivatives"))
    end

    Spl.spline_integrate!(sq)
    volume = max(Float64(sq.fsi[psi_idx, 3]), 0.0)
    vprime_phys = sq.fs[psi_idx, 3] / psio
    q_ref = sq.fs[psi_idx, 4]
    qprime_norm_ref = sq.fs1[psi_idx, 4]
    pprime_norm_ref = sq.fs1[psi_idx, 2]
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
    ctrl::DconControl=DconControl(; verbose=false),
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
function integrate_ballooning_ode(
    ode_coefficient_spline::Spl.CubicSpline;
    theta_k::Float64=0.0
)
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
    coeff_matrix = Spl.spline_eval(ode_coefficient_spline, [poloidal_angle])

    periodic = coeff_matrix[1, 1]
    peculiar_1st = coeff_matrix[1, 2]
    peculiar_2nd = coeff_matrix[1, 3]
    kappaw_periodic = coeff_matrix[1, 4]
    kappaw_secular = coeff_matrix[1, 5]
    jac_chiprime = coeff_matrix[1, 6]
    pprime_chiprime = coeff_matrix[1, 7]

    f = (periodic + poloidal_angle * peculiar_1st + poloidal_angle^2 * peculiar_2nd) / jac_chiprime
    kappaw = kappaw_periodic - poloidal_angle * kappaw_secular
    g = -2.0 * jac_chiprime * pprime_chiprime * kappaw

    derivatives[1] = solution[2] / f
    derivatives[2] = g * solution[1]
end