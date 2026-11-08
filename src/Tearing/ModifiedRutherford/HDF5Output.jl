# HDF5Output.jl
#
# Writer for a `ModifiedRutherfordResult`, under the caller's parent group:
#
#   Tearing/
#   └── ModifiedRutherford/
#       ├── w                      -- width grid (ψ_N)
#       ├── w_m_star               -- smallest seed threshold over the surfaces
#       ├── PerSurface/            -- surface identity, indices, term inputs, roots and flags
#       └── Terms/                 -- F(w), each term and w_d(w), surface × width

"""
    save_to_hdf5(parent::Union{HDF5.File,HDF5.Group}, result::ModifiedRutherfordResult)

Write `result` into `Tearing/ModifiedRutherford/` of `parent`, replacing any existing one and keeping the rest of
`Tearing/`, and record its deck section.
"""
function save_to_hdf5(parent::Union{HDF5.File,HDF5.Group}, result::ModifiedRutherfordResult)
    tearing = haskey(parent, "Tearing") ? parent["Tearing"] : create_group(parent, "Tearing")
    g = replace_group!(tearing, "ModifiedRutherford")
    g["w"] = result.w
    g["w_m_star"] = result.w_m_star
    for key in _MRE_PER_SURFACE
        g["PerSurface/$key"] = getfield(result, key)
    end
    # Flags as 0/1 integers, readable without an HDF5 boolean type.
    for key in _MRE_FLAGS
        g["PerSurface/$key"] = Int8.(getfield(result, key))
    end
    for key in _MRE_TERMS
        g["Terms/$key"] = getfield(result, key)
    end
    Utilities.HDF5Annotations.annotate!(g, MRE_H5_ANNOTATIONS)
    Utilities.record_inputs!(parent, result)
    return g
end

const _MRE_PER_SURFACE = (:psi, :q, :m, :n, :delta_prime, :delta_prime_imag, :D_I, :D_R, :H, :D_nc, :f_trap, :mu_ratio, :zeff, :Wc_prefac,
    :chi_perp, :v_te, :R0, :omega_star_i, :omega_star_e, :omega_hat, :omega_hat_pi, :beta_p, :rho_theta, :w_ib, :eps, :gamma, :delta_r, :w_lo,
    :w_m, :w_sat)
const _MRE_FLAGS = (:fully_stabilized, :unstable_any_seed, :interchange_unstable, :linear_unstable, :below_w_ib, :large_delta_prime_w,
    :complex_delta_prime)
const _MRE_TERMS = (:F, :classical, :bootstrap, :curvature, :polarization, :w_d)

# Julia's surface × width matrices land in HDF5 as (width, surface).
const _MRE_TERM_DIMS = ("w", "surface")

const MRE_H5_ANNOTATIONS = [
    "w" => (; long_name="full island width grid", units="psi_N", dims=("w",)),
    "w_m_star" => (; long_name="smallest seed-threshold full width over the surfaces, NaN if none", units="psi_N"),
    "PerSurface/psi" => (; long_name="normalized poloidal flux of the rational surface", units="psi_N", dims=("surface",)),
    "PerSurface/q" => (; long_name="safety factor at the surface", dims=("surface",)),
    "PerSurface/m" => (; long_name="resonant poloidal mode number", dims=("surface",)),
    "PerSurface/n" => (; long_name="resonant toroidal mode number", dims=("surface",)),
    "PerSurface/delta_prime" => (;
        long_name="real part of the diagonal Delta' (PEST-3 convention; cylindrical -2m referred to psi_N with delta_prime_source = cylindrical), times delta_prime_factor",
        dims=("surface",)
    ),
    "PerSurface/delta_prime_imag" => (; long_name="imaginary part of the diagonal Delta'", dims=("surface",)),
    "PerSurface/D_I" => (; long_name="Mercier index D_I = E + F + H - 1/4", dims=("surface",)),
    "PerSurface/D_R" => (; long_name="resistive interchange index D_R = E + F + H^2", dims=("surface",)),
    "PerSurface/H" => (; long_name="Glasser-Greene-Johnson coefficient H", dims=("surface",)),
    "PerSurface/D_nc" => (; long_name="bootstrap drive coefficient D_nc (Hegna 1999)", dims=("surface",)),
    "PerSurface/f_trap" => (; long_name="Sauter 2002 trapped fraction", dims=("surface",)),
    "PerSurface/mu_ratio" => (; long_name="banana-limit viscosity ratio mu_e/(nu_e + mu_e)", dims=("surface",)),
    "PerSurface/zeff" => (; long_name="effective charge Z_eff", dims=("surface",)),
    "PerSurface/Wc_prefac" => (; long_name="transport-width prefactor W_c (Schlutt & Hegna 2012), before chi_perp/chi_par and 1/m^2", units="psi_N^4",
        dims=("surface",)),
    "PerSurface/chi_perp" => (; long_name="perpendicular electron heat diffusivity used in w_d", units="m^2/s", dims=("surface",)),
    "PerSurface/v_te" => (; long_name="electron thermal speed sqrt(2T_e/m_e)", units="m/s", dims=("surface",)),
    "PerSurface/R0" => (; long_name="major radius (R_max + R_min)/2 of the surface, used in lambda_par", units="m", dims=("surface",)),
    "PerSurface/omega_star_i" => (; long_name="ion diamagnetic frequency from the density gradient", units="rad/s", dims=("surface",)),
    "PerSurface/omega_star_e" => (; long_name="electron diamagnetic frequency from the density gradient", units="rad/s", dims=("surface",)),
    "PerSurface/omega_hat" => (; long_name="island frequency normalized to the ion diamagnetic frequency", dims=("surface",)),
    "PerSurface/omega_hat_pi" => (; long_name="omega_*pi/omega_*i = 1 + eta_i", dims=("surface",)),
    "PerSurface/beta_p" => (; long_name="poloidal ion beta 2 mu0 n_i T_i/<B_pol^2>", dims=("surface",)),
    "PerSurface/rho_theta" => (; long_name="poloidal ion Larmor radius", units="m", dims=("surface",)),
    "PerSurface/w_ib" => (; long_name="ion banana width", units="psi_N", dims=("surface",)),
    "PerSurface/eps" => (; long_name="inverse aspect ratio of the surface", dims=("surface",)),
    "PerSurface/gamma" => (; long_name="linear tearing growth rate of the gate, NaN without one", units="1/s", dims=("surface",)),
    "PerSurface/delta_r" => (; long_name="resistive layer width of the gate, NaN without one", units="psi_N", dims=("surface",)),
    "PerSurface/w_lo" => (; long_name="lower end of the root search max(delta_r, w_min)", units="psi_N", dims=("surface",)),
    "PerSurface/w_m" => (; long_name="seed threshold: smallest F crossing from negative to positive, NaN if none", units="psi_N", dims=("surface",)),
    "PerSurface/w_sat" => (; long_name="saturated width: largest F crossing from positive to negative, NaN if none", units="psi_N",
        dims=("surface",)),
    "PerSurface/fully_stabilized" => (; long_name="1 if F < 0 on the whole search range", dims=("surface",)),
    "PerSurface/unstable_any_seed" => (; long_name="1 if F > 0 at w_lo", dims=("surface",)),
    "PerSurface/interchange_unstable" => (; long_name="1 if D_R > 0 (excluded)", dims=("surface",)),
    "PerSurface/linear_unstable" => (; long_name="1 if the gate's growth rate is positive (no seed threshold, w_sat kept)", dims=("surface",)),
    "PerSurface/below_w_ib" => (; long_name="1 if w_m < w_ib", dims=("surface",)),
    "PerSurface/large_delta_prime_w" => (; long_name="1 if |Delta'| w >= 1 at w_m or w_sat", dims=("surface",)),
    "PerSurface/complex_delta_prime" => (; long_name="1 if |Im Delta'| > 1e-3 |Re Delta'|", dims=("surface",)),
    "Terms/F" => (; long_name="modified Rutherford F(w), the sum of the switched-on terms", dims=_MRE_TERM_DIMS),
    "Terms/classical" => (; long_name="classical term Delta'(w)", dims=_MRE_TERM_DIMS),
    "Terms/bootstrap" => (; long_name="bootstrap term Delta_nc(w)", dims=_MRE_TERM_DIMS),
    "Terms/curvature" => (; long_name="curvature term Delta_GGJ(w)", dims=_MRE_TERM_DIMS),
    "Terms/polarization" => (; long_name="ion polarization term Delta_pol(w), including pol_sign", dims=_MRE_TERM_DIMS),
    "Terms/w_d" => (; long_name="transport threshold width w_d(w)", units="psi_N", dims=_MRE_TERM_DIMS)
]

"""
    save_to_hdf5(parent::Union{HDF5.File,HDF5.Group}, result::CallenResult)

Write `result` into `Tearing/Callen/` of `parent`, one dataset per field over the resonant rows, replacing any existing
one and keeping the rest of `Tearing/`, and record its deck section.
"""
function save_to_hdf5(parent::Union{HDF5.File,HDF5.Group}, result::CallenResult)
    tearing = haskey(parent, "Tearing") ? parent["Tearing"] : create_group(parent, "Tearing")
    g = replace_group!(tearing, "Callen")
    for (key, _) in CALLEN_H5_ANNOTATIONS
        g[key] = getfield(result, Symbol(key))
    end
    Utilities.HDF5Annotations.annotate!(g, CALLEN_H5_ANNOTATIONS)
    Utilities.record_inputs!(parent, result)
    return g
end

const CALLEN_H5_ANNOTATIONS = [
    "psi" => (; long_name="normalized poloidal flux of the rational surface", units="psi_N", dims=("surface",)),
    "q" => (; long_name="safety factor at the surface", dims=("surface",)),
    "m" => (; long_name="resonant poloidal mode number", dims=("surface",)),
    "n" => (; long_name="resonant toroidal mode number", dims=("surface",)),
    "rho_tor" => (; long_name="toroidal-flux radius (Phi_tor/(pi B0))^(1/2)", units="m", dims=("surface",)),
    "drho_dpsi" => (; long_name="d rho_tor / d psi_N", units="m", dims=("surface",)),
    "rho_i" => (; long_name="ion gyroradius in the axis field", units="m", dims=("surface",)),
    "w_pol" => (; long_name="polarization width 0.5 q rho_i/(rho/R0)^(1/2)", units="m", dims=("surface",)),
    "delta_rmp" => (; long_name="Callen Delta'_RMP from the total resonant field", units="1/m", dims=("surface",)),
    "delta_c" => (; long_name="Callen intrinsic Delta' in 1/m of rho: computed Delta'_mn, or -2m/rho with delta_prime_source = cylindrical", units="1/m", dims=("surface",)),
    "b_res" => (; long_name="total resonant field |flux|/area", units="T", dims=("surface",)),
    "b_vac" => (; long_name="vacuum (coil) resonant field |flux|/area", units="T", dims=("surface",)),
    "hw_isl" => (; long_name="total-field island half-width", units="psi_N", dims=("surface",)),
    "hw_v" => (; long_name="vacuum island half-width", units="psi_N", dims=("surface",)),
    "hw_v_crit" => (; long_name="critical vacuum island half-width", units="psi_N", dims=("surface",)),
    "hw_min" => (; long_name="island half-width at the smaller positive root of the Callen cubic, NaN if none", units="psi_N", dims=("surface",)),
    "hw_sat" => (; long_name="island half-width at the larger positive root of the Callen cubic, NaN if none", units="psi_N", dims=("surface",)),
    "n_roots" => (; long_name="number of positive roots of the Callen cubic", dims=("surface",)),
    "b_crit" => (; long_name="critical vacuum resonant field", units="T", dims=("surface",))
]
