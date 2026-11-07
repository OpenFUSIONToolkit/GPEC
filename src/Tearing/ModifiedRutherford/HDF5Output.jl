# HDF5Output.jl
#
# Writer for a `ModifiedRutherfordResult`, under the caller's parent group:
#
#   Tearing/
#   └── ModifiedRutherford/
#       ├── w                      -- width grid (ψ_N)
#       ├── PerSurface/            -- surface identity, indices and term inputs
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
    for key in _MRE_PER_SURFACE
        g["PerSurface/$key"] = getfield(result, key)
    end
    for key in _MRE_TERMS
        g["Terms/$key"] = getfield(result, key)
    end
    Utilities.HDF5Annotations.annotate!(g, MRE_H5_ANNOTATIONS)
    Utilities.record_inputs!(parent, result)
    return g
end

const _MRE_PER_SURFACE = (:psi, :q, :m, :n, :delta_prime, :D_I, :D_R, :H, :D_nc, :f_trap, :mu_ratio, :zeff, :Wc_prefac, :chi_perp,
    :v_te, :R0, :omega_star_i, :omega_hat, :omega_hat_pi, :beta_p, :rho_theta, :w_ib, :eps)
const _MRE_TERMS = (:F, :classical, :bootstrap, :curvature, :polarization, :w_d)

# Julia's surface × width matrices land in HDF5 as (width, surface).
const _MRE_TERM_DIMS = ("w", "surface")

const MRE_H5_ANNOTATIONS = [
    "w" => (; long_name="full island width grid", units="psi_N", dims=("w",)),
    "PerSurface/psi" => (; long_name="normalized poloidal flux of the rational surface", units="psi_N", dims=("surface",)),
    "PerSurface/q" => (; long_name="safety factor at the surface", dims=("surface",)),
    "PerSurface/m" => (; long_name="resonant poloidal mode number", dims=("surface",)),
    "PerSurface/n" => (; long_name="resonant toroidal mode number", dims=("surface",)),
    "PerSurface/delta_prime" => (; long_name="real part of the diagonal Delta' (PEST-3 convention)", dims=("surface",)),
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
    "PerSurface/omega_hat" => (; long_name="island frequency normalized to the ion diamagnetic frequency", dims=("surface",)),
    "PerSurface/omega_hat_pi" => (; long_name="omega_*pi/omega_*i = 1 + eta_i", dims=("surface",)),
    "PerSurface/beta_p" => (; long_name="poloidal ion beta 2 mu0 n_i T_i/<B_pol^2>", dims=("surface",)),
    "PerSurface/rho_theta" => (; long_name="poloidal ion Larmor radius", units="m", dims=("surface",)),
    "PerSurface/w_ib" => (; long_name="ion banana width", units="psi_N", dims=("surface",)),
    "PerSurface/eps" => (; long_name="inverse aspect ratio of the surface", dims=("surface",)),
    "Terms/F" => (; long_name="modified Rutherford F(w), the sum of the switched-on terms", dims=_MRE_TERM_DIMS),
    "Terms/classical" => (; long_name="classical term Delta'(w)", dims=_MRE_TERM_DIMS),
    "Terms/bootstrap" => (; long_name="bootstrap term Delta_nc(w)", dims=_MRE_TERM_DIMS),
    "Terms/curvature" => (; long_name="curvature term Delta_GGJ(w)", dims=_MRE_TERM_DIMS),
    "Terms/polarization" => (; long_name="ion polarization term Delta_pol(w), including pol_sign", dims=_MRE_TERM_DIMS),
    "Terms/w_d" => (; long_name="transport threshold width w_d(w)", units="psi_N", dims=_MRE_TERM_DIMS)
]
