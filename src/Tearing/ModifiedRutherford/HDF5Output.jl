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
    # One dataset per annotated field; flags as 0/1 integers, readable without an HDF5 boolean type.
    for (key, _) in MRE_H5_ANNOTATIONS
        value = getfield(result, Symbol(basename(key)))
        g[key] = value isa AbstractVector{Bool} ? Int8.(value) : value
    end
    Utilities.HDF5Annotations.annotate!(g, MRE_H5_ANNOTATIONS)
    Utilities.record_inputs!(parent, result)
    return g
end

# Julia's surface × width matrices land in HDF5 as (width, surface).
const _MRE_TERM_DIMS = ("w", "surface")
const _MRE_BRANCH_DIMS = ("branch", "surface")

const MRE_H5_ANNOTATIONS = [
    "w" => (; long_name="full island width grid", units="psi_N", dims=("w",)),
    "w_m_star" => (; long_name="smallest seed-threshold full width over the surfaces, NaN if none", units="psi_N"),
    "b_m_star" => (; long_name="smallest b_m over the surfaces, NaN if none", units="T"),
    "PerSurface/rational_psi" => (; long_name="normalized poloidal flux of the rational surface", units="psi_N", dims=("surface",)),
    "PerSurface/rational_q" => (; long_name="safety factor at the surface", dims=("surface",)),
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
    "PerSurface/D_c" => (; long_name="SLAYER toroidal critical offset used by curvature_model = lutjens (r_s reference), NaN otherwise",
        dims=("surface",)),
    "PerSurface/W_d_layer" => (; long_name="transport width sqrt(2) pi^1.5 (-D_R)/D_c in the (r - r_s)/r_s reference, NaN otherwise",
        dims=("surface",)),
    "PerSurface/alpha_s" =>
        (; long_name="saturation current-gradient coefficient alpha_s = -r q J_z'/s (Fitzpatrick 2023 Eq. 9.16), J_z = (R0^2/B0) <mu0 j_phi/R>", dims=("surface",)),
    "PerSurface/beta_s" => (; long_name="saturation current-curvature coefficient beta_s = -r^2 q J_z''/s", dims=("surface",)),
    "PerSurface/C_sat" => (; long_name="saturation coefficient 0.8 alpha_s^2 - 0.27 beta_s - 0.09 alpha_s (Fitzpatrick 2023 Eq. 9.15)",
        dims=("surface",)),
    "PerSurface/w_m" => (; long_name="seed threshold: smallest F crossing from negative to positive, NaN if none", units="psi_N", dims=("surface",)),
    "PerSurface/w_sat" => (; long_name="saturated width: first falling F crossing above the island start (w_lo or w_m), NaN if none", units="psi_N",
        dims=("surface",)),
    "PerSurface/fully_stabilized" => (; long_name="1 if F < 0 on the whole search range", dims=("surface",)),
    "PerSurface/unstable_any_seed" => (; long_name="1 if F > 0 at w_lo", dims=("surface",)),
    "PerSurface/unsaturated" => (; long_name="1 if F > 0 at w_max: no saturation in the search range", dims=("surface",)),
    "PerSurface/outside_plasma" => (; long_name="1 if psi +- w_sat/2 leaves the equilibrium psi grid", dims=("surface",)),
    "PerSurface/gate_disagrees" => (; long_name="1 if F(w_lo) > 0 with gamma < 0, or F(w_lo) < 0 with gamma > 0", dims=("surface",)),
    "PerSurface/interchange_unstable" => (; long_name="1 if D_R > 0 (excluded)", dims=("surface",)),
    "PerSurface/linear_unstable" => (; long_name="1 if the gate's growth rate is positive (no seed threshold, w_sat kept)", dims=("surface",)),
    "PerSurface/below_w_ib" => (; long_name="1 if w_m < w_ib", dims=("surface",)),
    "PerSurface/large_delta_prime_w" => (; long_name="1 if |Delta'(w)| w >= 1 at w_m or w_sat, Delta'(w) the classical term", dims=("surface",)),
    "PerSurface/saturation_unconverged" => (; long_name="1 if C_sat moves more than 20% between half and double the fit window, or the window leaves the grid",
        dims=("surface",)),
    "PerSurface/w_sat_all" => (; long_name="every saturated width (falling F crossing) in increasing order, NaN-padded; w_sat is the first",
        units="psi_N", dims=_MRE_BRANCH_DIMS),
    "PerSurface/w_seed_all" => (; long_name="seed threshold reaching each w_sat_all entry (rising crossing below it), NaN if F > 0 from w_lo",
        units="psi_N", dims=_MRE_BRANCH_DIMS),
    "PerSurface/w_seed_unsaturated" => (; long_name="seed above which the island grows past w_max (F > 0 there), NaN otherwise", units="psi_N",
        dims=("surface",)),
    "PerSurface/b_m" => (; long_name="resonant area-weighted field whose vacuum island has full width w_m, B = pi |n q'| chi1 w^2/(8 A)",
        units="T", dims=("surface",)),
    "PerSurface/b_seed_all" => (; long_name="resonant area-weighted field whose vacuum island is as wide as each w_seed_all entry", units="T",
        dims=_MRE_BRANCH_DIMS),
    "PerSurface/b_seed_unsaturated" => (; long_name="resonant area-weighted field whose vacuum island is as wide as w_seed_unsaturated",
        units="T", dims=("surface",)),
    "PerSurface/n_sat" => (; long_name="number of saturated widths", dims=("surface",)),
    "PerSurface/multiple_saturated" => (; long_name="1 if more than one saturated width (see w_sat_all)", dims=("surface",)),
    "PerSurface/complex_delta_prime" => (; long_name="1 if |Im Delta'| > 1e-3 |Re Delta'|", dims=("surface",)),
    "Terms/F" => (; long_name="modified Rutherford F(w), the sum of the switched-on terms", dims=_MRE_TERM_DIMS),
    "Terms/classical" => (; long_name="classical term Delta'(w)", dims=_MRE_TERM_DIMS),
    "Terms/saturation" => (; long_name="finite-width saturation term Delta_sat(w)", dims=_MRE_TERM_DIMS),
    "Terms/bootstrap" => (; long_name="bootstrap term Delta_nc(w)", dims=_MRE_TERM_DIMS),
    "Terms/curvature" => (; long_name="curvature term Delta_GGJ(w) of curvature_model", dims=_MRE_TERM_DIMS),
    "Terms/polarization" => (; long_name="ion polarization term Delta_pol(w), including pol_sign", dims=_MRE_TERM_DIMS),
    "Terms/w_d" => (; long_name="transport threshold width w_d(w)", units="psi_N", dims=_MRE_TERM_DIMS)
]
