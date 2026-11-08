# HDF5Output.jl
#
# Writer for a `CallenResult`, under the caller's parent group:
#
#   Tearing/
#   └── Callen/
#       └── PerSurface/            -- one dataset per field over the resonant rows

"""
    save_to_hdf5(parent::Union{HDF5.File,HDF5.Group}, result::CallenResult)

Write `result` into `Tearing/Callen/PerSurface/` of `parent`, one dataset per field over the resonant rows, replacing any existing
one and keeping the rest of `Tearing/`, and record its deck section.
"""
function save_to_hdf5(parent::Union{HDF5.File,HDF5.Group}, result::CallenResult)
    tearing = haskey(parent, "Tearing") ? parent["Tearing"] : create_group(parent, "Tearing")
    g = replace_group!(tearing, "Callen")
    for (key, _) in CALLEN_H5_ANNOTATIONS
        g[key] = getfield(result, Symbol(basename(key)))
    end
    Utilities.HDF5Annotations.annotate!(g, CALLEN_H5_ANNOTATIONS)
    Utilities.record_inputs!(parent, result)
    return g
end

const CALLEN_H5_ANNOTATIONS = [
    "PerSurface/rational_psi" => (; long_name="normalized poloidal flux of the rational surface", units="psi_N", dims=("surface",)),
    "PerSurface/rational_q" => (; long_name="safety factor at the surface", dims=("surface",)),
    "PerSurface/m" => (; long_name="resonant poloidal mode number", dims=("surface",)),
    "PerSurface/n" => (; long_name="resonant toroidal mode number", dims=("surface",)),
    "PerSurface/rho_tor" => (; long_name="toroidal-flux radius (Phi_tor/(pi B0))^(1/2)", units="m", dims=("surface",)),
    "PerSurface/drho_dpsi" => (; long_name="d rho_tor / d psi_N", units="m", dims=("surface",)),
    "PerSurface/rho_i" => (; long_name="ion gyroradius in the axis field", units="m", dims=("surface",)),
    "PerSurface/w_pol" => (; long_name="polarization width 0.5 q rho_i/(rho/R0)^(1/2)", units="m", dims=("surface",)),
    "PerSurface/delta_rmp" => (; long_name="Callen Delta'_RMP from the total resonant field", units="1/m", dims=("surface",)),
    "PerSurface/delta_c" =>
        (; long_name="Callen intrinsic Delta' in 1/m of rho: computed Delta'_mn, or -2m/rho with delta_prime_source = cylindrical", units="1/m", dims=("surface",)),
    "PerSurface/b_res" => (; long_name="total resonant field |flux|/area", units="T", dims=("surface",)),
    "PerSurface/b_vac" => (; long_name="vacuum (coil) resonant field |flux|/area", units="T", dims=("surface",)),
    "PerSurface/hw_isl" => (; long_name="total-field island half-width", units="psi_N", dims=("surface",)),
    "PerSurface/hw_v" => (; long_name="vacuum island half-width", units="psi_N", dims=("surface",)),
    "PerSurface/hw_v_crit" => (; long_name="critical vacuum island half-width", units="psi_N", dims=("surface",)),
    "PerSurface/hw_min" => (; long_name="island half-width at the smaller positive root of the Callen cubic, NaN if none", units="psi_N", dims=("surface",)),
    "PerSurface/hw_sat" => (; long_name="island half-width at the larger positive root of the Callen cubic, NaN if none", units="psi_N", dims=("surface",)),
    "PerSurface/n_roots" => (; long_name="number of positive roots of the Callen cubic", dims=("surface",)),
    "PerSurface/b_crit" => (; long_name="critical vacuum resonant field", units="T", dims=("surface",))
]
