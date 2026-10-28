import ..Utilities: save_to_hdf5
import ..Utilities: EQUIL_H5_NAMES, EQUIL_H5_SKIP, MAIN_H5_ANNOTATIONS
import ..Utilities.HDF5Annotations: replace_group!
using HDF5

"""
    save_to_hdf5(target, equil::PlasmaEquilibrium)

Save the processed equilibrium profiles, geometry, and derived scalar quantities.
"""
function save_to_hdf5(out_h5::Union{HDF5.File,HDF5.Group}, equil::PlasmaEquilibrium)
    replace_group!(out_h5, "Equilibrium")
    for f in fieldnames(EquilibriumParameters)
        f in EQUIL_H5_SKIP && continue
        val = getfield(equil.params, f)
        val === nothing && continue
        out_h5["Equilibrium/$(get(EQUIL_H5_NAMES, f, String(f)))"] = val
    end
    out_h5["Equilibrium/psi_total"] = equil.psio
    out_h5["Equilibrium/R_axis"] = equil.ro
    out_h5["Equilibrium/Z_axis"] = equil.zo

    # Write equilibrium profile and geometry arrays (from the named splines)
    profiles = equil.profiles
    out_h5["Equilibrium/Profiles/psi"] = profiles.xs
    out_h5["Equilibrium/Profiles/2piF"] = profiles.F_spline.y
    out_h5["Equilibrium/Profiles/mu0p"] = profiles.P_spline.y
    out_h5["Equilibrium/Profiles/dVdpsi"] = profiles.dVdpsi_spline.y
    out_h5["Equilibrium/Profiles/q"] = profiles.q_spline.y
    out_h5["Equilibrium/Geometry/psi"] = equil.rzphi_xs
    out_h5["Equilibrium/Geometry/theta"] = equil.rzphi_ys
    # Extract grid point values from interpolants for HDF5 output
    out_h5["Equilibrium/Geometry/rcoords"] = equil.rzphi_rsquared.nodal_derivs.partials[1, :, :]
    out_h5["Equilibrium/Geometry/offset"] = equil.rzphi_offset.nodal_derivs.partials[1, :, :]
    out_h5["Equilibrium/Geometry/nu"] = equil.rzphi_nu.nodal_derivs.partials[1, :, :]
    out_h5["Equilibrium/Geometry/jac"] = equil.rzphi_jac.nodal_derivs.partials[1, :, :]

    Utilities.HDF5Annotations.annotate!(out_h5, MAIN_H5_ANNOTATIONS)
    return nothing
end
