import ..Utilities: save_to_hdf5
import ..Utilities.HDF5Annotations: replace_group!, write_fields!, annotate!
import ..Utilities: MAIN_H5_ANNOTATIONS
using HDF5

"""
    save_to_hdf5(target, result::LocalStabilityResult)

Save the local-stability profiles and ballooning boundary scan.
"""
function save_to_hdf5(target::Union{HDF5.File,HDF5.Group}, result::LocalStabilityResult)
    group = replace_group!(target, "LocalStability")
    write_fields!(group, result, (:psi, :D_I, :D_R, :ballooning_Delta_prime, :ballooning_psi, :alpha, :alpha_critical))
    annotate!(target, MAIN_H5_ANNOTATIONS)
    return nothing
end
