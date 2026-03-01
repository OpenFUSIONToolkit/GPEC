"""
    write_imas(dd::IMASdd.dd, result::NamedTuple) -> IMASdd.dd

Write JPEC/DCON linear stability results into the `mhd_linear` IDS of an IMAS data
dictionary `dd`.

TODO: Implement this function to write stability results to dd.mhd_linear

## What to implement:

1. Extract variables from result:
   - ctrl = result.ctrl
   - intr = result.intr
   - vac_data = result.vac_data

2. Get time from dd.equilibrium.time_slice[1].time

3. Set top-level mhd_linear metadata:
   - mhd.ids_properties.comment = "JPEC DCON ideal MHD linear stability, n = $(intr.nlow):$(intr.nhigh)"
   - mhd.ids_properties.homogeneous_time = 1
   - mhd.ideal_flag = 1
   - mhd.code.name = "JPEC"
   - mhd.time = [t]

4. Create ONE time slice (IMAS-only version - single slice):
   - resize!(mhd.time_slice, 1)
   - Set ts.time = t

5. Create toroidal_mode entries (one per eigenmode):
   - resize!(ts.toroidal_mode, intr.numpert_total)
   - For each eigenmode i:
     * Calculate n_tor from block position: ipert_n = (i-1) ÷ intr.mpert + 1
     * Set tm.n_tor = intr.nlow + ipert_n - 1
     * Set tm.energy_perturbed = real(vac_data.et[i]) if vac_flag is true

6. Return dd
"""
function write_imas(dd::IMASdd.dd, result::NamedTuple)

    # TODO: Your code here!
    # Extract ctrl, intr, vac_data from result

    # TODO: Get time from dd.equilibrium.time_slice[1].time

    # TODO: Get mhd = dd.mhd_linear

    # TODO: Set all the metadata fields listed above

    # TODO: Create the time slice and resize to 1

    # TODO: Loop over eigenmodes and fill toroidal_mode entries

    # TODO: Return dd

end
