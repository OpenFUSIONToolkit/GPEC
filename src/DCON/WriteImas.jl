"""
    write_imas(dd::IMASdd.dd, result::NamedTuple) -> IMASdd.dd

Write JPEC ideal linear stability results into the `mhd_linear` IDS of an IMAS
data dictionary `dd`.
"""
function write_imas(dd::IMASdd.dd, result::NamedTuple)
    ctrl = result.ctrl       # run parameters (nn_low, nn_high, vac_flag)
    intr = result.intr       # internal data (mode numbers, counts)
    vac_data = result.vac_data  # vacuum calculations (eigenvalue array et)

    t = dd.equilibrium.time_slice[1].time
    mhd = dd.mhd_linear

    mhd.ids_properties.comment = "JPEC ideal linear stability, n = $(intr.nlow):$(intr.nhigh)"
    mhd.ids_properties.homogeneous_time = 1  # all data corresponds to a single time value
    mhd.ideal_flag = 1  # ideal MHD
    mhd.code.name = "JPEC"
    mhd.time = [t]

    resize!(mhd.time_slice, 1)
    ts = mhd.time_slice[1]
    ts.time = t

    resize!(ts.toroidal_mode, intr.numpert_total)
    for i in 1:intr.numpert_total
        tm = ts.toroidal_mode[i]
        ipert_n = (i - 1) ÷ intr.mpert + 1
        tm.n_tor = intr.nlow + ipert_n - 1

        if ctrl.vac_flag && vac_data !== nothing
            tm.energy_perturbed = real(vac_data.et[i])
        end
    end

    return dd
end
