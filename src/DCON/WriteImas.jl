"""
    write_imas(dd::IMASdd.dd, result::NamedTuple) -> IMASdd.dd

Write JPEC/DCON linear stability results into the `mhd_linear` IDS of an IMAS data
dictionary `dd`.

"""
function write_imas(dd::IMASdd.dd, result::NamedTuple)

    ctrl = result.ctrl # extracts DconControl struct run parameters(nn_low, vac_flag)
    intr = result.intr #Dcon internal data
    vac_data = result.vac_data #vacuum caluclations
                               #also contains eigenvalue array et-- important for stability info

t = dd.equilibrium.time_slice[1].time # variable to hold time valuse, IMAS time dependent data
mhd = dd.mhd_linear

    mhd.ids_properties.comment = "JPEC DCON ideal linear stability, n = $(intr.nlow):$(intr.nhigh)"

    mhd.ids_properties.homogeneous_time = 1 # true: that all the data in this IDS corresponds to a single value
                                           #this is also a data consistency check

    mhd.ideal_flag = 1 #as DCON is an ideal MHD code

    mhd.code.name = "JPEC"

    mhd.time = [t]

    resize!(mhd.time_slice,1)

    ts = mhd.time_slice[1]

    ts.time = t

    resize!(ts.toroidal_mode, intr.numpert_total)# resizing the ts.toroidal_mode vector to the exact number of eigenmodes

for i in 1:intr.numpert_total

        tm = ts.toroidal_mode[i]

        ipert_n = (i-1) ÷ intr.mpert + 1

        tm.n_tor = intr.nlow + ipert_n - 1 #tm.n toroidal mode number

        if ctrl.vac_flag && vac_data !==nothing #

            tm.energy_perturbed = real(vac_data.et[i])#checking the sign of the energy
                                                      #if the energy is negative then unstable, if positive stable

        end

    end

        return dd #dd .mhd_linear now is populated with results



    end

#important notes: dd.mhd_linear now has stabilty results, the list of eigenvalues mode numbers and metadata





