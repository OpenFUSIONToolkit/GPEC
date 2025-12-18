using JLD2
using Pkg
println("Activating JPEC environment at $(@__DIR__)/../..")
Pkg.activate("$(@__DIR__)/../.."); using JPEC

@load "$(@__DIR__)/vacuum_response_inputs.jld2" benchmark_inputs

(; wv_block, mpert, mtheta_eq, mthvac, complex_flag, kernelsign,
                wall_flag, farwall_flag, grri, xzpts, ahg_file, dir_path,
                vac_inputs, wall_settings,
                n, ipert_n, psifac) = benchmark_inputs

# Lower mpert to be managable for testing
mlow = -2
mhigh = 2
mpert = mhigh-mlow+1

wv_block = zeros(ComplexF64, mpert, mpert)


JPEC.Vacuum.set_dcon_params(mtheta_eq, mlow, mhigh, 
                                    n, vac_inputs.qa, vac_inputs.r, 
                                    vac_inputs.z, vac_inputs.delta)
        
wv_fortran = JPEC.Vacuum.mscvac(wv_block, mpert, mtheta_eq, mthvac, 
                         complex_flag, kernelsign, wall_flag, 
                         farwall_flag, grri, xzpts, ahg_file, "$(@__DIR__)")

JPEC.Vacuum.unset_dcon_params()

# Now deepcopy the julia vac_inputs and set the new ms
vac_inputs = deepcopy(vac_inputs)
vac_inputs.mlow = mlow
vac_inputs.mhigh = mhigh
vac_inputs.mpert = mhigh - mlow + 1

wv_julia, grri, xzpts = JPEC.Vacuum.compute_vacuum_response(vac_inputs, wall_settings)

println("Fortran output:")
display(wv_fortran)

println("Julia output:")
display(wv_julia)

println("Abs difference of Fortran and Julia outputs:")
display(abs.(wv_fortran .- wv_julia))