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

println("\n=== Testing WITHOUT WALL ===")

JPEC.Vacuum.set_dcon_params(mtheta_eq, mlow, mhigh, 
                                    n, vac_inputs.qa, vac_inputs.r, 
                                    vac_inputs.z, vac_inputs.delta)
        
wv_fortran = JPEC.Vacuum.mscvac(wv_block, mpert, mtheta_eq, mthvac, 
                         complex_flag, kernelsign, false,  # wall_flag = false
                         farwall_flag, grri, xzpts, ahg_file, "$(@__DIR__)")

JPEC.Vacuum.unset_dcon_params()

# Now deepcopy the julia vac_inputs and set the new ms
vac_inputs_nowall = deepcopy(vac_inputs)
vac_inputs_nowall.mlow = mlow
vac_inputs_nowall.mhigh = mhigh
vac_inputs_nowall.mpert = mhigh - mlow + 1

# Create no-wall settings
nowall_settings = JPEC.Vacuum.WallShapeSettings(shape="nowall")

wv_julia, grri_j, xzpts_j = JPEC.Vacuum.compute_vacuum_response(vac_inputs_nowall, nowall_settings)

println("\nFortran output (NO WALL):")
display(wv_fortran)

println("\nJulia output (NO WALL):")
display(wv_julia)

println("\nAbs difference of Fortran and Julia outputs (NO WALL):")
diff_nowall = abs.(wv_fortran .- wv_julia)
display(diff_nowall)
println("\nMax difference (NO WALL): ", maximum(diff_nowall))
