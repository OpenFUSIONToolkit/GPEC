# Tests for JPEC IMAS integration (read_imas + write_imas)
#
# TODO: Create comprehensive tests for IMAS integration
#
# Test 1: Verify read_imas produces same result as read_efit
# Test 2: Verify write_imas correctly populates dd.mhd_linear
#
# Import required packages:
# - import EFIT
# - import EFIT.IMASdd

# TODO: Test 1 - IMAS equilibrium: read_imas matches read_efit
#
# Steps to implement:
# 1. Read test gEQDSK file with EFIT path (reference)
# 2. Convert same gEQDSK to IMAS dd using EFIT.geqdsk2imas!
# 3. Load via IMAS path with same grid settings
# 4. Compare key quantities (should match within ~1% tolerance):
#    - equil.ro (magnetic axis R)
#    - equil.zo (magnetic axis Z)
#    - equil.psio (normalizing flux)
#    - q profile
#    - p profile
#
# Test data location: joinpath(@__DIR__, "test_data", "regression_equilibrium_example", "EQDSK_COCOS_02")

# TODO: Test 2 - IMAS write: write_imas populates dd.mhd_linear
#
# Steps to implement:
# 1. Create empty dd with one equilibrium time slice (t=0.0)
# 2. Build minimal synthetic DCON result:
#    - DconControl with nn_low=1, nn_high=1, vac_flag=true
#    - DconInternal with n_modes=7 (m=-3 to m=3)
#    - VacuumData with synthetic eigenvalues (first one negative = unstable)
# 3. Call JPEC.DCON.write_imas(dd, result)
# 4. Verify dd.mhd_linear is correctly populated:
#    - ideal_flag == 1
#    - code.name == "JPEC"
#    - One time slice at t=0.0
#    - 7 toroidal_mode entries, all with n_tor==1
#    - First eigenmode has energy_perturbed < 0
#    - Energy values match real(vac_data.et[i])

println("\n" * "="^70)
println("IMAS integration tests need to be implemented!")
println("="^70)


import EFIT #read EFIT g-files and converts them to IMAS

import EFIT.IMASdd #because IMAS is a submodule of EFIT pack

@testset "IMAS equilbrium: read_imas mathces read_efit" begin #if any test of the following fails the failure will come from IMAS eq testset

  data_dir = joinpath(@__DIR__,"test data", "regression_equilibrium_example)
    geqdsk_path = joinpath(data_dir, "EQDSK_COCOS_02")


    
    
    efit_control = JPEC.Equilbrium.EquilbriumControl(;

    eq_type = "efit",

    eq_filename = geqdsk_path,

  jac_type = "boozer",

    psilow = 0.01, 

    psihigh = 0.994,
    )

    efit_config = JPEC.Equilibrium.EquilibriumConfig(efit_control, JPEC.Equilibrium.EquilibriumOutput)

    equil_efit = JPEC.Equilibrium.setup_equilibrium(efit_config)




    #Now we change the gEQDSK file to IMAS dd by EFIT.jl

    g = EFIT.readg(geqdsk_path; set_time = 0.0) #read g file in a EFIT.jl internal format

    dd = IMAS.dd() #create a new empty IMAS structure

    dd.global_time = g.time

    dd.equilbrium.time = [g.time]

    resize!(dd.equilibrium.time_slice, 1)

    dd.equilibrium.time_slice[1].time = g.time

    EFIT.geqdsk2imas!(g, dd.equilbrium.time_slice[1]; wall = nothing) #converted the g file to IMAS for the test

                                                                    #wall = nothing : didnt include the wall geometry 
                                                                    #fills 1D profiles and 2D profile(psi map)
                                                                    #Conversion g file COCOS2 -> IMAS COCOS11

    #now we load IMAS path

    imas_control = JPEC.Equilibrium.EquilibriumControl(; 

    eq_type = "imas", #JPEC reads from IMAS

    eq_filename = joipath(data_dir, "imas_placeholder"), #filename cause eq_filename directory is used for diagnostic output files, but the actual equilibrium data is from a dd file

    jac_type = "boozer"

    psilow = 0.01

    psihigh = 0.994, #same settings, bith must match efit_control for comparison
    )

    imas_config = JPEC.Equilibrium.EquilibriumConfig(imas_control, JPEC.Equilibrium.EquilibriumOutput()) #imas_config from imas_control plus JPEC.Equilibrium.EquilibriumOutput()
    equil_imas = JPEC.Equilibrium.setup_equilibrium(imas_config, dd) #fills the ouput values 
                                                                    # I used function setup_equilibrium(...) to do the phyiscs
    @test equil_imas isa JPEC.Equilibrium.PlasmaEquilibrium
    #testing if read_imas() returned the right type.

    #Both results come from the same physical data, so differences are only
    # from COCOS conversion (EFIT.jl) and our IMAS reader interpolation.
    # Tolerances are set at ~0.1% for scalar quantities and ~1% for profiles.

    @test isapprox(equil_efit.ro, equil_imas.ro, rtol = 1e-3)

    # ryol is relative tolerance, as values must match with 0.1 tolerance.
    # we tested ro = major radius of magnetic axis in m.

    @test isaprox(equil_efit.psio, equil_imas.psio, rtol = 1e-3)

    # psio = total flux swing 

  # Normalising flux |psi_boundary - psi_axis|

    @test isaprox(equil_efit.profiles.q_spline.y, equil_imas.profiles.q_spline.ym rtol = 1e-2)

    # testing the q values of cubic spline(all grid point values, not the spline function itself)

    @test isapprox(equil_efit.profiles.P_spline.y, equil_imas.profiles.P_spline.y, rtol = 1e-2)

    #check pressure profiles

    # done the first test we sucessfully checked that read_imas matches read_efit

    
# Test 2: see if write_imas populates correctly dd.mhd_linear

    #Creates fake DCON results without running the full solver, verify IMAS struct is filled correctly

    @testset "IMAS write: write_imas pupulates dd.mhd_linear" begin

    dd_out = IMASdd.dd()

    dd_out.equilibrium.time = [0.0]

    resize!(dd_out.equilibrium.time_slice, 1)

    dd_out.equilibrium.time_slice[1].time = 0.0

    n_modes = 7


    #building the control panel for DCON
    
    ctrl = JPEC.DCON.DconControl(;

    nn_low = 1,

    nn_high = 1, #keeping it minimal for the test

    vac_flag = true,

    verbose = false, #keep output clean

    )

    intr = JPEC.DCON.DconInternal(;

    nlow = 1,

    nhigh  =1,
    npert = 1,

    mlow = -3,

    mhigh = 3,

    mpert = n_modes

    numpert_total = n_modes,
    ) # set the DconInternal

    vac_data = JPEC.DCON.VacuumData(10, n_modes, n_modes)


    vac_data.et .= ComplexF64[-0.3, 0.1,0.2,0.4,0.6,0.8,1,2] #et[1] < 0 unstable as -0.3<0 
    

    #fake eigenvalues but kept the strucutre, fast and simple
    # as testing if write_imas is stores correctly not that the phyiscs behin is right.


    fake_result = (ctrl = ctrl, equil = nothing, intr = intr, ffit = nothing, odet = nothing, vac_data = vac_data)

    #fake_result - NamedTuple that mimick DCON output

    #write_imas needs ctrl, intr, vac_data fields

    # the rest are unused by Imas so I set them to nothing 

    #Also worth to note that this is the format a Main() is returining

    JPEC.DCON.write_imas(dd_out, fake_result) #the function I am testing in action 

    mhd = dd_out.mhd_linear #contains stabilty results

    #verify IMAS metadata
    @test mhd.ideal_flag ==1
    @test mhd.ids_properties.homogenous_time ==1
    @test mhd.code.name == "JPEC"

    @test length(mhd.time_slice) == 1 #single time slice so 1
    @test mhd.time_slice[1].time == 0.0# from dd_out.equilibrium

    @test length(mhd.time_slice[1].toroidal_mode) == n_modes # 7 array - one per eigenmode

    for tm in mhd.time_slice[1].toroidal_mode

    @test tm.n_tor == 1

  end

  #each toroidal mode should have n = 1 since this is the minimal condition I put n_high =1 and n_low = 1

    @test mhd.time_slice[1].toroidal_mode[1].energy_perturbed < 0

    #in case of first eigenmode, unstable energy perturbed should be negative to check if IMAS indentifies unstable modes

    for i in 1:n_modes

    @test isapprox(mhd.time_slice[1].toroidal_mode[i].energy_perturbed

    real(vac_data.et[i]); atol = 1e-10)

  end

    #I checked here if the energy values will match the real parts of the et

    #finish line of the testing: 1.Test read_imas matches read_efit 4 @test
                                2.Test if IMAS populates correctly dd.mhd_linear 8 @test
    

    
    

    

    

    
    

    

    


            
