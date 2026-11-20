# The coil-forced TJ run the error-field tests share: three PF hoops outside the plasma (R ≈ 9–11 m), one tilted so
# the run has a nominal n=1 forcing, one axisymmetric so its rigid-motion response is known exactly, and one tilted the
# other way as the correction array, which is swept but is not an error-field source. Copies the deck into `dir`, runs
# it and returns main's results, the gpec.h5 path and the deck as run.
function error_fields_fixture(dir)
    GPEC = GeneralizedPerturbedEquilibrium
    template = joinpath(@__DIR__, "test_data", "regression_tj_ideal_example")
    for name in readdir(template)
        cp(joinpath(template, name), joinpath(dir, name))
    end
    toml_path = joinpath(dir, "gpec.toml")
    inputs = TOML.parsefile(toml_path)
    inputs["ForcingTerms"] = Dict{String,Any}(
        "forcing_data_format" => "coil", "mtheta_coil" => 240, "nzeta_coil" => 32,
        "coil_set" => [
            Dict{String,Any}("name" => "hoop_tilted", "source" => "pf_hoop", "radius" => 12.0, "height" => 1.2, "currents" => [2.0e3], "tiltx" => [3.0]),
            Dict{String,Any}("name" => "hoop_axi", "source" => "pf_hoop", "radius" => 12.0, "height" => -1.2, "currents" => [2.0e3]),
            Dict{String,Any}("name" => "hoop_efc", "source" => "pf_hoop", "radius" => 12.0, "height" => 0.0, "currents" => [2.0e3], "tilty" => [2.0])
        ])
    inputs["PerturbedEquilibrium"] = Dict{String,Any}(
        "compute_singular_coupling" => true)
    cp(joinpath(@__DIR__, "test_data", "ErrorFields", "tolerances_two_hoops.toml"), joinpath(dir, "tolerances.toml"))
    # Kinetic profiles consistent with this pressure, for the NTV torque couplings.
    cp(joinpath(@__DIR__, "..", "examples", "TJ_kinetic_NTV_example", "kinetic.dat"), joinpath(dir, "kinetic.dat"))
    inputs["Equilibrium"]["kinetic_file"] = "kinetic.dat"
    # Exercise the kinetic pipeline at smoke-test precision, retaining all bounce harmonics.
    inputs["KineticForces"] = Dict{String,Any}(
        "rtol_psi" => 0.2, "rtol_pitch" => 1e-2, "atol_pitch" => 1e-5,
        "rtol_x" => 1e-3, "atol_x" => 1e-6)
    # Small batches exercise orchestration; distribution accuracy is tested on synthetic inputs.
    inputs["ErrorFields"] = Dict{String,Any}("tolerance_file" => "tolerances.toml",
        "MonteCarlo" => Dict{String,Any}("nsample" => 512, "nbatch" => 2, "seed" => 5, "nbins" => 16),
        "scenario" => Dict{String,Any}("n_e" => 12.0),
        "Risk" => Dict{String,Any}("nsample_threshold" => 512, "seed" => 3, "scan_scales" => [0.5, 1.0, 2.0]),
        # The smallest rotation scan that exercises the table path (three shifts, no refinement).
        "NTV" => Dict{String,Any}("efc_coils" => ["hoop_efc"], "rotation_scan_points" => 3, "rotation_scan_max_points" => 3))
    open(io -> TOML.print(io, inputs), toml_path, "w")
    return GPEC.main([dir]), joinpath(dir, "gpec.h5"), inputs
end
