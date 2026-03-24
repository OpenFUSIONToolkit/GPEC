"""
Julia vs Fortran GPEC coil normal field comparison.

Runs the coil Biot-Savart pipeline on the DIII-D shot 147131 case using
parameters matching the Fortran GPEC run exactly (psihigh, dmlim, coil currents,
grid resolution), then compares the mode spectrum against the Fortran output
in gpec_control_output_n1.nc.

Usage:
    julia --project=. benchmarks/check_coil_vs_fortran.jl
"""

using GeneralizedPerturbedEquilibrium
using GeneralizedPerturbedEquilibrium.ForcingTerms
using GeneralizedPerturbedEquilibrium.Equilibrium
using GeneralizedPerturbedEquilibrium.Analysis.CoilForcing
using NCDatasets
using Plots
using Printf

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FORTRAN_DIR = expanduser("~/Code/gpec/docs/examples/DIIID_ideal_example")
OUTPUT_DIR  = joinpath(@__DIR__, "coil_vs_fortran")
mkpath(OUTPUT_DIR)

isdir(FORTRAN_DIR) || error("Fortran GPEC example not found at $FORTRAN_DIR")

# ---------------------------------------------------------------------------
# Parameters matching Fortran run exactly
# ---------------------------------------------------------------------------

N_TOROIDAL = 1
MLOW       = -12    # mpert=34 from gpec_control_output_n1.nc global attributes
MHIGH      = 21
MTHETA     = 480    # cmtheta from coil.in
NZETA      = 40     # cmzeta from coil.in

# Currents from coil.in (in Amperes)
CURRENTS_C  = [982.0,  656.0, -326.0, -982.0, -656.0,  326.0]
CURRENTS_IL = [1000.0, 500.0, -500.0, -1000.0, -500.0, 500.0]
CURRENTS_IU = [1000.0, 500.0, -500.0, -1000.0, -500.0, 500.0]

println("="^70)
println("Julia vs Fortran GPEC Coil Normal Field Comparison")
println("="^70)

# ---------------------------------------------------------------------------
# Load equilibrium matching Fortran equil.in (psihigh=0.993, psilow=1e-4)
# ---------------------------------------------------------------------------

println("\n[1/4] Loading equilibrium (g147131.02300_DIIID_KEFIT, psihigh=0.993) ...")
t_equil = @elapsed begin
    eq_dict = Dict{String,Any}(
        "eq_type"      => "efit",
        "eq_filename"  => "g147131.02300_DIIID_KEFIT",
        "jac_type"     => "hamada",
        "psilow"       => 1e-4,
        "psihigh"      => 0.993,
        "mtheta"       => 256,
        "psi_accuracy" => 0.001,
        "grid_type"    => "log_asymptotic",
        "etol"         => 1e-7,
    )
    eq_config = Equilibrium.EquilibriumConfig(eq_dict, FORTRAN_DIR)
    equil = Equilibrium.setup_equilibrium(eq_config)
end
@printf "    Done in %.1f s\n" t_equil

# ---------------------------------------------------------------------------
# Build coil sets matching coil.in
# ---------------------------------------------------------------------------

println("\n[2/4] Loading coil geometry (d3d c + il + iu) ...")

coil_sets_cfg = [
    CoilSetConfig(; name="c",  currents=CURRENTS_C),
    CoilSetConfig(; name="il", currents=CURRENTS_IL),
    CoilSetConfig(; name="iu", currents=CURRENTS_IU),
]
COIL_DAT_DIR = joinpath(@__DIR__, "..", "src", "ForcingTerms", "coil_geometries")
cfg = CoilConfig(; machine="d3d", dat_dir=COIL_DAT_DIR, mtheta_coil=MTHETA, nzeta_coil=NZETA,
                   coil_sets=coil_sets_cfg)
coil_sets = load_coil_sets(cfg, N_TOROIDAL)

for cs in coil_sets
    @printf "    %-8s %d conductors, currents: %s A\n" cs.name cs.ncoil string(round.(cs.currents; digits=0))
end

# ---------------------------------------------------------------------------
# Julia: Biot-Savart → normal field → Fourier decomposition
# ---------------------------------------------------------------------------

println("\n[3/4] Computing Julia coil forcing modes (n=$N_TOROIDAL, m=$MLOW:$MHIGH, $(MTHETA)×$(NZETA)) ...")

julia_modes = ForcingMode[]
t_biot = @elapsed begin
    bnd_grid = sample_boundary_grid(equil, MTHETA, NZETA)
    nobs = MTHETA * NZETA
    obs_R   = zeros(nobs); obs_phi = zeros(nobs); obs_Z = zeros(nobs)
    for j in 1:NZETA, i in 1:MTHETA
        idx = i + (j-1)*MTHETA
        obs_R[idx]   = bnd_grid.R[i]
        obs_phi[idx] = bnd_grid.phi_grid[j]
        obs_Z[idx]   = bnd_grid.Z[i]
    end
    B_R = zeros(nobs); B_phi = zeros(nobs); B_Z = zeros(nobs)
    compute_biot_savart_boundary!(B_R, B_phi, B_Z, obs_R, obs_phi, obs_Z, coil_sets)
    bn = zeros(MTHETA, NZETA)
    project_normal_field!(bn, B_R, B_Z, bnd_grid)
    append!(julia_modes, fourier_decompose_bn(bn, bnd_grid, N_TOROIDAL, MLOW, MHIGH))
end
@printf "    Done in %.1f s,  max|bn| = %.3e T\n" t_biot maximum(abs, bn)

julia_m    = [md.m for md in sort(julia_modes; by=md -> md.m)]
julia_amps = abs.([md.amplitude for md in sort(julia_modes; by=md -> md.m)])

# ---------------------------------------------------------------------------
# Fortran: read b_n_x from gpec_control_output_n1.nc
# ---------------------------------------------------------------------------

println("\n[4/4] Reading Fortran output (gpec_control_output_n1.nc) ...")

nc_path = joinpath(FORTRAN_DIR, "gpec_control_output_n1.nc")
isfile(nc_path) || error("NetCDF file not found: $nc_path")

fortran_m    = Int[]
fortran_amps = Float64[]
NCDatasets.Dataset(nc_path, "r") do ds
    global fortran_m    = Int.(ds["m"][:])
    # NCDatasets reverses NetCDF dim order: file (i=2, m=34) → Julia (34, 2)
    b_n_x_raw = ds["b_n_x"][:, :]   # Julia shape: (34, 2); col 1=cos, col 2=sin
    global fortran_amps = sqrt.(b_n_x_raw[:, 1].^2 .+ b_n_x_raw[:, 2].^2)
end

println("    Fortran m range: $(minimum(fortran_m)):$(maximum(fortran_m)), mpert=$(length(fortran_m))")

# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------

println("\n    m    Julia |bmn| [T]     Fortran |bmn| [T]   ratio")
println("    " * "-"^52)
for (jm, ja) in zip(julia_m, julia_amps)
    k = findfirst(==(jm), fortran_m)
    if !isnothing(k)
        fa = fortran_amps[k]
        ratio = fa > 1e-30 ? ja / fa : NaN
        @printf "    %3d  %12.4e         %12.4e       %.3f\n" jm ja fa ratio
    end
end

# ---------------------------------------------------------------------------
# Comparison plot
# ---------------------------------------------------------------------------

println("\nGenerating comparison figure ...")

# Step-line helper: extend with zeros at both ends
function step_series(m_vals, amps)
    m_ext   = [m_vals[1]-1; m_vals; m_vals[end]+1]
    amp_ext = [0.0; amps; 0.0]
    return m_ext, amp_ext
end

p_spec = plot(; xlabel="Poloidal mode m", ylabel="|bmn| [T]",
                title="Normal field spectrum: Julia vs Fortran GPEC  (n=$N_TOROIDAL)",
                legend=:topright, size=(800, 400))

jm_ext, ja_ext = step_series(julia_m, julia_amps)
fm_ext, fa_ext = step_series(fortran_m, fortran_amps)

plot!(p_spec, jm_ext, ja_ext; seriestype=:steppre, lw=2, color=:blue,  label="Julia")
plot!(p_spec, fm_ext, fa_ext; seriestype=:steppre, lw=2, color=:red, linestyle=:dash, label="Fortran GPEC")

ylims!(p_spec, (0, Inf))
xlims!(p_spec, MLOW - 2, MHIGH + 2)

p_bn = plot_bn_contour(bn, MTHETA, NZETA; n=N_TOROIDAL)

fig = plot(p_bn, p_spec; layout=(1, 2), size=(1200, 450),
    plot_title="DIII-D 147131  (c+il+iu coils, n=$N_TOROIDAL)")

out_png = joinpath(OUTPUT_DIR, "coil_vs_fortran_n$(N_TOROIDAL).png")
savefig(fig, out_png)
println("Saved → $out_png")
println("\n" * "="^70)
println("Done.")
