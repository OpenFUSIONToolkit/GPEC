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

println("\n[1/5] Loading equilibrium (g147131.02300_DIIID_KEFIT, psihigh=0.993, psilim read from NC) ...")
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
println("    bt_sign=$(equil.params.bt_sign), crnt=$(round(equil.params.crnt; digits=0)) A")

# ---------------------------------------------------------------------------
# Build coil sets matching coil.in
# ---------------------------------------------------------------------------

println("\n[2/5] Loading coil geometry (d3d c + il + iu) ...")

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

println("\n[3/5] Computing Julia coil forcing modes (n=$N_TOROIDAL, m=$MLOW:$MHIGH, $(MTHETA)×$(NZETA)) ...")

julia_modes = ForcingMode[]
t_biot = @elapsed begin
    bnd_grid = sample_boundary_grid(equil, MTHETA, NZETA)
    nobs = MTHETA * NZETA
    obs_R   = zeros(nobs); obs_phi = zeros(nobs); obs_Z = zeros(nobs)
    for j in 1:NZETA, i in 1:MTHETA
        idx = i + (j-1)*MTHETA
        obs_R[idx]   = bnd_grid.R[i]
        obs_phi[idx] = bnd_grid.phi_grid[j] + bnd_grid.phi_offset[i]
        obs_Z[idx]   = bnd_grid.Z[i]
    end
    B_R = zeros(nobs); B_phi = zeros(nobs); B_Z = zeros(nobs)
    compute_biot_savart_boundary!(B_R, B_phi, B_Z, obs_R, obs_phi, obs_Z, coil_sets)
    bn = zeros(MTHETA, NZETA)
    project_normal_flux!(bn, B_R, B_Z, bnd_grid)
    append!(julia_modes, fourier_decompose_bn(bn, bnd_grid, N_TOROIDAL, MLOW, MHIGH))
end
@printf "    Done in %.1f s,  max|Phi_x| = %.3e T·m²\n" t_biot maximum(abs, bn)
println("    Boundary (psi=1.0): R∈[$(round(minimum(bnd_grid.R);digits=3)), $(round(maximum(bnd_grid.R);digits=3))]  Z∈[$(round(minimum(bnd_grid.Z);digits=3)), $(round(maximum(bnd_grid.Z);digits=3))]")
println("    phi_offset range: [$(round(minimum(bnd_grid.phi_offset);digits=3)), $(round(maximum(bnd_grid.phi_offset);digits=3))] rad")

julia_m    = [md.m for md in sort(julia_modes; by=md -> md.m)]
julia_amps = abs.([md.amplitude for md in sort(julia_modes; by=md -> md.m)])

# ---------------------------------------------------------------------------
# Fortran: read b_n_x and Phi_x from gpec_control_output_n1.nc
# ---------------------------------------------------------------------------

println("\n[4/5] Reading Fortran output (gpec_control_output_n1.nc) ...")

nc_path = joinpath(FORTRAN_DIR, "gpec_control_output_n1.nc")
isfile(nc_path) || error("NetCDF file not found: $nc_path")

fortran_m       = Int[]
fortran_bnx     = Float64[]   # b_n_x: externally applied normal field [T]
fortran_Phix    = Float64[]   # Phi_x: external flux [Wb]
fortran_psilim  = NaN
NCDatasets.Dataset(nc_path, "r") do ds
    global fortran_m      = Int.(ds["m"][:])
    global fortran_psilim = Float64(ds.attrib["psilim"])
    # NCDatasets reverses NetCDF dim order: file (i=2, m=34) → Julia (m=34, i=2)
    b_n_x_raw  = ds["b_n_x"][:, :]   # (34, 2); col 1=cos, col 2=sin
    phi_x_raw  = ds["Phi_x"][:, :]   # (34, 2)
    global fortran_bnx  = sqrt.(b_n_x_raw[:, 1].^2 .+ b_n_x_raw[:, 2].^2)
    global fortran_Phix = sqrt.(phi_x_raw[:, 1].^2  .+ phi_x_raw[:, 2].^2)
end

@printf "    Fortran psilim = %.6f\n" fortran_psilim
println("    Fortran m range: $(minimum(fortran_m)):$(maximum(fortran_m)), mpert=$(length(fortran_m))")

# ---------------------------------------------------------------------------
# Recompute Julia modes at the Fortran psilim surface for apples-to-apples
# ---------------------------------------------------------------------------

println("\n[5/5] Recomputing Julia modes at psi=psilim=$(round(fortran_psilim; digits=6)) ...")

julia_modes_lim = ForcingMode[]
t_lim = @elapsed begin
    bnd_grid_lim = sample_boundary_grid(equil, MTHETA, NZETA; psi=fortran_psilim)
    nobs = MTHETA * NZETA
    obs_R_l   = zeros(nobs); obs_phi_l = zeros(nobs); obs_Z_l = zeros(nobs)
    for j in 1:NZETA, i in 1:MTHETA
        idx = i + (j-1)*MTHETA
        obs_R_l[idx]   = bnd_grid_lim.R[i]
        obs_phi_l[idx] = bnd_grid_lim.phi_grid[j] + bnd_grid_lim.phi_offset[i]
        obs_Z_l[idx]   = bnd_grid_lim.Z[i]
    end
    B_R_l = zeros(nobs); B_phi_l = zeros(nobs); B_Z_l = zeros(nobs)
    compute_biot_savart_boundary!(B_R_l, B_phi_l, B_Z_l, obs_R_l, obs_phi_l, obs_Z_l, coil_sets)
    bn_lim = zeros(MTHETA, NZETA)
    project_normal_flux!(bn_lim, B_R_l, B_Z_l, bnd_grid_lim)
    append!(julia_modes_lim, fourier_decompose_bn(bn_lim, bnd_grid_lim, N_TOROIDAL, MLOW, MHIGH))
end
@printf "    Done in %.1f s,  max|Phi_x| at psilim = %.3e T·m²\n" t_lim maximum(abs, bn_lim)
println("    Boundary (psi=psilim): R∈[$(round(minimum(bnd_grid_lim.R);digits=3)), $(round(maximum(bnd_grid_lim.R);digits=3))]  Z∈[$(round(minimum(bnd_grid_lim.Z);digits=3)), $(round(maximum(bnd_grid_lim.Z);digits=3))]")
# Check ±m symmetry: if |Phi_x(m)| == |Phi_x(-m)| for all m, boundary may be incorrectly symmetric
amps_lim_dict = Dict(md.m => abs(md.amplitude) for md in julia_modes_lim)
println("    ±m symmetry check: |Phi_x(3)|=$(round(get(amps_lim_dict,3,0.0);sigdigits=4))  |Phi_x(-3)|=$(round(get(amps_lim_dict,-3,0.0);sigdigits=4))")

julia_m_lim    = [md.m for md in sort(julia_modes_lim; by=md -> md.m)]
julia_amps_lim = abs.([md.amplitude for md in sort(julia_modes_lim; by=md -> md.m)])

# ---------------------------------------------------------------------------
# Print comparison table (psilim surface vs both Fortran outputs)
# ---------------------------------------------------------------------------

println("\n    m    Julia Phi_x [T·m²]   Fortran b_n_x [T]  ratio_bnx    Fortran Phi_x [Wb]  Phix/Julia")
println("    " * "-"^90)
for (jm, ja) in zip(julia_m_lim, julia_amps_lim)
    k = findfirst(==(jm), fortran_m)
    if !isnothing(k)
        fb = fortran_bnx[k]; fp = fortran_Phix[k]
        ratio_bnx      = fb > 1e-30 ? ja/fb : NaN
        phix_over_julia = ja > 1e-40 ? fp/ja : NaN
        @printf "    %3d  %12.4e        %12.4e      %.3f        %12.4e       %.4f\n" jm ja fb ratio_bnx fp phix_over_julia
    end
end
println("    Expected Phix/Julia ≈ 1.000 for all modes (Julia now outputs unit-norm = Phi_x).")

# ---------------------------------------------------------------------------
# Comparison plot
# ---------------------------------------------------------------------------

println("\nGenerating comparison figure ...")

function step_series(m_vals, amps)
    m_ext   = [m_vals[1]-1; m_vals; m_vals[end]+1]
    amp_ext = [0.0; amps; 0.0]
    return m_ext, amp_ext
end

# Left panel: Julia at psi=1 (boundary) vs psilim to show surface sensitivity
p_surf = plot(; xlabel="Poloidal mode m", ylabel="|Phi_x| [T·m²]",
                title="Surface sensitivity",
                legend=:topright)
jm0_ext, ja0_ext = step_series(julia_m,     julia_amps)
jml_ext, jal_ext = step_series(julia_m_lim, julia_amps_lim)
plot!(p_surf, jm0_ext, ja0_ext; seriestype=:steppre, lw=2, color=:blue,
      label="Julia ψ=1.0")
plot!(p_surf, jml_ext, jal_ext; seriestype=:steppre, lw=2, color=:dodgerblue,
      linestyle=:dash, label="Julia ψ=$(round(fortran_psilim; digits=4))")
ylims!(p_surf, (0, Inf)); xlims!(p_surf, MLOW-2, MHIGH+2)

# Right panel: Julia@psilim vs Fortran b_n_x and Phi_x
p_comp = plot(; xlabel="Poloidal mode m", ylabel="amplitude [T·m²]",
                title="vs Fortran (n=$N_TOROIDAL)",
                legend=:topright)
fm_bnx_ext,  fa_bnx_ext  = step_series(fortran_m, fortran_bnx)
fm_Phix_ext, fa_Phix_ext = step_series(fortran_m, fortran_Phix)
plot!(p_comp, jml_ext,      jal_ext;      seriestype=:steppre, lw=2, color=:blue,
      label="Julia Phi_x ψ=psilim [T·m²]")
plot!(p_comp, fm_bnx_ext,   fa_bnx_ext;   seriestype=:steppre, lw=2, color=:red,
      linestyle=:dash,  label="Fortran b_n_x [T]")
plot!(p_comp, fm_Phix_ext,  fa_Phix_ext;  seriestype=:steppre, lw=2, color=:orange,
      linestyle=:dot,   label="Fortran Phi_x [Wb]")
ylims!(p_comp, (0, Inf)); xlims!(p_comp, MLOW-2, MHIGH+2)

# Third panel: Phi_x / Julia ratio per mode (should be ≈ 1.0 — Julia now outputs unit-norm)
p_ratio = plot(; xlabel="Poloidal mode m", ylabel="Phi_x / Julia",
                 title="Normalization ratio (should be flat at 1.0)",
                 legend=:topright)
ratio_m   = Int[]
ratio_vals = Float64[]
for (jm, ja) in zip(julia_m_lim, julia_amps_lim)
    k = findfirst(==(jm), fortran_m)
    if !isnothing(k) && ja > 1e-40
        push!(ratio_m, jm)
        push!(ratio_vals, fortran_Phix[k] / ja)
    end
end
if !isempty(ratio_m)
    scatter!(p_ratio, ratio_m, ratio_vals; color=:blue, label="Phi_x / Julia", markersize=5)
    hline!(p_ratio, [1.0]; color=:red, linestyle=:dash, label="expected = 1.0")
end
xlims!(p_ratio, MLOW-2, MHIGH+2)

fig = plot(p_surf, p_comp, p_ratio; layout=(1, 3), size=(1800, 450),
    plot_title="DIII-D 147131  c+il+iu, n=$N_TOROIDAL | " *
               "Fortran psilim=$(round(fortran_psilim; digits=4))  " *
               "Julia psi=1.0 vs $(round(fortran_psilim; digits=4))")

out_png = joinpath(OUTPUT_DIR, "coil_vs_fortran_n$(N_TOROIDAL).png")
savefig(fig, out_png)
println("Saved → $out_png")
println("\n" * "="^70)
println("Done.")
