"""
End-to-end check script for the coil ForcingTerms pipeline.

Loads a DIII-D-like equilibrium, computes the normal field from DIII-D I-coils
using Biot-Savart, Fourier decomposes the result, and produces a 4-panel
diagnostic figure:

  Panel 1: 3D coil wire geometry
  Panel 2: Coil cross-sections in (R, Z) with plasma boundary
  Panel 3: Normal field bn(θ, ζ) contour on plasma boundary
  Panel 4: Mode spectrum |bmn| vs m

Usage:
    julia --project=. benchmarks/check_coil_forcingterms.jl
"""

using GeneralizedPerturbedEquilibrium
using GeneralizedPerturbedEquilibrium.ForcingTerms
using GeneralizedPerturbedEquilibrium.Equilibrium
using GeneralizedPerturbedEquilibrium.Analysis.CoilForcing
using TOML
using Plots
using Printf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXAMPLE_DIR    = joinpath(@__DIR__, "..", "examples", "DIIID-like_ideal_example")
COIL_DIR       = joinpath(@__DIR__, "..", "src", "ForcingTerms", "coil_geometries")
OUTPUT_FILE    = joinpath(@__DIR__, "coil_forcingterms_check.png")

# Alternating-polarity current pattern (tests color mapping and n≠3 content)
COIL_CURRENTS  = [1000.0, 500.0, -500.0, -1000.0, -500.0, 500.0]  # Amperes per conductor
N_TOROIDAL     = 3
M_LOW          = -5   # ≈ n*floor(q_min) - delta_mlow = 3*1 - 8 (matching FFS convention)
M_HIGH         = 20   # ≈ n*ceil(q_max) + delta_mhigh = 3*4 + 8
MTHETA_COIL    = 192
NZETA_COIL     = 96    # = 32 * N_TOROIDAL

# ---------------------------------------------------------------------------
println("="^70)
println("Coil ForcingTerms End-to-End Check")
println("="^70)

# ---------------------------------------------------------------------------
# Load equilibrium
# ---------------------------------------------------------------------------
println("\n[1/4] Loading DIII-D-like equilibrium from $EXAMPLE_DIR ...")
t_equil = @elapsed begin
    inputs    = TOML.parsefile(joinpath(EXAMPLE_DIR, "gpec.toml"))
    eq_config = Equilibrium.EquilibriumConfig(inputs["Equilibrium"], EXAMPLE_DIR)
    equil     = Equilibrium.setup_equilibrium(eq_config)
end
@printf "    Done in %.1f s\n" t_equil

# ---------------------------------------------------------------------------
# Build coil sets
# ---------------------------------------------------------------------------
println("\n[2/4] Loading coil geometry and applying configuration ...")

il_file = joinpath(COIL_DIR, "d3d_il.dat")
iu_file = joinpath(COIL_DIR, "d3d_iu.dat")

isfile(il_file) || error("d3d_il.dat not found in coil_geometries/")
isfile(iu_file) || error("d3d_iu.dat not found in coil_geometries/")

il_raw = read_coil_dat(il_file)
iu_raw = read_coil_dat(iu_file)

il_set = CoilSet(il_raw.name, il_raw.ncoil, il_raw.s, il_raw.nw, il_raw.nsec,
    il_raw.x, il_raw.y, il_raw.z, Float64.(COIL_CURRENTS[1:il_raw.ncoil]))
iu_set = CoilSet(iu_raw.name, iu_raw.ncoil, iu_raw.s, iu_raw.nw, iu_raw.nsec,
    iu_raw.x, iu_raw.y, iu_raw.z, Float64.(COIL_CURRENTS[1:iu_raw.ncoil]))

coil_sets = [il_set, iu_set]
@printf "    %d coil sets loaded: %s (%d conductors), %s (%d conductors)\n" length(coil_sets) il_set.name il_set.ncoil iu_set.name iu_set.ncoil

# ---------------------------------------------------------------------------
# Compute coil forcing modes
# ---------------------------------------------------------------------------
println("\n[3/4] Computing Biot-Savart field and Fourier decomposition ...")
println("    n=$N_TOROIDAL, m=$M_LOW:$M_HIGH, grid=$(MTHETA_COIL)×$(NZETA_COIL) ...")

cfg = CoilConfig(;
    mtheta_coil = MTHETA_COIL,
    nzeta_coil  = NZETA_COIL,
    coil_sets   = CoilSetConfig[],  # config only used for grid parameters here
)

forcing_modes = ForcingMode[]
bnd_grid = nothing; bn = nothing  # declare outside @elapsed so they're accessible later
t_biot = @elapsed begin
    # Also capture bn for the contour plot
    bnd_grid = sample_boundary_grid(equil, MTHETA_COIL, NZETA_COIL)

    nobs  = MTHETA_COIL * NZETA_COIL
    obs_R   = zeros(nobs); obs_phi = zeros(nobs); obs_Z = zeros(nobs)
    for j in 1:NZETA_COIL, i in 1:MTHETA_COIL
        idx = i + (j-1)*MTHETA_COIL
        obs_R[idx]   = bnd_grid.R[i]
        obs_phi[idx] = bnd_grid.phi_grid[j]
        obs_Z[idx]   = bnd_grid.Z[i]
    end

    B_R = zeros(nobs); B_phi = zeros(nobs); B_Z = zeros(nobs)
    compute_biot_savart_boundary!(B_R, B_phi, B_Z, obs_R, obs_phi, obs_Z, coil_sets)

    bn = zeros(MTHETA_COIL, NZETA_COIL)
    project_normal_flux!(bn, B_R, B_Z, bnd_grid)

    modes = fourier_decompose_bn(bn, bnd_grid, N_TOROIDAL, M_LOW, M_HIGH)
    append!(forcing_modes, modes)
end

@printf "    Done in %.1f s\n" t_biot
@printf "    Max |bn|    = %.3e T\n" maximum(abs, bn)

amps = abs.([m.amplitude for m in forcing_modes])
dom_idx = argmax(amps)
@printf "    Max |bmn|   = %.3e T  (m=%d, n=%d)\n" amps[dom_idx] forcing_modes[dom_idx].m forcing_modes[dom_idx].n

println("\n    Mode spectrum:")
for mode in sort(forcing_modes, by=m -> m.m)
    @printf "      m=%2d  n=%d  |bmn|=%.3e T\n" mode.m mode.n abs(mode.amplitude)
end

# ---------------------------------------------------------------------------
# Generate 4-panel visualization
# ---------------------------------------------------------------------------
println("\n[4/4] Generating diagnostic figure ...")

p1 = plot_coil_geometry_3d(coil_sets)
p2 = plot_coil_geometry_rz(coil_sets; equil=equil)
p3 = plot_bn_contour(bn, MTHETA_COIL, NZETA_COIL; n=N_TOROIDAL)
p4 = plot_mode_spectrum(forcing_modes; mlow=M_LOW, mhigh=M_HIGH)

fig = plot(p1, p2, p3, p4;
    layout=(2, 2),
    size=(1200, 900),
    plot_title="DIII-D I-coil RMP check  (I=$(COIL_CURRENTS) A, n=$N_TOROIDAL)",
)

savefig(fig, OUTPUT_FILE)
println("    Saved → $OUTPUT_FILE")
println("\n" * "="^70)
println("All checks complete.")
