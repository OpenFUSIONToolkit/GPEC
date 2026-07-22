# Driver: DIII-D shot 196073 vacuum/plasma/total field-line tracing across three time slices.
#
# For each time slice (2840, 2845, 2850 ms) and each perturbation source
# (vacuum, plasma, total) this generates a gpec.toml, runs the full JPEC pipeline
# (equilibrium -> force-free states -> perturbed equilibrium -> field-line tracing),
# and writes every FieldLineTracing figure the flux tracer produces.
#
# Equilibria and coil currents are the OMFIT save for shot 196073 (I-coils iu/il + C-coil).

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
using GeneralizedPerturbedEquilibrium, Plots, HDF5
using GeneralizedPerturbedEquilibrium: Analysis
gr()

const HERE = @__DIR__

# Per-slice coil currents [A]: iu (I-coil upper), il (I-coil lower), c (C-coil), 6 coils each.
const SLICES = Dict(
    "2840" => (
        iu=[1543.6, -983.8, -2496.7, -1548.4, 935.1, 2503.5],
        il=[-3055.7, -424.0, 2572.0, 764.4, 420.3, -2631.4],
        c=[662.1, 709.0, 131.3, -637.6, -693.1, -125.4],
    ),
    "2845" => (
        iu=[874.8, -1416.9, -2245.8, -908.3, 1351.9, 2247.9],
        il=[-2808.9, 212.2, 2968.7, 762.3, -229.2, -3048.9],
        c=[661.2, 719.5, 130.5, -637.6, -704.7, -125.5],
    ),
    "2850" => (
        iu=[15.9, -1670.4, -1773.4, -88.4, 1583.9, 1762.7],
        il=[-2343.0, 1028.8, 3190.8, 756.4, -1067.3, -3291.3],
        c=[667.8, 714.3, 127.4, -649.4, -705.1, -128.1],
    ),
)

_arr(v) = "[" * join(string.(v), ", ") * "]"

function write_toml(dir, slice, source)
    cur = SLICES[slice]
    open(joinpath(dir, "gpec.toml"), "w") do io
        print(io, """
# DIII-D shot 196073 @ $(slice) ms — n=1 field-line tracing, source = $(source).
# EFIT equilibrium with I-coil (iu/il) + C-coil forcing from the shot's OMFIT save.

[Equilibrium]
eq_filename = "../g196073.0$(slice)"  # EFIT g-file for this time slice
eq_type = "efit"                       # Type of the input 2D equilibrium file
jac_type = "hamada"                    # Coordinate system
grid_type = "ldp"                      # Radial grid packing
psilow = 1e-4                          # Lower limit of normalized poloidal flux
psihigh = 0.99                         # Upper limit of normalized poloidal flux
mpsi = 256                             # Fixed radial grid
mtheta = 256                           # Number of poloidal grid points
newq0 = 0                              # Override for on-axis safety factor (0 = input value)
etol = 1e-10                           # Error tolerance for equilibrium solver
force_termination = false              # Terminate after equilibrium setup

[Wall]
shape = "nowall"                       # Free-boundary (no conducting wall)

[ForceFreeStates]
local_stability_flag = false           # Skip local Mercier/ballooning (not needed for tracing)
mat_flag = true                        # Construct coefficient matrices
ode_flag = true                        # Integrate Euler-Lagrange equation
vac_flag = true                        # Compute plasma/vacuum/total energies
qlow = 1.02                            # Integration lower q limit
qhigh = 1e3                            # Integration upper q limit
sing_start = 0                         # Start at the sing_start'th rational from the axis
nn_low = 1                             # Smallest toroidal mode number
nn_high = 1                            # Largest toroidal mode number
delta_mlow = 8                         # Expands lower bound of Fourier harmonics
delta_mhigh = 8                        # Expands upper bound of Fourier harmonics
mthvac = 512                           # Poloidal spline points at the plasma-vacuum interface
kinetic_source = "fixed"               # Ideal path
kinetic_factor = 0.0                   # Ideal (no kinetic)
eulerlagrange_tolerance = 1e-10        # ODE integration tolerance
singfac_min = 1e-4                     # Fractional distance from rational q for ideal jump
use_parallel = true                    # Parallel FM-propagator BVP path (unlocks Δ' matrix)
parallel_threads = 2                   # BVP thread cap
populate_dense_xi = true               # Dense ξ storage — required with [PerturbedEquilibrium]
set_psilim_via_dmlim = true            # Truncate at (last_rational_q + dmlim)/n (diverted eq.)
dmlim = 0.2                            # Truncation offset

[ForcingTerms]
forcing_data_format = "coil"           # Biot-Savart from 3D coil wires
machine = "d3d"                        # Geometry prefix (bundled d3d_*.dat)
mtheta_coil = 480                      # Poloidal grid for boundary B·n̂ evaluation
nzeta_coil = 40                        # Toroidal grid

[[ForcingTerms.coil_set]]
name = "iu"                            # DIII-D I-coil upper (6 coils)
currents = $(_arr(cur.iu))

[[ForcingTerms.coil_set]]
name = "il"                            # DIII-D I-coil lower (6 coils)
currents = $(_arr(cur.il))

[[ForcingTerms.coil_set]]
name = "c"                             # DIII-D C-coil (6 coils)
currents = $(_arr(cur.c))

[PerturbedEquilibrium]
fixed_boundary = false                 # Free-boundary response
output_eigenmodes = true               # Output eigenmode fields
compute_response = true                # Compute plasma response to forcing
compute_singular_coupling = true       # Singular-coupling metrics (island_width_sq drive)
verbose = true
write_outputs_to_HDF5 = true
reg_spot = 0.05                         # Regularization width for singular surfaces

[FieldLineTracing]
tracing_coords = "flux"                # Interior (ψ,θ,ζ) Hamiltonian map
tracing_field = "$(source)"            # Perturbation source
n_lines = 80                           # Field lines launched
psi_start = 0.05                       # Innermost launch surface
psi_end = 0.98                         # Outermost launch surface
n_transits = 500                       # Toroidal transits per line
phi_planes = [0.0]                     # Poincaré section angle [rad]
tol = 1e-9                             # ODE tolerance
compute_connection_length = true
compute_footprints = true
compute_island_width = true
write_outputs_to_HDF5 = true
verbose = true
""")
    end
end

function make_plots(dir, slice, source)
    h5 = joinpath(dir, "gpec.h5")
    tag = "$(slice)ms_$(source)"
    A = Analysis.FieldLineTracing
    savefig(A.plot_poincare(h5), joinpath(dir, "poincare_RZ_$(tag).png"))
    savefig(A.plot_poincare_flux(h5), joinpath(dir, "poincare_flux_$(tag).png"))
    savefig(A.plot_connection_length(h5), joinpath(dir, "connection_length_$(tag).png"))
    savefig(A.plot_island_widths(h5), joinpath(dir, "island_widths_$(tag).png"))
    savefig(A.plot_field_line_tracing_summary(h5), joinpath(dir, "summary_$(tag).png"))
end

const RESULTS = Tuple{String,String,Vector{Float64}}[]

for slice in ["2840", "2845", "2850"]
    for source in ["vacuum", "plasma", "total"]
        dir = joinpath(HERE, "t$(slice)_$(source)")
        mkpath(dir)
        @info "==== RUN: shot 196073 @ $(slice) ms, source=$(source) ===="
        write_toml(dir, slice, source)
        try
            GeneralizedPerturbedEquilibrium.main([dir])
            make_plots(dir, slice, source)
            w = h5open(joinpath(dir, "gpec.h5"), "r") do f
                read(f["field_line_tracing/islands/half_width"])
            end
            push!(RESULTS, (slice, source, w))
            @info "   done: island half-widths = $(round.(w; digits=4))"
        catch e
            @error "   FAILED $(slice)/$(source)" exception = (e, catch_backtrace())
        end
    end
end

println("\n==================== SUMMARY: island half-widths (ψ_N) ====================")
for (slice, source, w) in RESULTS
    println("  $(slice) ms  $(rpad(source, 7))  ", round.(w; digits=4))
end
println("Figures written under examples/TEMP_DIIID_196073_field_lines/t<slice>_<source>/")
