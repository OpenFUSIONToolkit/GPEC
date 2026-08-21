# Plot the m=2 area-normalized b^ψ (PerturbedEquilibrium/Response/psi_area) across a resistivity scan
# of the RESISTIVE gal matched PE runs (gal_match_flag=true, gal_ideal_flag=false), one curve per η.
# Overlays the forward (ideal, η→0) reference. The η-scan dirs are produced by the bash loop over
# /tmp/etascan_<factor> (each a copy of the 0.993 config with gal_eta scaled).
# Usage: julia --project=. benchmarks/scan_resistivity_m2.jl [out.png] [m]

using HDF5, Plots, Printf, TOML

outpng = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "scan_resistivity_m2.png")
mtarget = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2

to_c(a) = eltype(a) <: Complex ? ComplexF64.(a) : map(x -> ComplexF64(x.re, x.im), a)

# read m=target area-normalized b^ψ on the run's PE grid
function read_m2(h5; gal::Bool)
    h5open(h5) do f
        pa = to_c(read(f["PerturbedEquilibrium/Response/psi_area"]))   # [npsi, mpert]
        col = mtarget - read(f["Info/mlow"]) + 1
        psi = gal ? read(f["ForceFreeStates/Solutions/GalerkinIntegration/psi"]) :
              read(f["ForceFreeStates/Solutions/ForwardIntegration/psi"])
        (psi, pa[:, col])
    end
end

# collect scan dirs that finished
scandirs = filter(d -> isfile(joinpath(d, "gpec.h5")),
    ["/tmp/etascan_0p01", "/tmp/etascan_0p1", "/tmp/etascan_1", "/tmp/etascan_10", "/tmp/etascan_100"])
etas = [TOML.parsefile(joinpath(d, "gpec.toml"))["ForceFreeStates"]["gal_eta"][1] for d in scandirs]
ord = sortperm(etas)
scandirs, etas = scandirs[ord], etas[ord]
eta_ref = 8e-8
@printf("%d scan runs: η = %s  (×η_ref where η_ref=%.0e)\n", length(etas),
    join((@sprintf("%.0e", e) for e in etas), ", "), eta_ref)

# rational surface for m=target
sing_psi, sing_m = h5open(joinpath(scandirs[1], "gpec.h5")) do f
    (read(f["ForceFreeStates/Solutions/GalerkinIntegration/rational_psi"]), read(f["ForceFreeStates/Solutions/GalerkinIntegration/rational_m"]))
end
psi_res = mtarget in sing_m ? sing_psi[findfirst(==(mtarget), sing_m)] : NaN

cols = cgrad(:viridis, max(length(etas), 2); categorical=true)
plt = plot(; size=(1000, 640), xlabel="ψ_N", ylabel="|b^ψ / ⟨J·|∇ψ|⟩|  (area-normalized)",
    title="m=$mtarget perturbed normal field — resistivity scan (resistive gal-matched PE)",
    legend=:topleft, left_margin=13Plots.mm, bottom_margin=5Plots.mm, right_margin=4Plots.mm)
for (i, (d, e)) in enumerate(zip(scandirs, etas))
    psi, v = read_m2(joinpath(d, "gpec.h5"); gal=true)
    plot!(plt, psi, abs.(v); color=cols[i], lw=2, label=@sprintf("η = %g×η_ref", round(e / eta_ref; sigdigits=2)))
end
# forward (ideal, η→0) reference
if isfile("/tmp/forward_test/gpec.h5")
    psis, vs = read_m2("/tmp/forward_test/gpec.h5"; gal=false)
    plot!(plt, psis, abs.(vs); color=:black, ls=:dash, lw=2, label="forward (ideal, η→0)")
end
isnan(psi_res) || vline!(plt, [psi_res]; color=:red, ls=:dot, lw=1.6, label="q=$mtarget surface")

savefig(plt, outpng)
println("saved: ", abspath(outpng))
