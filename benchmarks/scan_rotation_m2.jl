# Plot the m=2 area-normalized b^ψ (PerturbedEquilibrium/Response/psi_area) across a ROTATION scan
# of the resistive gal matched PE runs (gal_match_flag=true, gal_ideal_flag=false), fixed η=8e-8,
# rotation f = 1,2,4,8,16 Hz (forced eigenvalue γ_s = 2πi·n·f). One curve per rotation; overlays the
# shooting (ideal) reference. Scan dirs produced by the bash loop over /tmp/rotscan_<f>.
# Usage: julia --project=. benchmarks/scan_rotation_m2.jl [out.png] [m]

using HDF5, Plots, Printf, TOML

outpng = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "scan_rotation_m2.png")
mtarget = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2

to_c(a) = eltype(a) <: Complex ? ComplexF64.(a) : map(x -> ComplexF64(x.re, x.im), a)

function read_m2(h5; gal::Bool)
    h5open(h5) do f
        pa = to_c(read(f["PerturbedEquilibrium/Response/psi_area"]))
        col = mtarget - read(f["Info/mlow"]) + 1
        psi = gal ? read(f["ForceFreeStates/Solutions/GalerkinIntegration/Solution/psi"])[.!Bool.(read(f["ForceFreeStates/Solutions/GalerkinIntegration/Solution/issing"]))] :
              read(f["ForceFreeStates/Solutions/ForwardIntegration/psi"])
        (psi, pa[:, col])
    end
end

scandirs = filter(d -> isfile(joinpath(d, "gpec.h5")),
    ["/tmp/rotscan_1", "/tmp/rotscan_2", "/tmp/rotscan_4", "/tmp/rotscan_8", "/tmp/rotscan_16"])
rots = [TOML.parsefile(joinpath(d, "gpec.toml"))["ForceFreeStates"]["gal_rotation"][1] for d in scandirs]
ord = sortperm(rots)
scandirs, rots = scandirs[ord], rots[ord]
@printf("%d scan runs: rotation f = %s Hz  (η fixed = 8e-8)\n", length(rots), join((@sprintf("%g", r) for r in rots), ", "))

sing_psi, sing_m = h5open(joinpath(scandirs[1], "gpec.h5")) do f
    (read(f["SingularSurfaces/GalerkinDeltaPrime/sing_psi"]), read(f["SingularSurfaces/GalerkinDeltaPrime/sing_m"]))
end
psi_res = mtarget in sing_m ? sing_psi[findfirst(==(mtarget), sing_m)] : NaN

cols = cgrad(:plasma, max(length(rots), 2); categorical=true)
plt = plot(; size=(1000, 640), xlabel="ψ_N", ylabel="|b^ψ / ⟨J·|∇ψ|⟩|  (area-normalized)",
    title="m=$mtarget perturbed normal field — rotation scan (η=8e-8, resistive gal-matched PE)",
    legend=:topleft, left_margin=13Plots.mm, bottom_margin=5Plots.mm, right_margin=4Plots.mm)
for (i, (d, r)) in enumerate(zip(scandirs, rots))
    psi, v = read_m2(joinpath(d, "gpec.h5"); gal=true)
    plot!(plt, psi, abs.(v); color=cols[i], lw=2, label=@sprintf("f = %g Hz", r))
end
if isfile("/tmp/shooting_test/gpec.h5")
    psis, vs = read_m2("/tmp/shooting_test/gpec.h5"; gal=false)
    plot!(plt, psis, abs.(vs); color=:black, ls=:dash, lw=2, label="shooting (ideal)")
end
isnan(psi_res) || vline!(plt, [psi_res]; color=:red, ls=:dot, lw=1.6, label="q=$mtarget surface")

savefig(plt, outpng)
println("saved: ", abspath(outpng))
