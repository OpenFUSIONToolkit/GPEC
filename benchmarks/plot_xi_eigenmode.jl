# Plot the ξ(ψ) profile of the eigenmode with the highest total-energy eigenvalue.
#
# The total energy operator W = W_plasma + W_vacuum (free_run, Free.jl); its eigenvectors are the
# free-boundary edge displacement patterns and the eigenvalues are δW. This picks the eigenmode with the
# largest Re(eigenvalue) and reconstructs its radial profile by projecting the EL fundamental matrix
# (ForceFreeStates/Solutions/ForwardIntegration/xi_psi) onto that edge eigenvector:  c = U_edge \ w,  ξ(ψ) = U(ψ)·c.
#
# Requires a run with integrator="forward" so ForceFreeStates/Solutions/ForwardIntegration/xi_psi is the dense axis-basis fundamental
# matrix (not the Riccati S-matrices).
# Usage: julia --project=. benchmarks/plot_xi_eigenmode.jl [path/to/gpec.h5] [out.png]

using HDF5, LinearAlgebra, Plots, Printf

h5path = length(ARGS) >= 1 ? ARGS[1] : "examples/DIIID-like_ideal_example/gpec.h5"
outpng = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "xi_eigenmode.png")

to_c(a) = eltype(a) <: Complex ? ComplexF64.(a) : map(x -> ComplexF64(x.re, x.im), a)

et, wt, u1, psi, mlow, sing_psi = h5open(h5path) do f
    (to_c(read(f["ForceFreeStates/FreeBoundaryStability/eigenmode_energies"])),
        to_c(read(f["ForceFreeStates/FreeBoundaryStability/W_freeboundary_eigenmodes"])),
        to_c(read(f["ForceFreeStates/Solutions/ForwardIntegration/xi_psi"])),
        read(f["ForceFreeStates/Solutions/ForwardIntegration/psi"]),
        read(f["Info/mlow"]),
        haskey(f, "ForceFreeStates/Solutions/GalerkinIntegration/rational_psi") ? read(f["ForceFreeStates/Solutions/GalerkinIntegration/rational_psi"]) : Float64[])
end

mpert, _, nstep = size(u1)
k = argmax(real.(et))                  # eigenmode with the highest total-energy eigenvalue
@printf("highest-eigenvalue mode: k=%d   δW = %.4e %+.4ei   (of %d modes)\n",
    k, real(et[k]), imag(et[k]), length(et))

# Reconstruct ξ(ψ) for the eigenmode whose edge displacement is the eigenvector w = wt[:,k].
w = wt[:, k]
U_edge = u1[:, :, nstep]               # edge fundamental matrix (psi inner→edge, last index = edge)
c = U_edge \ w                         # U_edge·c = w
xi = Array{ComplexF64}(undef, mpert, nstep)
for ip in 1:nstep
    @views mul!(xi[:, ip], u1[:, :, ip], c)
end

ms = mlow .+ (0:mpert-1)
peak = [maximum(abs, @view xi[i, :]) for i in 1:mpert]
order = sortperm(peak; rev=true)       # dominant harmonics first

plt = plot(; size=(950, 620), xlabel="ψ_N", ylabel="|ξ^ψ_m(ψ)|",
    title=@sprintf("Highest total-energy eigenmode   δW = %.3e", real(et[k])),
    legend=:topleft, left_margin=12Plots.mm, bottom_margin=5Plots.mm, right_margin=4Plots.mm)
for i in 1:mpert                        # all harmonics, faint
    plot!(plt, psi, abs.(@view xi[i, :]); color=:gray85, lw=0.6, label="")
end
for i in order[1:min(8, mpert)]         # dominant harmonics, bold + labeled
    plot!(plt, psi, abs.(@view xi[i, :]); lw=2, label="m=$(ms[i])")
end
for (j, ps) in enumerate(sing_psi)      # rational surfaces
    vline!(plt, [ps]; color=:black, ls=:dash, lw=1, label=(j == 1 ? "rational q" : ""))
end

savefig(plt, outpng)
println("dominant harmonics (m): ", [ms[i] for i in order[1:min(8, mpert)]])
println("saved: ", abspath(outpng))
