# 2D scan: gal-vs-STRIDE q=3 tracking over (pc, pfac). STRIDE is pfac-independent
# (recorded per run as a consistency check). Output /tmp/gal_pfac_scan2d.csv.
using GeneralizedPerturbedEquilibrium
using HDF5, TOML, Printf
const REPO = "/Users/pharr/Projects/GPEC_dev/docs/GPEC_julia"
const BASE = TOML.parsefile(joinpath(REPO, "examples", "LAR_beta_scan", "gpec.toml"))
const PCS = [0.005, 0.010, 0.020, 0.030, 0.040, 0.050, 0.060, 0.070, 0.085, 0.100]
const PFACS = [5e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]

function extract(h5)
    h5open(h5, "r") do f
        mlow = read(f, "info/mlow"); mpert = read(f, "info/mpert"); nlow = read(f, "info/nlow")
        m_s = vec(read(f, "singular/m"))
        dpm = read(f, "singular/delta_prime_matrix")
        gm = vec(read(f, "galerkin/sing_m")); gp = read(f, "galerkin/pest3_Delta")
        g3 = NaN; s3 = NaN; g2 = NaN; s2 = NaN
        for (s, m) in enumerate(m_s)
            gi = findfirst(==(m), gm)
            if m == 2; g2 = real(gp[gi, gi]); s2 = real(dpm[s, s]); end
            if m == 3; g3 = real(gp[gi, gi]); s3 = real(dpm[s, s]); end
        end
        return (g2, s2, g3, s3)
    end
end

io = open("/tmp/gal_pfac_scan2d.csv", "w")
println(io, "pc,pfac,gal_q2,stride_q2,gal_q3,stride_q3"); flush(io)
t0 = time(); n = 0; ntot = length(PCS) * length(PFACS)
for pc in PCS, pfac in PFACS
    global n += 1
    dir = mktempdir(; prefix="gal2d_")
    c = deepcopy(BASE)
    c["TJ_ANALYTIC_INPUT"]["pc"] = pc
    c["ForceFreeStates"]["gal_flag"] = true
    c["ForceFreeStates"]["gal_solver"] = "LU"
    c["ForceFreeStates"]["gal_pfac"] = pfac
    c["ForceFreeStates"]["HDF5_filename"] = joinpath(dir, "gpec.h5")
    open(joinpath(dir, "gpec.toml"), "w") do f; TOML.print(f, c); end
    try
        GeneralizedPerturbedEquilibrium.main([dir])
        g2, s2, g3, s3 = extract(joinpath(dir, "gpec.h5"))
        @printf(io, "%.6f,%.6e,%.6f,%.6f,%.6f,%.6f\n", pc, pfac, g2, s2, g3, s3); flush(io)
        @printf("[%3d/%3d] pc=%.4f pfac=%.0e  q3: gal=%+8.3f stride=%+8.3f  relerr=%5.1f%%  tot %.0fs\n",
            n, ntot, pc, pfac, g3, s3, 100abs(g3 - s3) / abs(s3), time() - t0); flush(stdout)
    catch e
        @warn "pc=$pc pfac=$pfac failed" exception=(e, catch_backtrace())
    finally
        rm(dir; force=true, recursive=true)
    end
end
close(io)
println("Saved /tmp/gal_pfac_scan2d.csv")
println("DONE")
