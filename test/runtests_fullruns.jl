using HDF5
using LinearAlgebra
using TOML

"""
Walk two HDF5 groups and require every numeric dataset present in both to agree. Reflective on
purpose: a field added later, or left describing a stale plasma edge, is compared without anyone
having to remember to list it here.
"""
function compare_h5_numeric(ha, hb, path; skip=String[], rtol=1e-4, atol=1e-12)
    haskey(ha, path) && haskey(hb, path) || return
    ga, gb = ha[path], hb[path]
    for k in keys(ga)
        k in skip && continue
        haskey(gb, k) || continue
        isempty(read(ga[k])) && continue
        if ga[k] isa HDF5.Group
            compare_h5_numeric(ha, hb, "$path/$k"; skip, rtol, atol)
            continue
        end
        va, vb = read(ga[k]), read(gb[k])
        (eltype(va) <: Number && eltype(vb) <: Number) || continue
        @test size(va) == size(vb)
        size(va) == size(vb) || continue
        @test isapprox(va, vb; rtol, atol)
    end
    return
end


# Run GeneralizedPerturbedEquilibrium.main on the provided example directories and assert it completes without throwing.
@testset "Full ForceFreeStates runs" begin
    @testset "dW-peak truncation leaves no state describing the old plasma edge" begin
        # truncate_at_dW_peak moves the plasma boundary inward after the run has already derived
        # state from the original one. Anything keyed to the edge has to move with it. These
        # settings put the dW peak back inside a rational surface the integration had already
        # crossed, which is the case that exposes it: psiedge = 0.97 brings a second rational into
        # the scan band so a complete interior lobe outranks the truncated outermost one.
        ex = joinpath(@__DIR__, "..", "examples", "DIIID-like_ideal_example")
        d = mktempdir()
        for f in readdir(ex)
            cp(joinpath(ex, f), joinpath(d, f); force=true)
        end
        inputs = TOML.parsefile(joinpath(ex, "gpec.toml"))
        inputs["ForceFreeStates"]["psiedge"] = 0.97
        inputs["ForceFreeStates"]["truncate_at_dW_peak"] = true
        inputs["ForceFreeStates"]["dmlim"] = 0.01
        inputs["ForceFreeStates"]["nn_low"] = 2
        inputs["ForceFreeStates"]["nn_high"] = 2
        inputs["ForceFreeStates"]["verbose"] = false
        open(joinpath(d, "gpec.toml"), "w") do io
            TOML.print(io, inputs)
        end
        GeneralizedPerturbedEquilibrium.main([d])

        qlim_pullback = h5open(f -> read(f["Info/qlim"]), joinpath(d, "gpec.h5"), "r")

        h5open(joinpath(d, "gpec.h5"), "r") do h5
            psilim = read(h5["Info/psilim"])
            qlim = read(h5["Info/qlim"])
            rational_psi = read(h5["SingularSurfaces/rational_psi"])
            scan_psi = read(h5["ForceFreeStates/EdgeScan/psi"])
            scan_q = read(h5["ForceFreeStates/EdgeScan/q"])

            # The precondition the rest of the test rests on. The scan integrated past q = 6, which
            # is resonant at n = 2 (m = 12), and the edge then settled below it — so that surface
            # was crossed and is now outside the plasma. Without this the test would pass while
            # exercising nothing, which is the failure mode that hides a regression here.
            @test maximum(scan_q) > 6.0
            @test qlim < 6.0
            @test psilim < maximum(scan_psi)

            # Nothing outside the plasma survives in the surface list. Left stale, the perturbed
            # equilibrium evaluates those surfaces by extrapolating off the end of the solution.
            @test all(<=(psilim), rational_psi)
            # and the per-surface asymptotic stores stay the same length as the list.
            @test size(read(h5["SingularSurfaces/ca_left"]), 4) == length(rational_psi)
            @test maximum(read(h5["SingularSurfaces/rational_q"])) < qlim

            # dq/dψ is re-derived at the new edge rather than left describing the old one. The scan
            # tabulates q(ψ) across the band, so the edge value is checkable against a difference
            # taken from the run's own output; a stale q1lim is off by roughly a factor of two here.
            i = searchsortedfirst(scan_psi, psilim)
            dqdpsi_edge = (scan_q[i+1] - scan_q[i-1]) / (scan_psi[i+1] - scan_psi[i-1])
            @test read(h5["Info/dqdpsi_lim"]) ≈ dqdpsi_edge rtol = 0.05

            # The stored edge displacement matrix is what the perturbed-equilibrium boundary solve
            # inverts. Reductions recorded beyond the new edge would leave it rank deficient — the
            # unfixed code reaches cond ~1e17 here, against O(1) when the edge is consistent.
            xi = read(h5["ForceFreeStates/Solutions/ForwardIntegration/xi_psi"])
            edge = ndims(xi) == 3 ? xi[:, :, end] : xi
            @test cond(edge) < 1e8
        end

        # Now reach the same edge without truncating, and require the two to agree. Asking for the
        # edge through qhigh rather than psihigh keeps qmax, and with it the poloidal mode range and
        # the radial grid, identical — so anything that differs is the route to the edge, not the
        # discretization. This is the part that catches state nobody thought to assert: a field left
        # describing the old boundary shows up here whether or not the test knows it exists.
        d2 = mktempdir()
        for f in readdir(ex)
            cp(joinpath(ex, f), joinpath(d2, f); force=true)
        end
        inputs["ForceFreeStates"]["psiedge"] = 1.0                  # no scan, so no truncation
        inputs["ForceFreeStates"]["truncate_at_dW_peak"] = false
        inputs["ForceFreeStates"]["set_psilim_via_dmlim"] = false
        inputs["ForceFreeStates"]["qhigh"] = qlim_pullback
        open(joinpath(d2, "gpec.toml"), "w") do io
            TOML.print(io, inputs)
        end
        GeneralizedPerturbedEquilibrium.main([d2])

        h5open(joinpath(d, "gpec.h5"), "r") do ha
            h5open(joinpath(d2, "gpec.h5"), "r") do hb
                # EdgeScan exists only in the truncated run; git_version is not a result.
                compare_h5_numeric(ha, hb, "Info"; skip=["git_version"])
                compare_h5_numeric(ha, hb, "SingularSurfaces")
                # Solutions/ is skipped wholesale: the adaptive integrator takes a different number
                # of steps when it stops at the edge rather than integrating past it and truncating,
                # so those arrays differ in length by construction. The boundary values are what the
                # perturbed equilibrium consumes, so compare the final step of each explicitly.
                compare_h5_numeric(ha, hb, "ForceFreeStates"; skip=["EdgeScan", "Solutions"])
                for k in ("xi_psi", "u2", "xi_s", "dxi_psidpsi")
                    va = read(ha["ForceFreeStates/Solutions/ForwardIntegration/$k"])
                    vb = read(hb["ForceFreeStates/Solutions/ForwardIntegration/$k"])
                    @test isapprox(va[:, :, end], vb[:, :, end]; rtol=1e-4, atol=1e-12)
                end
                pa = read(ha["ForceFreeStates/Solutions/ForwardIntegration/psi"])
                pb = read(hb["ForceFreeStates/Solutions/ForwardIntegration/psi"])
                @test pa[end] ≈ pb[end]
            end
        end

        rm(d; recursive=true, force=true)
        rm(d2; recursive=true, force=true)
    end

    ex1 = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example")
    @info "Running Solovev ideal example"
    @test begin
        GeneralizedPerturbedEquilibrium.main([ex1])
        true
    end

    ex2 = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example_multi_n")
    @info "Running Solovev ideal multi-n example"
    @test begin
        GeneralizedPerturbedEquilibrium.main([ex2])
        true
    end

    ex3 = joinpath(@__DIR__, "test_data", "regression_solovev_kinetic_example")
    @info "Running Solovev kinetic example (kinetic_source=fixed, kinetic_factor=1e-9)"
    @test begin
        GeneralizedPerturbedEquilibrium.main([ex3])
        h5open(joinpath(ex3, "gpec.h5"), "r") do h5
            # Smoke test: this nerfed deck (mpsi=16, delta_m=0) is for "does it run / no NaN",
            # not numeric regression — the mode-converged physical value is pinned in the
            # regression harness (examples/Solovev_kinetic_calculated_example). Assert only
            # nerfed-grid-robust facts: finite and positive (Solovev is stable).
            et = read(h5["ForceFreeStates/FreeBoundaryStability/eigenmode_energies"])
            @test isfinite(real(et[1]))
            @test real(et[1]) > 0
            # Kinetic runs never populate the asymptotic ca coefficients; the writer must
            # emit deterministic zero-extent sentinels, not uninitialized memory.
            @test isempty(read(h5["SingularSurfaces/ca_left"]))
            @test isempty(read(h5["SingularSurfaces/ca_right"]))
        end
        rm(joinpath(ex3, "gpec.h5"); force=true)
        true
    end

    ex4 = joinpath(@__DIR__, "test_data", "regression_solovev_kinetic_multi_n")
    @info "Running Solovev kinetic multi-n example (kinetic_factor=1e-9, nn_low=1, nn_high=2)"
    @test begin
        GeneralizedPerturbedEquilibrium.main([ex4])
        h5open(joinpath(ex4, "gpec.h5"), "r") do h5
            # Smoke test only (nerfed mpsi=16, delta_m=0 deck): runs without faulting and
            # produces a finite leading eigenvalue. Numeric regression tracking lives in the
            # harness on the mode-converged deck, not here — et[1] is a near-marginal,
            # ill-conditioned, FP-reassociation-sensitive quantity on this grid (sign not even
            # robust across platforms), so no value is pinned.
            et = read(h5["ForceFreeStates/FreeBoundaryStability/eigenmode_energies"])
            @test isfinite(real(et[1]))
        end
        rm(joinpath(ex4, "gpec.h5"); force=true)
        true
    end

    ex5 = joinpath(@__DIR__, "test_data", "regression_solovev_kinetic_calculated")
    @info "Running Solovev self-consistent kinetic-MHD example (kinetic_source=calculated)"
    @test begin
        GeneralizedPerturbedEquilibrium.main([ex5])
        h5open(joinpath(ex5, "gpec.h5"), "r") do h5
            et = read(h5["ForceFreeStates/FreeBoundaryStability/eigenmode_energies"])
            # Smoke test (nerfed mpsi=16, delta_m=0 deck): exercises the full self-consistent
            # KF→FFS kinetic-MHD path end-to-end. NO numeric value is pinned here — the prior
            # imag(et[1]) ≈ -0.711 rtol=0.08 pin was platform-fragile (failed on macOS aarch64
            # at -0.856; issue #273). Physical regression is tracked in the harness on the
            # mode-converged deck (examples/Solovev_kinetic_calculated_example, mpert=32).
            # Assert only nerfed-grid-robust physics: finite, positive total energy,
            # negative kinetic damping (imag sign is robust across configs; only its
            # magnitude is FP-sensitive).
            @test isfinite(real(et[1]))
            @test isfinite(imag(et[1]))
            @test real(et[1]) > 0
            @test imag(et[1]) < 0
        end
        rm(joinpath(ex5, "gpec.h5"); force=true)
        true
    end

    ex6 = joinpath(@__DIR__, "test_data", "regression_solovev_kinetic_nuzero")
    @info "Running Solovev self-consistent kinetic-MHD example (kinetic_source=calculated, nutype=zero)"
    @test begin
        GeneralizedPerturbedEquilibrium.main([ex6])
        h5open(joinpath(ex6, "gpec.h5"), "r") do h5
            et = read(h5["ForceFreeStates/FreeBoundaryStability/eigenmode_energies"])
            # Smoke test (nerfed mpsi=16, delta_m=0 deck): exercises the collisionless
            # (nutype="zero") real-x-space energy-integral path end-to-end — the #281 fix —
            # without faulting/NaN (the bug this guards against). The precise collisionless
            # physics is locked down by the deterministic runtests_kinetic.jl unit tests
            # (tail-pole, ν→0⁺ limit, Ω′<0); the mode-converged eigenvalue is pinned in the
            # harness (calculated example deck run collisionless via override). No value pinned here — only
            # finite, positive total energy, negative resonant (Landau) damping.
            @test isfinite(real(et[1]))
            @test isfinite(imag(et[1]))
            @test real(et[1]) > 0
            @test imag(et[1]) < 0
        end
        rm(joinpath(ex6, "gpec.h5"); force=true)
        true
    end
end
