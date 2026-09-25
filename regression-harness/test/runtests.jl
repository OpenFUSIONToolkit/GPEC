# Unit tests for the golden-value comparison logic (src/golden.jl), on synthetic data only —
# no GPEC runs. Run against the harness project, not the GPEC package:
#
#   julia --project=regression-harness regression-harness/test/runtests.jl
#
using Test
using Dates, HDF5, JSON, Printf, SHA, SQLite, Tables, TOML

const HARNESS_DIR = abspath(joinpath(@__DIR__, ".."))
const REPO_ROOT = abspath(joinpath(HARNESS_DIR, ".."))
const DEFAULT_DB_PATH = joinpath(mktempdir(), "test_cache.sqlite")
const CASES_DIR = joinpath(HARNESS_DIR, "cases")
# Redirected to a temp dir so tests can write and corrupt golden files freely without ever
# touching the tracked pins under regression-harness/golden/.
const GOLDEN_DIR = mktempdir()

for f in ("types.jl", "env.jl", "config.jl", "database.jl", "utils.jl", "extractor.jl", "runner.jl", "reporter.jl", "golden.jl")
    include(joinpath(HARNESS_DIR, "src", f))
end

qspec(name; type="real_scalar", extract="value", label=name, noise=0.0, order=1, class="physics_converged") =
    QuantitySpec(name, "", type, extract, label, noise, order, class)

golden_val(name; value_type="real", value_real=nothing, value_int=nothing, value_text=nothing,
    rtol=1e-6, atol=0.0, class="physics_converged", basis="class-default (provisional)", drift=NaN, spread=NaN, at="") =
    GoldenValue(name, value_type, value_real, value_int, value_text, rtol, atol, class, basis, drift, spread, at)

qtuple(; value_real=nothing, value_int=nothing, value_text=nothing, value_type="real", label="q") =
    (label=label, value_real=value_real, value_int=value_int, value_text=value_text, value_type=value_type, noise_threshold=0.0)

extracted_q(name; value_real=nothing, value_int=nothing, value_text=nothing, value_type="real") =
    ExtractedQuantity(name, name, value_real, value_int, value_text, value_type, 0.0)

@testset "golden comparison logic" begin

    @testset "scalar real comparison" begin
        g = golden_val("x"; value_real=1.0, rtol=1e-6)
        passed, dev, detail = compare_to_golden(qtuple(; value_real=1.0 + 5e-7), g)
        @test passed
        @test dev ≈ 5e-7 rtol = 1e-3
        @test isempty(detail)

        passed, dev, detail = compare_to_golden(qtuple(; value_real=1.0 + 1e-5), g)
        @test !passed
        @test dev ≈ 1e-5 rtol = 1e-3
        @test !isempty(detail)

        # Zero golden value: rtol contributes nothing, deviation is reported absolute.
        g0 = golden_val("x"; value_real=0.0, rtol=1e-6, atol=0.0)
        passed, dev, _ = compare_to_golden(qtuple(; value_real=1e-12), g0)
        @test !passed
        @test dev ≈ 1e-12
        g0a = golden_val("x"; value_real=0.0, rtol=1e-6, atol=1e-9)
        @test compare_to_golden(qtuple(; value_real=1e-12), g0a)[1]

        # Non-gating classes are never judged, even on a zero golden value where Inf*0 = NaN.
        gd = golden_val("x"; value_real=0.0, rtol=Inf, atol=Inf, class="diagnostic")
        @test compare_to_golden(qtuple(; value_real=42.0), gd)[1]
        # Finite tolerances on a non-gating class do not make it gate either.
        @test compare_to_golden(qtuple(; value_real=42.0), golden_val("x"; value_real=1.0, class="unconverged"))[1]

        @test !compare_to_golden(qtuple(; value_real=nothing), g)[1]
        # A NaN run value fails a finite gold rather than slipping through a false comparison.
        passed, dev, _ = compare_to_golden(qtuple(; value_real=NaN), g)
        @test !passed && isnan(dev)
    end

    @testset "integer and token comparison" begin
        # Topological integers gate on exact equality no matter how loose the rtol.
        gt = golden_val("n"; value_type="integer", value_int=5, rtol=10.0, class="topological")
        @test compare_to_golden(qtuple(; value_int=5, value_type="integer"), gt)[1]
        @test !compare_to_golden(qtuple(; value_int=6, value_type="integer"), gt)[1]

        # Non-topological integers use the tolerance like a real.
        gi = golden_val("n"; value_type="integer", value_int=100, rtol=0.05, class="physics_converged")
        @test compare_to_golden(qtuple(; value_int=103, value_type="integer"), gi)[1]
        @test !compare_to_golden(qtuple(; value_int=110, value_type="integer"), gi)[1]

        # Tokens are discrete: deviation 0.0 on match, 1.0 (not NaN) on mismatch so it sorts as
        # a real failure.
        gk = golden_val("integ"; value_type="token", value_text="riccati", class="topological")
        passed, dev, _ = compare_to_golden(qtuple(; value_text="riccati", value_type="token"), gk)
        @test passed && dev == 0.0
        passed, dev, detail = compare_to_golden(qtuple(; value_text="galerkin", value_type="token"), gk)
        @test !passed && dev == 1.0
        @test occursin("riccati", detail) && occursin("galerkin", detail)
    end

    @testset "array worst-element path" begin
        g = golden_val("v"; value_type="json_array", value_text="[1.0,2.0,4.0]", rtol=1e-6)

        passed, dev, detail = compare_to_golden(qtuple(; value_text="[1.0000000001,2.0,4.0]", value_type="json_array"), g)
        @test passed
        @test dev ≈ 1e-10 rtol = 1e-3
        @test isempty(detail)

        # One element out of tolerance fails the whole array and the detail names it.
        passed, dev, detail = compare_to_golden(qtuple(; value_text="[1.0,2.002,4.0]", value_type="json_array"), g)
        @test !passed
        @test dev ≈ 1e-3 rtol = 1e-3
        @test occursin("worst element 2", detail)

        # The reported deviation is the worst RELATIVE element, not the first failing one.
        passed, dev, detail = compare_to_golden(qtuple(; value_text="[1.001,2.0,4.02]", value_type="json_array"), g)
        @test !passed
        @test dev ≈ 5e-3 rtol = 1e-3
        @test occursin("worst element 3", detail)

        # A zero golden element is judged absolutely; atol=0 makes any movement on it fail.
        gz = golden_val("v"; value_type="json_array", value_text="[0.0,1.0]", rtol=1e-6, atol=0.0)
        @test !compare_to_golden(qtuple(; value_text="[1.0e-9,1.0]", value_type="json_array"), gz)[1]
        gza = golden_val("v"; value_type="json_array", value_text="[0.0,1.0]", rtol=1e-6, atol=1e-8)
        @test compare_to_golden(qtuple(; value_text="[1.0e-9,1.0]", value_type="json_array"), gza)[1]

        # Non-finite tolerances never judge, including zero golden elements (the Inf*0 = NaN guard).
        gdz = golden_val("v"; value_type="json_array", value_text="[0.0,1.0]", rtol=Inf, atol=Inf, class="diagnostic")
        @test compare_to_golden(qtuple(; value_text="[5.0,99.0]", value_type="json_array"), gdz)[1]

        # Complex encoding: [re, im] pairs go through the same worst-element machinery.
        gc = golden_val("v"; value_type="json_array", value_text="[[1.0,0.5],[2.0,-1.0]]", rtol=1e-6)
        @test compare_to_golden(qtuple(; value_text="[[1.0,0.5],[2.0,-1.0]]", value_type="json_array"), gc)[1]
        passed, _, detail = compare_to_golden(qtuple(; value_text="[[1.0,0.5],[2.0,-1.01]]", value_type="json_array"), gc)
        @test !passed
        @test occursin("worst element 2", detail)

        passed, _, detail = compare_to_golden(qtuple(; value_text="[1.0,2.0]", value_type="json_array"), g)
        @test !passed
        @test occursin("length", detail)

        # A NaN element fails and is the element the detail names; NaN on both sides is "no result"
        # in both runs and matches.
        passed, dev, detail = compare_to_golden(qtuple(; value_text="[1.0,NaN,4.0]", value_type="json_array"), g)
        @test !passed && isinf(dev)
        @test occursin("worst element 2", detail)
        gn = golden_val("v"; value_type="json_array", value_text="[1.0,NaN]", rtol=1e-6)
        @test compare_to_golden(qtuple(; value_text="[1.0,NaN]", value_type="json_array"), gn)[1]

        # A nested row that changed length fails instead of being truncated by zip.
        gr = golden_val("m"; value_type="json_array", value_text="[[1.0,2.0],[3.0,4.0]]", rtol=1e-6)
        @test !compare_to_golden(qtuple(; value_text="[[1.0,2.0],[3.0]]", value_type="json_array"), gr)[1]
    end

    @testset "type-change handling" begin
        g = golden_val("x"; value_real=1.0)
        passed, dev, detail = compare_to_golden(qtuple(; value_text="[1.0]", value_type="json_array"), g)
        @test !passed && isnan(dev)
        @test occursin("type changed", detail)

        # A value_type change through build_golden_values resets class and tolerances to
        # provisional: measurements made under the old type must not carry over.
        spec = qspec("x")
        prior = Dict("x" => golden_val("x"; value_type="integer", value_int=5, rtol=0.0,
            class="topological", basis="measured", drift=1e-8, spread=1e-7, at="mpsi=1024"))
        vals = @test_logs (:warn, r"value_type changed") build_golden_values(
            [extracted_q("x"; value_real=5.0)], [spec], prior)
        @test vals["x"].class == "physics_converged"
        @test vals["x"].rtol == CLASS_DEFAULT_TOLERANCE["physics_converged"].rtol
        @test vals["x"].tolerance_basis == "class-default (provisional)"
        @test isnan(vals["x"].plateau_drift) && isnan(vals["x"].platform_spread)

        # Same type, moved within the old tolerance: the prior's measurements carry forward unchanged.
        prior2 = Dict("x" => golden_val("x"; value_real=1.0, rtol=3e-7, atol=1e-12,
            basis="measured", drift=1e-8, spread=2e-7, at="mpsi=1024"))
        vals2 = build_golden_values([extracted_q("x"; value_real=1.0 + 1e-7)], [spec], prior2)
        @test vals2["x"].value_real == 1.0 + 1e-7
        # An absolute floor on a nonzero value is not a legal gate, so it is dropped rather than carried.
        @test vals2["x"].rtol == 3e-7 && vals2["x"].atol == 0.0
        @test vals2["x"].tolerance_basis == "measured"
        @test vals2["x"].plateau_drift == 1e-8 && vals2["x"].platform_spread == 2e-7

        # A move beyond the old tolerance drops the evidence, and the tolerance never loosens.
        vals5 = @test_logs (:warn, r"beyond its old tolerance") build_golden_values(
            [extracted_q("x"; value_real=1.5)], [spec], prior2)
        @test vals5["x"].value_real == 1.5
        @test vals5["x"].rtol == 3e-7 && vals5["x"].atol == 0.0
        @test vals5["x"].tolerance_basis == "provisional" && is_provisional(vals5["x"])
        @test isnan(vals5["x"].plateau_drift) && isnan(vals5["x"].platform_spread) && isempty(vals5["x"].converged_at)

        # A structural zero keeps its absolute floor across a re-pin that stays zero.
        priorz = Dict("z" => golden_val("z"; value_type="json_array", value_text="[0.0,1.0]", atol=1e-10))
        valsz = build_golden_values([extracted_q("z"; value_text="[0.0,1.0000001]", value_type="json_array")], [qspec("z"; type="real_array")], priorz)
        @test valsz["z"].atol == 1e-10
        # A measured tolerance wider than the class default is tightened back to the default.
        prior5 = Dict("x" => golden_val("x"; value_real=1.0, rtol=1e-4, basis="measured", drift=5e-5))
        vals6 = @test_logs (:warn, r"beyond its old tolerance") build_golden_values(
            [extracted_q("x"; value_real=1.5)], [spec], prior5)
        @test vals6["x"].rtol == CLASS_DEFAULT_TOLERANCE["physics_converged"].rtol
        @test vals6["x"].tolerance_basis == "class-default (provisional)"

        # A change of the case-declared class resets measurements, like a type change.
        prior3 = Dict("x" => golden_val("x"; value_real=1.0, rtol=3e-7, basis="measured", drift=1e-8))
        vals4 = @test_logs (:warn, r"class changed") build_golden_values(
            [extracted_q("x"; value_real=1.0)], [qspec("x"; class="equilibrium_scalar")], prior3)
        @test vals4["x"].class == "equilibrium_scalar"
        @test vals4["x"].tolerance_basis == "class-default (provisional)"

        # Missing extractions, checksums, runtimes, and quantities without a spec never become entries.
        vals3 = build_golden_values(
            [extracted_q("gone"; value_type="missing"),
                extracted_q("x"; value_text="abc123", value_type="checksum"),
                extracted_q("rt"; value_real=12.5),
                extracted_q("unspecced"; value_real=1.0)],
            [spec, qspec("rt"; type="runtime", class="diagnostic")], nothing)
        @test isempty(vals3)
    end

    @testset "load-validation refusals" begin
        write(
            golden_path("bad_class"),
            """
[values.x]
class = "definitely_not_a_class"
rtol = 1.0e-6
value = 1.0
"""
        )
        @test_throws ErrorException load_golden("bad_class")

        # A gating entry with no finite rtol is malformed whether it was hand-edited or merged.
        write(
            golden_path("no_rtol"),
            """
[values.x]
class = "physics_converged"
value = 1.0
"""
        )
        @test_throws ErrorException load_golden("no_rtol")

        # An rtol below the recorded platform spread is a gate no second platform can pass;
        # enforced on every load so a hand edit cannot ship what save_golden refused to write.
        write(
            golden_path("too_tight"),
            """
[values.x]
class = "physics_converged"
rtol = 1.0e-9
platform_spread = 1.0e-6
value = 1.0
"""
        )
        @test_throws ErrorException load_golden("too_tight")

        # A non-finite atol on a gating class would make every comparison pass.
        for bad in ("inf", "nan")
            write(
                golden_path("atol_$bad"),
                """
[values.x]
class = "physics_converged"
rtol = 1.0e-6
atol = $bad
value = 1.0
"""
            )
            @test_throws ErrorException load_golden("atol_$bad")
        end

        # An rtol below the measured plateau drift is a gate the converged answer itself fails.
        write(
            golden_path("below_drift"),
            """
[values.x]
class = "physics_converged"
rtol = 1.0e-9
plateau_drift = 1.0e-7
value = 1.0
"""
        )
        @test_throws ErrorException load_golden("below_drift")

        # atol on a gating entry is only for structural zeros.
        write(
            golden_path("atol_nonzero"),
            """
[values.x]
class = "physics_converged"
rtol = 1.0e-6
atol = 1.0e-12
value = 1.0
"""
        )
        @test_throws ErrorException load_golden("atol_nonzero")
        write(
            golden_path("atol_zeros"),
            """
[values.x]
class = "physics_converged"
rtol = 1.0e-6
atol = 1.0e-12
value = 0.0

[values.v]
class = "physics_converged"
value_type = "json_array"
value_text = "[1.0,0.0]"
rtol = 1.0e-6
atol = 1.0e-12

[values.c]
class = "physics_converged"
value_type = "json_array"
value_text = "[[1.0,0.0],[0.0,0.0]]"
rtol = 1.0e-6
atol = 1.0e-12
"""
        )
        @test length(load_golden("atol_zeros").values) == 3
        # A complex pair with only one zero component is not a structural zero.
        @test !has_exact_zero(golden_val("c"; value_type="json_array", value_text="[[1.0,0.0],[2.0,0.5]]"))
        @test has_exact_zero(golden_val("n"; value_type="integer", value_int=0))
        @test_throws ErrorException save_golden(GoldenMeta("atol_save", 1, "", "", "", "", "", "", 1, 1),
            Dict("x" => golden_val("x"; value_real=2.0, atol=1e-9)))

        # "measured" must be backed by a recorded measurement.
        write(
            golden_path("unbacked"),
            """
[values.x]
class = "physics_converged"
tolerance_basis = "measured"
rtol = 1.0e-6
value = 1.0
"""
        )
        @test_throws ErrorException load_golden("unbacked")

        # Non-gating classes are exempt: they are recorded, never judged.
        write(
            golden_path("diag_inf"),
            """
[values.rt]
class = "diagnostic"
rtol = inf
atol = inf
value = 12.5
"""
        )
        loaded = load_golden("diag_inf")
        @test loaded !== nothing && !is_gating(loaded.values["rt"])

        @test load_golden("no_such_case") === nothing
        @test !has_golden("no_such_case")
    end

    @testset "save_golden refusals and round trip" begin
        meta = GoldenMeta("rt_case", 1, "2026-08-31", "deadbeef", "unit test", "1.11.6", "arm64", "abc", 4, 4)

        @test_throws ErrorException save_golden(meta, Dict("empty" => golden_val("empty")))
        @test_throws ErrorException save_golden(meta,
            Dict("tight" => golden_val("tight"; value_real=1.0, rtol=1e-9, spread=1e-6)))
        @test_throws ErrorException save_golden(meta,
            Dict("open" => golden_val("open"; value_real=1.0, rtol=1e-6, atol=Inf, spread=1e-7)))

        vals = Dict(
            "a" => golden_val("a"; value_real=2.5, rtol=1e-6, drift=1e-8, spread=1e-7, at="mpsi=512"),
            "n" => golden_val("n"; value_type="integer", value_int=7, rtol=0.0, class="topological"),
            "v" => golden_val("v"; value_type="json_array", value_text="[1.0,2.0]", rtol=1e-6))
        save_golden(meta, vals)
        back = load_golden("rt_case")
        @test back.meta.commit == "deadbeef" && back.meta.golden_version == 1
        @test back.values["a"].value_real == 2.5
        @test back.values["a"].rtol == 1e-6
        @test back.values["a"].plateau_drift == 1e-8 && back.values["a"].platform_spread == 1e-7
        @test back.values["n"].value_int == 7 && back.values["n"].class == "topological"
        @test back.values["v"].value_text == "[1.0,2.0]"
        @test isempty(back.meta.exceeded)

        # Accepted moves beyond the old tolerance round-trip through [meta.exceeded].
        exc = exceeding_changes(Dict("a" => golden_val("a"; value_real=2.5)), Dict("a" => golden_val("a"; value_real=2.6)), 1)
        meta2 = GoldenMeta("rt_case", 2, "2026-09-25", "deadbeef", "unit test", "1.11.6", "arm64", "abc", 4, 4, exc)
        save_golden(meta2, Dict("a" => golden_val("a"; value_real=2.6)))
        back2 = load_golden("rt_case")
        @test back2.meta.exceeded["a"]["previous"] == 2.5 && back2.meta.exceeded["a"]["previous_version"] == 1
        @test back2.meta.exceeded["a"]["deviation"] ≈ 0.04
        @test occursin("[meta.exceeded.a]", read(golden_path("rt_case"), String))
    end

    @testset "exceeding_changes" begin
        existing = Dict(
            "in" => golden_val("in"; value_real=1.0),
            "out" => golden_val("out"; value_real=1.0),
            "diag" => golden_val("diag"; value_real=1.0, rtol=Inf, atol=Inf, class="diagnostic"),
            "typ" => golden_val("typ"; value_real=1.0),
            "tok" => golden_val("tok"; value_type="token", value_text="riccati", class="topological"))
        values = Dict(
            "in" => golden_val("in"; value_real=1.0 + 1e-7),
            "out" => golden_val("out"; value_real=1.1),
            "diag" => golden_val("diag"; value_real=50.0, rtol=Inf, atol=Inf, class="diagnostic"),
            "typ" => golden_val("typ"; value_type="json_array", value_text="[1.0]"),
            "tok" => golden_val("tok"; value_type="token", value_text="galerkin", class="topological"),
            "new" => golden_val("new"; value_real=3.0))
        exc = exceeding_changes(existing, values, 4)
        @test sort(collect(keys(exc))) == ["out", "tok", "typ"]
        @test exc["out"]["previous"] == 1.0 && exc["out"]["deviation"] ≈ 0.1 && exc["out"]["previous_version"] == 4
        @test exc["tok"]["previous"] == "riccati"
        @test isnan(exc["typ"]["deviation"])
    end

    @testset "update refuses a move beyond the old tolerance" begin
        repo = mktempdir()
        run(`git -C $repo init -q`)
        write(joinpath(repo, "f.txt"), "x\n")
        run(`git -C $repo add -A`)
        run(`git -C $repo -c user.name=t -c user.email=t@t commit -q -m init`)
        sha = String(strip(read(`git -C $repo rev-parse HEAD`, String)))
        db = open_database(joinpath(mktempdir(), "update.sqlite"))
        case = CaseSpec("exceed_case", "synthetic", "", [qspec("a"), qspec("b")], "example", Dict{String,Any}())
        save_golden(GoldenMeta("exceed_case", 3, "2026-09-01", "cafe", "seed", "1.11.6", "arm64", "abc", 4, 4),
            Dict("a" => golden_val("a"; value_real=1.0), "b" => golden_val("b"; value_real=2.0)))
        store_run(db, sha, sha, "", "", "exceed_case", 1.0, [extracted_q("a"; value_real=1.2), extracted_q("b"; value_real=2.0)])
        @test_throws ErrorException redirect_stdout(devnull) do
            update_golden_from_run(db, case, sha, "moved", repo)
        end
        @test load_golden("exceed_case").meta.golden_version == 3
        @test_logs (:warn, r"beyond its old tolerance") match_mode=:any redirect_stdout(devnull) do
            update_golden_from_run(db, case, sha, "moved", repo; accept_exceeding=true)
        end
        back = load_golden("exceed_case")
        @test back.meta.golden_version == 4 && back.values["a"].value_real == 1.2
        @test collect(keys(back.meta.exceeded)) == ["a"]
        @test back.meta.exceeded["a"]["previous"] == 1.0 && back.meta.exceeded["a"]["previous_version"] == 3
        close_database(db)
    end

    @testset "infer_class" begin
        # The declared class is returned as-is, whatever the quantity's name or type suggests.
        for class in TOLERANCE_CLASSES
            @test infer_class(qspec("x"; class=class)) == class
        end
        @test infer_class(qspec("q0"; class="physics_converged")) == "physics_converged"
        @test infer_class(qspec("nstep"; type="int_scalar", class="topological")) == "topological"
        # A missing or unknown class is an error, never a default.
        @test_throws ErrorException infer_class(qspec("x"; class=""))
        @test_throws ErrorException infer_class(qspec("x"; class="gating_optional"))
    end

    @testset "load_case requires a declared class" begin
        dir = mktempdir()
        write_case(body) = (path = joinpath(dir, "c.toml"); write(path, "[case]\nname = \"c\"\n\n" * body); path)
        block = "[quantities.a]\nh5path = \"x\"\ntype = \"real_scalar\"\nextract = \"value\"\n"
        @test load_case(write_case(block * "class = \"diagnostic\"\n")).quantities[1].class == "diagnostic"
        err = try
            load_case(write_case(block))
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("c.toml", err) && occursin("'a'", err) && occursin("no `class`", err)
        @test_throws ErrorException load_case(write_case(block * "class = \"gating_optional\"\n"))
        # Every committed case declares a valid class on every quantity.
        @test all(q -> q.class in TOLERANCE_CLASSES, Iterators.flatten(c.quantities for c in values(load_all_cases(CASES_DIR))))
    end

    @testset "is_pinnable" begin
        @test is_pinnable(qspec("a"))
        @test !is_pinnable(qspec("rt"; type="runtime", class="diagnostic"))
        @test !is_pinnable(qspec("h"; extract="checksum"))
    end

    @testset "golden check counting and crash classification" begin
        db = open_database(joinpath(mktempdir(), "check.sqlite"))
        case = CaseSpec("count_case", "synthetic", "",
            [
                qspec("a"),
                qspec("rt"; type="runtime", extract="", class="diagnostic"),
                qspec("c"; type="int_scalar", class="topological"),
                qspec("d"; class="diagnostic"),
                qspec("e"),
                qspec("f"),
                qspec("demoted")], "example", Dict{String,Any}())

        meta = GoldenMeta("count_case", 1, "2026-08-31", "deadbeef", "unit test", "1.11.6", "arm64", "abc", 4, 4)
        save_golden(
            meta,
            Dict(
                "a" => golden_val("a"; value_real=1.0, rtol=1e-6),
                "c" => golden_val("c"; value_type="integer", value_int=5, rtol=0.0, class="topological"),
                "d" => golden_val("d"; value_real=3.0, rtol=Inf, atol=Inf, class="diagnostic"),
                "f" => golden_val("f"; value_real=9.0, rtol=1e-6),
                "demoted" => golden_val("demoted"; value_real=2.0, rtol=Inf, atol=Inf, class="unconverged"),
                "orphan" => golden_val("orphan"; value_real=7.0, rtol=1e-6))
        )

        # a passes, c fails (5 → 6), d moves wildly but is declared diagnostic, e has no golden,
        # f is golden-pinned but missing from the run, demoted is non-gating in the golden file only,
        # orphan has no spec left in the case.
        store_run(
            db,
            "hash1",
            "hash1",
            "",
            "",
            "count_case",
            1.0,
            [
                extracted_q("a"; value_real=1.0 + 1e-8),
                extracted_q("rt"; value_real=33.0),
                extracted_q("c"; value_int=6, value_type="integer"),
                extracted_q("d"; value_real=300.0),
                extracted_q("e"; value_real=1.0),
                extracted_q("demoted"; value_real=50.0)]
        )
        s = report_golden_check(db, case, "hash1")
        @test s.n_pass == 1
        @test s.n_fail == 4         # c mismatch + f missing + class demoted outside the case + orphan
        @test s.n_informational == 1
        @test s.n_untracked == 1    # e only: an unpinned runtime is structurally un-goldenable
        @test s.n_run_failed == 0

        # A crashed run is a crash, not a tolerance failure: n_run_failed, never n_fail.
        store_failed_run(db, "hash2", "hash2", "", "", "count_case", "boom: solver exploded")
        s = report_golden_check(db, case, "hash2")
        @test s.n_run_failed == 1
        @test s.n_fail == 0 && s.n_pass == 0

        # No golden file: every count zero, so the caller's zero-coverage guard (a --check that
        # gated nothing must exit red) can see that nothing was checked.
        nocase = CaseSpec("never_pinned", "synthetic", "", [qspec("a")], "example", Dict{String,Any}())
        s = report_golden_check(db, nocase, "hash1")
        @test s == (n_pass=0, n_fail=0, n_untracked=0, n_informational=0, n_run_failed=0)

        # A golden file holding only non-gating entries gates nothing, and the counts say so.
        diagcase = CaseSpec("diag_only", "synthetic", "", [qspec("d"; class="diagnostic")], "example", Dict{String,Any}())
        save_golden(GoldenMeta("diag_only", 1, "2026-08-31", "deadbeef", "unit test", "1.11.6", "arm64", "abc", 4, 4),
            Dict("d" => golden_val("d"; value_real=3.0, rtol=Inf, atol=Inf, class="diagnostic")))
        store_run(db, "hash3", "hash3", "", "", "diag_only", 1.0, [extracted_q("d"; value_real=30.0)])
        s = report_golden_check(db, diagcase, "hash3")
        @test s.n_pass + s.n_fail == 0 && s.n_informational == 1
        close_database(db)
    end

    @testset "re-pin reporting" begin
        old = golden_val("a"; value_real=1.0, rtol=1e-6)
        @test describe_golden_change(old, golden_val("a"; value_real=1.0, rtol=1e-6)) === nothing
        @test !occursin("EXCEEDS", describe_golden_change(old, golden_val("a"; value_real=1.0 + 1e-7)))
        @test occursin("EXCEEDS", describe_golden_change(old, golden_val("a"; value_real=1.1)))
        # Integer and token moves print their values, not just "changed".
        oi = golden_val("n"; value_type="integer", value_int=5, rtol=0.0, class="topological")
        line = describe_golden_change(oi, golden_val("n"; value_type="integer", value_int=4, rtol=0.0, class="topological"))
        @test occursin("5 → 4", line) && occursin("EXCEEDS", line)
        ok = golden_val("k"; value_type="token", value_text="riccati", class="topological")
        @test occursin("riccati → galerkin", describe_golden_change(ok, golden_val("k"; value_type="token", value_text="galerkin", class="topological")))
        # Moves on a non-gating entry are shown but never flagged.
        od = golden_val("d"; value_real=1.0, rtol=Inf, atol=Inf, class="diagnostic")
        @test !occursin("EXCEEDS", describe_golden_change(od, golden_val("d"; value_real=9.0, rtol=Inf, atol=Inf, class="diagnostic")))
    end

    @testset "update refuses uncommitted source" begin
        repo = mktempdir()
        run(`git -C $repo init -q`)
        mkpath(joinpath(repo, "golden"))
        write(joinpath(repo, "src.jl"), "x = 1\n")
        write(joinpath(repo, "golden", "case.toml"), "a = 1\n")
        run(`git -C $repo add -A`)
        run(`git -C $repo -c user.name=t -c user.email=t@t commit -q -m init`)
        @test isempty(uncommitted_changes(repo, joinpath(repo, "golden")))
        # Rewriting a golden file is the update itself, and untracked files outside the source dirs are not source.
        write(joinpath(repo, "golden", "case.toml"), "a = 2\n")
        write(joinpath(repo, "scratch.md"), "notes\n")
        @test isempty(uncommitted_changes(repo, joinpath(repo, "golden")))
        # An untracked file under a source dir can change the run, so it is refused like a tracked edit.
        for f in ("src/new.jl", "examples/case/gpec.toml", "regression-harness/cases/new_case.toml")
            mkpath(dirname(joinpath(repo, f)))
            write(joinpath(repo, f), "x\n")
            @test occursin(f, uncommitted_changes(repo, joinpath(repo, "golden")))
            rm(joinpath(repo, f))
        end
        @test isempty(uncommitted_changes(repo, joinpath(repo, "golden")))
        write(joinpath(repo, "src.jl"), "x = 2\n")
        @test occursin("src.jl", uncommitted_changes(repo, joinpath(repo, "golden")))
    end
end
