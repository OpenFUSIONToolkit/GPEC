using TOML
using HDF5

# Structural equality, NaN-aware, through dictionaries, arrays and the fields of structs.
deck_same(a, b) = isequal(a, b)
deck_same(a::AbstractDict, b::AbstractDict) = keys(a) == keys(b) && all(deck_same(a[k], b[k]) for k in keys(a))
deck_same(a::AbstractArray, b::AbstractArray) = size(a) == size(b) && all(deck_same(x, y) for (x, y) in zip(a, b))
deck_same(a::NamedTuple, b::NamedTuple) = keys(a) == keys(b) && all(deck_same(a[k], b[k]) for k in keys(a))
function deck_same(a::T, b::T) where {T}
    isstructtype(T) && !(a isa Number || a isa AbstractString || a isa Symbol) || return isequal(a, b)
    return all(deck_same(getfield(a, f), getfield(b, f)) for f in fieldnames(T))
end

# The deck a run records is the inverse of the parser: parsing it gives back every problem, algorithm and control the
# original deck gave, for every example and test-fixture deck.
@testset "deck round trip" begin
    GPEC = GeneralizedPerturbedEquilibrium

    roots = (joinpath(@__DIR__, "..", "examples"), joinpath(@__DIR__, "test_data"))
    decks = [dir for root in roots for dir in readdir(root; join=true) if isfile(joinpath(dir, "gpec.toml"))]
    @test length(decks) >= 20
    for dir in decks
        @testset "$(basename(dir))" begin
            inputs = TOML.parsefile(joinpath(dir, "gpec.toml"))
            deck = GPEC.parse_deck(inputs, dir)
            config, analytic = GPEC.equilibrium_config(inputs, dir), GPEC.analytic_config(inputs)
            recorded = GPEC.deck_inputs(deck, config, analytic)
            # The record is a deck: it prints as TOML and parses back to the same objects.
            again = TOML.parse(sprint(TOML.print, recorded))
            norun(d) = Base.structdiff(d, (; run=nothing))
            @test deck_same(norun(GPEC.parse_deck(again, dir)), norun(deck))
            @test GPEC.parse_deck(again, dir).run.verbose == deck.run.verbose
            @test deck_same(GPEC.equilibrium_config(again, dir), config)
            @test deck_same(GPEC.analytic_config(again), analytic)
        end
    end
end

# A saved run records the deck that reproduces it: a deck run and a script run of the same workflow record the same
# deck, and either replays from its gpec.h5 alone, a forcing no deck states included.
@testset "runs record the deck that replays them" begin
    GPEC = GeneralizedPerturbedEquilibrium
    recorded(path) = h5open(f -> TOML.parse(read(f["Input/gpec_toml_raw"])), path)
    stages(deck, dir) = Base.structdiff(GPEC.parse_deck(deck, dir), (; run=nothing))
    mktempdir() do dir
        example = joinpath(@__DIR__, "..", "examples", "Solovev_ideal_example")
        foreach(name -> cp(joinpath(example, name), joinpath(dir, name)), readdir(example))
        inputs = TOML.parsefile(joinpath(dir, "gpec.toml"))
        inputs["Run"]["verbose"] = false
        open(io -> TOML.print(io, inputs), joinpath(dir, "gpec.toml"), "w")
        deck_run = GPEC.main([dir])
        deck_h5 = joinpath(dir, "gpec.h5")
        # The record states every default explicitly, and replaying it reproduces the run.
        rerun = GPEC.main([deck_h5, "--output-dir", joinpath(dir, "rerun")])
        @test rerun.ffs.free_boundary.et == deck_run.ffs.free_boundary.et
        @test rerun.pe.forcing_b == deck_run.pe.forcing_b
        @test rerun.local_stability.D_I == deck_run.local_stability.D_I

        # The same workflow as a script records the same deck.
        deck = GPEC.parse_deck(inputs, dir)
        equil = PlasmaEquilibrium(GPEC.equilibrium_config(inputs, dir), GPEC.analytic_config(inputs))
        ffs = solve(ForceBalanceProblem(equil; deck.force_balance.keywords...), deck.force_balance.algorithm)
        ls = solve(LocalStabilityProblem(ffs.equil; deck.local_stability.problem...), deck.local_stability.algorithm)
        resp = plasma_response(ffs; deck.perturbed_equilibrium.response...)
        pe = perturbed_equilibrium(resp, RMPField(deck.forcing); deck.perturbed_equilibrium.drive...)
        script_h5 = joinpath(dir, "script.h5")
        foreach(r -> save_to_hdf5(script_h5, r), (ffs, ls, pe))
        @test deck_same(stages(recorded(script_h5), dir), stages(recorded(deck_h5), dir))

        # The script run replays from its gpec.h5 alone.
        replay = GPEC.main([script_h5, "--output-dir", joinpath(dir, "replay")])
        @test replay.ffs.free_boundary.et == ffs.free_boundary.et
        @test replay.pe.forcing_b == pe.forcing_b

        # A weighted field has no [ForcingTerms]: the replay drives the recorded control-surface modes.
        doubled = perturbed_equilibrium(resp, 2 * RMPField(deck.forcing); deck.perturbed_equilibrium.drive...)
        weighted_h5 = joinpath(dir, "weighted.h5")
        foreach(r -> save_to_hdf5(weighted_h5, r), (ffs, doubled))
        @test recorded(weighted_h5)["ForcingTerms"]["forcing_data_file"] == "Input/RawInputs/ForcingTerms"
        again = GPEC.main([weighted_h5, "--output-dir", joinpath(dir, "replay_weighted")])
        @test [m.amplitude for m in again.pe.forcing_modes] == [m.amplitude for m in doubled.forcing_modes]
        @test again.pe.forcing_b == doubled.forcing_b
    end
end
