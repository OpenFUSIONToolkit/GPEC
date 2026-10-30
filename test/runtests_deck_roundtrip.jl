using TOML

# The deck a run records is the inverse of the parser: parsing it gives back every problem, algorithm and control the
# original deck gave, for every example and test-fixture deck.
@testset "deck round trip" begin
    GPEC = GeneralizedPerturbedEquilibrium
    # Structural equality, NaN-aware, through dictionaries, arrays and the fields of structs.
    same(a, b) = isequal(a, b)
    same(a::AbstractDict, b::AbstractDict) = keys(a) == keys(b) && all(same(a[k], b[k]) for k in keys(a))
    same(a::AbstractArray, b::AbstractArray) = size(a) == size(b) && all(same(x, y) for (x, y) in zip(a, b))
    same(a::NamedTuple, b::NamedTuple) = keys(a) == keys(b) && all(same(a[k], b[k]) for k in keys(a))
    function same(a::T, b::T) where {T}
        isstructtype(T) && !(a isa Number || a isa AbstractString || a isa Symbol) || return isequal(a, b)
        return all(same(getfield(a, f), getfield(b, f)) for f in fieldnames(T))
    end

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
            @test same(norun(GPEC.parse_deck(again, dir)), norun(deck))
            @test GPEC.parse_deck(again, dir).run.verbose == deck.run.verbose
            @test same(GPEC.equilibrium_config(again, dir), config)
            @test same(GPEC.analytic_config(again), analytic)
        end
    end
end
