# Shared display contract for the result types (src/Display.jl): a one-line compact form for
# containers and a short text/plain summary that summarizes arrays instead of printing them.
# Included by the test files that build each kind of result.

function _check_display(x)
    compact = repr(x)
    plain = repr(MIME"text/plain"(), x)
    @test !occursin('\n', compact) && length(compact) ≤ 160
    @test count('\n', plain) ≤ 15 && length(plain) ≤ 2000
    return plain
end

# Each summarized type's fields are split exactly between those its summary shows and those it omits.
function _check_display_ledger()
    for (T, f) in GeneralizedPerturbedEquilibrium._DISPLAY_FIELDS
        unclassified = setdiff(fieldnames(T), f.shown, f.omitted)
        stale = setdiff(union(f.shown, f.omitted), fieldnames(T))
        @test isempty(unclassified) && isempty(stale) && isdisjoint(f.shown, f.omitted)
        isempty(unclassified) || @error "$(nameof(T)): classify the new fields in _DISPLAY_FIELDS (src/Display.jl)" unclassified
        isempty(stale) || @error "$(nameof(T)): _DISPLAY_FIELDS lists fields the type no longer has" stale
    end
end

# Every stage result `main` returned has its own summary, recorded in the ledger.
function _check_main_display(ret)
    GPEC = GeneralizedPerturbedEquilibrium
    for x in ret
        x === nothing && continue
        T = typeof(x)
        @test parentmodule(which(show, Tuple{IO,MIME"text/plain",T})) === GPEC
        @test haskey(GPEC._DISPLAY_FIELDS, Base.typename(T).wrapper)
        _check_display(x)
    end
    @test !occursin('\n', repr(ret))
end
