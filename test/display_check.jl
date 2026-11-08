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

# Every stage result `main` returned has its own summary from this package.
function _check_main_display(ret)
    GPEC = GeneralizedPerturbedEquilibrium
    for x in ret
        x === nothing && continue
        @test parentmodule(which(show, Tuple{IO,MIME"text/plain",typeof(x)})) === GPEC
        _check_display(x)
    end
    @test !occursin('\n', repr(ret))
end
