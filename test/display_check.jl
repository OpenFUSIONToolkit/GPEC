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
