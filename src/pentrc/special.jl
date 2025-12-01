"""
Minimal stub for special.jl used in tests.
Provides small no-op functions referenced by pentrc_interface.jl: set_fymnl, set_ellip
"""
function set_fymnl()
    # no-op placeholder used in tests
    return nothing
end

function set_ellip()
    # no-op placeholder used in tests
    return nothing
end
