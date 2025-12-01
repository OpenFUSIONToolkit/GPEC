"""
Minimal energy integration stub for tests.
Defines tintgrl_lsode and tintgrl_grid placeholders used in pentrc_interface.jl
"""
function tintgrl_lsode(psilims, nn, nl, zi, mi, wdfac, divxfac, electron, method)
    return complex(0.0)
end

function tintgrl_grid(mode, psilims, nn, nl, zi, mi, wdfac, divxfac, electron, method)
    return complex(0.0)
end
