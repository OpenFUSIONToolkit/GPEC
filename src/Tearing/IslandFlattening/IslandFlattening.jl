# Static pressure flattening across a set of magnetic islands of given full width, with the
# stored energy and pressure each set of islands removes. No time evolution.

module IslandFlattening

using FastGaussQuadrature: gausslegendre

using ...Equilibrium: PlasmaEquilibrium, mu0
using ...ForceFreeStates: _find_rational_surfaces

include("Flattening.jl")

export FlattenedIsland, IslandFlatteningResult, island_flattening, merged_intervals, flattened_pressure

end # module IslandFlattening
