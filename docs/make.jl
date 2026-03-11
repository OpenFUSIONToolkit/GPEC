using Documenter

# Try to load GeneralizedPerturbedEquilibrium package
try
    using GeneralizedPerturbedEquilibrium
    @info "Successfully loaded GeneralizedPerturbedEquilibrium package"
catch e
    @error "Failed to load GeneralizedPerturbedEquilibrium package" exception = e
    # Try to provide helpful debugging info
    using Pkg
    @info "Current project:" Pkg.project().path
    @info "Installed packages:" keys(Pkg.dependencies())
    rethrow()
end

makedocs(;
    sitename="GPEC.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", nothing) == "true",
        canonical="https://OpenFUSIONToolkit.github.io/GPEC/"
    ),
    modules=[GeneralizedPerturbedEquilibrium],
    pages=[
        "Home" => "index.md",
        "Setup" => "set_up.md",
        "API Reference" => [
            "Vacuum" => "vacuum.md",
            "Equilibrium" => "equilibrium.md",
            "Utilities" => "utilities.md",
            "Forcing Terms" => "forcing_terms.md",
            "Perturbed Equilibrium" => "perturbed_equilibrium.md"
        ]
    ],
    checkdocs=:exports
)

deploydocs(;
    repo="github.com/OpenFUSIONToolkit/GPEC.git",
    branch="gh-pages",
    devbranch="develop",
    push_preview=true
)
