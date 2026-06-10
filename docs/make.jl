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
        "Workflow" => "workflow.md",
        "API Reference" => [
            "Vacuum" => "vacuum.md",
            "Equilibrium" => "equilibrium.md",
            "Stability Analysis" => "stability.md",
            "KineticForces" => "kinetic_forces.md",
            "Forcing Terms" => "forcing_terms.md",
            "Perturbed Equilibrium" => "perturbed_equilibrium.md",
            "Inner Layer" => "inner_layer.md",
            "Analysis" => "analysis.md",
            "Utilities" => "utilities.md"
       ],
        "Citations" => "citations.md",
        "Developer Notes" => "developer_notes.md",
    ],
    checkdocs=:exports
)

deploydocs(;
    repo="github.com/OpenFUSIONToolkit/GPEC.git",
    branch="gh-pages",
    devbranch="develop",
    push_preview=true
)
