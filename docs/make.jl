using Documenter

# Try to load JPEC package
try
    using JPEC
    @info "Successfully loaded JPEC package"
catch e
    @error "Failed to load JPEC package" exception = e
    # Try to provide helpful debugging info
    using Pkg
    @info "Current project:" Pkg.project().path
    @info "Installed packages:" keys(Pkg.dependencies())
    rethrow()
end

makedocs(;
    sitename="JPEC.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", nothing) == "true",
        canonical="https://OpenFUSIONToolkit.github.io/JPEC/"
    ),
    modules=[JPEC],
    pages=[
        "Home" => "index.md",
        "Setup" => "set_up.md",
        "API Reference" => [
            "Splines" => "splines.md",
            "Vacuum" => "vacuum.md",
            "Equilibrium" => "equilibrium.md"
        ],
        "Examples" => [
            "Spline Examples" => "examples/splines.md",
            "Vacuum Examples" => "examples/vacuum.md",
            "Equilibrium Examples" => "examples/equilibrium.md"
        ]
    ],
    checkdocs=:exports
)

deploydocs(;
    repo="github.com/OpenFUSIONToolkit/JPEC.git",
    branch="gh-pages",
    devbranch="main",
    push_preview=true
)
