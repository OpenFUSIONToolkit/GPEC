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
        canonical="https://OpenFUSIONToolkit.github.io/GPEC/",
        size_threshold=409_600,      # 400 KiB — stability.md is a large single-page module reference
        size_threshold_warn=204_800  # 200 KiB
    ),
    modules=[GeneralizedPerturbedEquilibrium],
    pages=[
        "Home" => "index.md",
        "Setup" => "set_up.md",
        "Workflow" => "workflow.md",
        "Conventions Reference" => "conventions.md",
        "API Reference" => [
            "Vacuum" => "vacuum.md",
            "Equilibrium" => "equilibrium.md",
            "Stability Analysis" => "stability.md",
            "Galerkin Solver" => "galerkin.md",
            "Ballooning Local Stability" => "ballooning.md",
            "KineticForces" => "kinetic_forces.md",
            "Forcing Terms" => "forcing_terms.md",
            "Perturbed Equilibrium" => "perturbed_equilibrium.md",
            "Inner Layer" => "inner_layer.md",
            "Islands" => "islands.md",
            "Analysis" => "analysis.md",
            "Utilities" => "utilities.md"
        ],
        "Islands" => [
            "Overview" => "islands/index.md",
            "Numerics (as implemented)" => "islands/numerics.md",
            "State dashboard" => "islands/state/STATE.md",
            "Derivations" => [
                "Overview" => "islands/derivations/index.md",
                "ψ̃ amplitude" => "islands/derivations/psi-tilde-amplitude.md",
                "ω̂_D drift frequency" => "islands/derivations/omega-D-drift-frequency.md",
                "Collision operator" => "islands/derivations/collision-operator.md",
                "Electron closure" => "islands/derivations/electron-closure.md",
                "Quasineutrality closure" => "islands/derivations/quasineutrality-closure.md",
                "Δ-moment prefactors" => "islands/derivations/delta-moment-prefactors.md",
                "Passing fraction (draft)" => "islands/derivations/passing-fraction.md"
            ],
            "Paper I — figure contract" => "islands/papers/paper-1/OUTLINE.md",
            "Design documents" => [
                "00 — Roadmap" => "islands/design/00-roadmap.md",
                "01 — Level-0 physics" => "islands/design/01-physics-level0.md",
                "02 — Species and EPs" => "islands/design/02-species-and-eps.md",
                "03 — Architecture" => "islands/design/03-architecture.md",
                "04 — Numerics" => "islands/design/04-numerics.md",
                "05 — Verification ladder" => "islands/design/05-verification.md",
                "06 — Autonomy and tooling" => "islands/design/06-autonomy-and-tooling.md",
                "07 — Documentation and papers" => "islands/design/07-documentation-and-papers.md",
                "08 — Reference library" => "islands/design/08-reference-library.md",
                "09 — Input manifests" => "islands/design/09-input-manifests.md"
            ]
        ],
        "Citations" => "citations.md",
        "Developer Notes" => "developer_notes.md"
    ],
    checkdocs=:exports
)

deploydocs(;
    repo="github.com/OpenFUSIONToolkit/GPEC.git",
    branch="gh-pages",
    devbranch="develop",
    push_preview=true
)
