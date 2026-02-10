# Test script to build documentation locally
using Pkg

# Activate and instantiate the main project
Pkg.activate(".")
Pkg.instantiate()

# Build the package
Pkg.build()

# Activate the docs environment (don't instantiate yet - JPEC is not registered)
Pkg.activate("docs")

# Add the local package to docs environment BEFORE instantiating
Pkg.develop(PackageSpec(; path="."))

# Now instantiate to get Documenter and other deps
Pkg.instantiate()

# Build the documentation
include("docs/make.jl")
