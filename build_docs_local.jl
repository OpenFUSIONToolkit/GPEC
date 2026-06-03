# Test script to build documentation locally
using Pkg

# Activate and instantiate the main project
Pkg.activate(".")
Pkg.instantiate()

# Activate the docs environment
Pkg.activate("docs")

# Add the local package to docs environment first
Pkg.develop(PackageSpec(; path="."))

# Now instantiate to get other dependencies
Pkg.instantiate()

# Build the documentation
include("docs/make.jl")
 