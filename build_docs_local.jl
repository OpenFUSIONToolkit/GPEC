# Test script to build documentation locally
using Pkg

# Activate and instantiate the main project
Pkg.activate(".")
Pkg.instantiate()

# Build the package
Pkg.build()

# Activate and instantiate the docs environment
Pkg.activate("docs")
Pkg.instantiate()

# Add the local package to docs environment
Pkg.develop(PackageSpec(; path="."))

# Build the documentation
include("docs/make.jl")
