# JPEC.jl Documentation

This directory contains the documentation source files for JPEC.jl.

## Building Documentation Locally

To build the documentation locally:

```bash
julia build_docs_local.jl
```

Or step by step:

```bash
# From the root directory
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.build()'
julia --project=docs -e 'using Pkg; Pkg.instantiate(); Pkg.develop(PackageSpec(path="."))'
julia --project=docs docs/make.jl
```

## GitHub Pages Deployment

The documentation is automatically built and deployed to GitHub Pages via GitHub Actions when:

1. Pushing to the `main` branch
2. Creating a tag
3. Opening a pull request (for preview)

## Structure

- `docs/src/`: Markdown source files
- `docs/make.jl`: Documenter.jl build script
- `docs/Project.toml`: Documentation dependencies
- `docs/src/examples/`: Examples and tutorials

## Requirements

- Julia 1.11+
- Documenter.jl
- All JPEC.jl dependencies (see main Project.toml)
