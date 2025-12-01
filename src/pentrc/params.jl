module Params
    # ----------------------------------------------------------------------
    # Perturbed Equilibrium Nonambipolar Transport Code
    #
    # Clean Julian rewrite of the Fortran 90 constants module.
    # ----------------------------------------------------------------------

    # -----------------------------
    # Numerical constants
    # -----------------------------
    const INF = floatmax(Float64)

    # -----------------------------
    # Physical constants
    # -----------------------------
    const mp = 1.672_614e-27      # proton mass (kg)
    const me = 9.109_1e-31        # electron mass (kg)
    const e  = 1.602_191_7e-19    # elementary charge (C)
    const eV = e                  # joules per electron-volt

    # -----------------------------
    # Math constants
    # -----------------------------
    const π     = 3.141_592_653_589_793
    const τ     = 2π
    const μ₀    = 4e-7 * π
    const rad2deg = 180 / π
    const deg2rad = π / 180
    const iunit = 1im               # equivalent to Fortran's (0,1)

    # -----------------------------
    # Code configuration
    # -----------------------------
    const Nψ_out    = 30
    const Nλ_out    = 5
    const Nℓ_out    = 3
    const Nmethods  = 18
    const Ngrids    = 3

    # Method labels
    const METHODS = [
        "fgar","tgar","pgar","rlar","clar","fcgl",
        "fwmm","twmm","pwmm","ftmm","ttmm","ptmm",
        "fkmm","tkmm","pkmm","frmm","trmm","prmm",
    ]

    # Grid types
    const GRIDS = [
        "lsode",
        "equil",
        "input",
    ]

    # Human-readable descriptions
    const DOCS = [
        "Full general-aspect-ratio calculation",
        "Trapped particle general-aspect-ratio calculation",
        "Passing particle general-aspect-ratio calculation",
        "Trapped particle large-aspect-ratio calculation",
        "Trapped particle cylindrical large-aspect-ratio calculation",
        "Fluid Chew–Goldberger–Low calculation",
        "Full energy calculation using MXM Euler–Lagrange matrix",
        "Trapped energy calculation using MXM Euler–Lagrange matrix",
        "Passing energy calculation using MXM Euler–Lagrange matrix",
        "Full torque calculation using MXM Euler–Lagrange matrix",
        "Trapped torque calculation using MXM Euler–Lagrange matrix",
        "Passing torque calculation using MXM Euler–Lagrange matrix",
        "Full MXM Euler–Lagrange energy matrix norm calculation",
        "Trapped MXM Euler–Lagrange energy matrix norm calculation",
        "Passing MXM Euler–Lagrange energy matrix norm calculation",
        "Full MXM Euler–Lagrange torque matrix norm calculation",
        "Trapped MXM Euler–Lagrange torque matrix norm calculation",
        "Passing MXM Euler–Lagrange torque matrix norm calculation",
    ]

end # module Params
