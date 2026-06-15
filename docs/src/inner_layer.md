# Tearing Module

The `Tearing` module groups the resistive tearing-mode analysis stack:
`InnerLayer` (per-surface inner-layer matching data Δ(Q) for the GGJ and
SLAYER models), `Dispersion` (physics-agnostic complex-plane scan and
contour-intersection root extraction), and `Runner` (user-facing TOML
configuration, profile loading, and HDF5 output).

## InnerLayer

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.InnerLayer]
```

## GGJ

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.InnerLayer.GGJ]
```

## SLAYER

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.InnerLayer.SLAYER]
```

## Dispersion

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.Dispersion]
```

## Runner

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.Runner]
```
