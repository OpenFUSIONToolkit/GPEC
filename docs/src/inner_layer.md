# Inner Layer Module

The InnerLayer module provides scaffolding for resistive inner-layer models used
in matched asymptotic expansions for resistive MHD stability analysis. It
includes the GGJ (Glasser–Greene–Johnson) single-fluid model, with a backward
stable-shoot solver and a singular-Galerkin Hermite finite-element solver, the
latter built on the shared, model-agnostic `WasowGalerkin` engine.

## InnerLayer

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.InnerLayer]
```

## WasowGalerkin engine

The `WasowGalerkin` engine holds the model-agnostic singular-Galerkin FEM and
Wasow asymptotic-basis machinery. A physics model supplies a `SystemSpec`
(sizes plus coefficient/asymptotic/eigenbasis/exponent/parity callbacks); the
engine contains no model-specific physics or dimensions. The GGJ model is its
first client.

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.InnerLayer.WasowGalerkin]
```

## GGJ

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.InnerLayer.GGJ]
```
