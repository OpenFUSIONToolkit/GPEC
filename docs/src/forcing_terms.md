# ForcingTerms Module

The ForcingTerms module handles external magnetic field perturbation specifications for GPEC.

## Overview

The ForcingTerms module provides:
- `ForcingTermsControl`: Configuration parameters for loading forcing data
- `ForcingMode`: Data structure representing a single (n, m) forcing mode
- `load_forcing_data!`: Function to load forcing data from ASCII or HDF5 files

## API Reference

```@autodocs
Modules = [GeneralizedPerturbedEquilibrium.ForcingTerms]
```
