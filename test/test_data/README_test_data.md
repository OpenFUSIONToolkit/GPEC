Put relevant data for various tests here, most likely outputs from the Fortran code used to validate Julia outputs

# TODO: store as hdf5 files instead?

## TokaMaker_ifile
One TokaMaker (OpenFUSIONToolkit) solve of a DIII-D-like H-mode, written as `g65.geqdsk` (`save_eqdsk`, 65×65)
and as `i33x65.ifile` / `i33x65_single.ifile` (`save_ifile`, 33 surfaces × 65 angles, real*8 / real*4, with the
trailing FF′ and p′ records). Used by the `ldp_i` reader test in `runtests_equil.jl`.
