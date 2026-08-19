# Fortran GPEC dynamic kinetic-grid machinery — correspondence facts

From a fortran-physics-reviewer consultation (2026-08-18) of ~/Code/gpec_1/gpec_dev, preserved for
the Phase B (certified adaptive kinetic grid) work. The remembered "white-noise" scheme is real but
deterministic: a flat xi spectrum driving LSODE in one-step mode, knots = accepted steps.

| item | location |
|---|---|
| kingridtype knob (default 0) | dcon/dcon_mod.f:106; input/dcon.in:27 (types 1-4 undocumented) |
| dispatch | dcon/dcon.F:235 |
| the five methods (-1/0/1/2/3/4) | dcon/fourfit.F:903-1485 (branches 983/985/1298/1359/1402/1445) |
| flat-xi probe setup (all components 1e-4) | dcon/dcon.F:227-232 |
| flat-spectrum default dbob/divx | pentrc/inputs.f90:869-872 (set_peq at 743-905) |
| adaptive engine (LSODE one-step, itask=5) | pentrc/torque.F90:1181-1366 (tintgrl_lsode) |
| type-4 xi-independent scalar (SVD norms of 6 EL matrices) | pentrc/torque.F90:895-930 |
| matrix harvest from integrator scratch by iwork(11) | pentrc/torque.F90:1324, splines ~1341-1348 |
| static buffer maxsteps x mpert^2 x 6 (~2.4 GB at mpert=50) | tintgrl_lsode locals |
| tolerances (pentrc namelist, defaults disagree with input file) | input/pentrc.in:32-33 vs torque.F90:50 |
| post-hoc shaping kinfac1/kinfac2/ktanh | dcon/dcon_mod.f:107-108 |
| rational layers never needed the grid (singfac Taylor) | dcon/sing.f:489-516 |

Why it died: split-namelist knobs with disagreeing defaults; the GB-scale static buffer; W and T
harvested onto two independent grids; unnormalized error scalar vs absolute tolerance (flat-xi form
~1e-8 against atol_psi 1e-3..1e-4); knot/matrix desync risk; and purely reactive refinement with no
seeding at all -- sharp nu->0 resonances get stepped over exactly when they matter.

Carried into the Julia design: the type-4 idea upgraded to a residual certificate on the TOTAL
matrices; batched certify-refine rounds instead of one-step sequential adaptivity (the Fortran's
parallel method is the non-adaptive one precisely because one-step adaptivity forbids batching);
pre-seeded rationals/resonances; one shared grid; normalized relative tolerance; single documented
control knob.
