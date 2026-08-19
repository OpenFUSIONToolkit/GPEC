# Questions for the morning

1. **PR #408 review-package sharing worked?** (You flipped both; reviewers should confirm access.)
2. **Certified-grid default**: kinetic_grid_tol is committed default-OFF (0). The B2 sweep will
   inform a recommended value (likely 1e-3); do you want it ON by default in the kinetic example
   decks, or opt-in only until it has survived a real DIII-D kinetic workflow?
3. **Kinetic reference truth**: Solovev-calculated et[1] moves ~1e-3 between m64 and m256 and the
   ladder is non-monotonic — there is no converged reference in the examples. Is there a case you
   trust as kinetic ground truth (PENTRC benchmark?) worth wiring in before we claim accuracy wins?
4. **G/H aliasing**: Kinetic/G vs Ideal/G/H statistics are identical in the h5 at every size —
   expected (small corrections on a shared large scale), or worth a look?
5. **Scope check**: certification currently applies only to kinetic_source="calculated". The
   "fixed" test-matrix path stays on the equilibrium grid. OK?
6. **q=5 Δ′ non-convergence** (out of scope per your earlier call) — still parked; unchanged.

7. **Kinetic harness deltas vs develop are LARGE and (pending attribution) mostly inherited from
   #398's Solovev geometry fix**: et[1] shifts of 1.7-2.5%, Im parts up to 23%, NTV torque
   magnitude change, and — striking — **NTV ψ-quadrature evaluations 840 → 60**: the torque
   quadrature appears to have been spending 14× the evaluations resolving geometry noise. An
   attribution harness (vs the #398 branch instead of develop) is running/complete — see
   attr_regress.log. Physics judgment needed: are the kinetic-case changes acceptable as
   "the geometry-noise fix propagating into resonance-sensitive quantities" (my reading), and
   should the kinetic harness baselines simply be re-baselined once #398 lands?
8. **Certified-grid PR**: B1 is committed on the #408 branch (default OFF, knob-off bit-identical),
   but arguably belongs in its own stacked PR per the plan ("kinetic work lands as its own PR
   stacked after Phase A's"). I did NOT split it out overnight — say the word and I'll
   cherry-pick 2225813fc onto a new branch stacked on #408 and open the PR with its review
   package, or keep it in #408 if you prefer fewer moving pieces.
