# Lead Developers

A short list of who to suggest as a reviewer or assignee, and the GitHub handle to use, so that a request like "assign this to Nik" resolves to an account without guesswork. Referenced by [`naming.md`](naming.md) when opening a pull request.

**This is a suggestion list, not a roster.** It names lead developers only — many more people contribute, and their absence here means nothing. It exists mainly so an AI agent has somewhere sensible to start; a human opening a pull request can simply pick from GitHub's own dropdown and does not need this file.

**Focus names Areas, including the ones outside `src/`.** Focus values are the Area names from [`naming.md`](naming.md), so they cover repository maintenance as well as physics modules: `CI` is the workflows under `.github/`, and `Repo` is the cross-cutting plumbing — conventions, templates, the changelog, and the regression harness. Those rows exist so a tooling change has somewhere to go; without them a CI fix has no obvious reviewer and lands unassigned.

**Focus is advisory, not ownership.** Nobody is obliged to review a change because they appear in a row, and nobody is barred from one because they do not. There is no `CODEOWNERS` file, so GitHub requests no reviewers automatically — choosing them is the author's job. When it is not obvious who should review, ask rather than guess, and open the pull request as a draft until it has a reviewer and an assignee.

This file is not published to the documentation site; `docs/development/` sits outside the Documenter source tree.

| Name | Handle | Focus |
|---|---|---|
| Nikolas Logan | `@logan-nc` | KineticForces, PerturbedEquilibrium, Equilibrium, ForceFreeStates, CI, Repo |
| Matthew Pharr | `@matt-pharr` | ForceFreeStates, InnerLayer, CI, Repo |
| Jake Halpern | `@jhalpern30` | Vacuum, ForceFreeStates, Equilibrium |
| Daniel Burgess | `@d-burg` | Tearing, ForceFreeStates |
| Jaebeom Cho | `@JaeBeom1019` | Vacuum, Equilibrium, ForceFreeStates |
| Jaymyoung Lee | `@jmlmir369` | LocalStability, Equilibrium |
| Sunjae Lee | `@jaesun57` | KineticForces |
| Evan Bursch | `@ebursch` | Equilibrium, PerturbedEquilibrium |
| Min-Gu Yoo | `@mgyoo86` | FastInterpolations (external package) |
