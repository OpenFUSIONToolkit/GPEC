<!--
Title this PR the same way you title a commit:  Area - TAG - Imperative summary
Append ! to the TAG if results move or an interface changes:  Equilibrium - BUGFIX! - ...
Conventions: docs/development/naming.md
-->

## Release note

<!-- Delete the option that does not apply and replace every <placeholder>. -->

- **Audience:** users | developers
- **Numerical impact:** none | <what moved> _(harness @ <sha>)_
- **Migration:** none | <what a user must change>

<One to three sentences a user of GPEC would understand: what you can now do, or what was wrong.>

## Regression report

<!--
Required on every PR (docs/development/regression-harness.md). Paste the report below and
put the commit you ran it at in the harness stamp above; CI fails if src/ changed afterwards.

    regress --cases diiid_n1 --refs develop,local
-->

```
```

## Notes for reviewers

<!-- Anything that is not obvious from the diff. Delete if there is nothing. -->

---

Set an **assignee** and at least one **human reviewer**; if you are not ready to name them, open this as a **draft**. Lead developers to ask are listed in `docs/development/contributors.md`. Labels are applied automatically from the title.
