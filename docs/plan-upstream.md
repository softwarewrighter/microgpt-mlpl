# Plan: remove workarounds as sw-mlpl fixes land

Maintenance saga. Each step removes the local workaround for one
upstream issue (docs/upstream-issues.md) once sw-mlpl ships the fix,
and proves nothing else changed: all reg-rs baselines byte-identical
(incl. microgpt-rs parity), tangle checks, tests, republished literate
HTML and pages/.

## Steps

1. remove-j-workaround -- sw-mlpl e6ee2203 fixed issue (j): string-valued
   statements inside repeat/train/for. Replace the `while` loops that
   existed only for (j) with `repeat` (sampling loops, a doc demo);
   update docs; mark (j) fixed.
2. remove-e-workarounds -- when sw-mlpl's copy-on-write values saga
   lands (issue e): re-measure, then drop `expunge` of big globals and
   the pre-gather/pre-encode moves that exist only for speed, where the
   numbers show they no longer matter; update docs/benchmarks.md and the
   readability-vs-speed table.
