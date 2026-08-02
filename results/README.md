# Published artifacts

- `work_plan.json`: exact pinned source-file list, byte sizes, deterministic 256 MiB regions, order, and self-hash.
- `mixture_progress.json`: final crash-safe cumulative state.
- `mixture_report.json`: all-cluster token/document totals and embedded hashes.
- `climbmix_code_free_weights.json`: accepted-cluster integer token weights. The historical filename is imprecise: the guarantee is **programming-cluster-excluded**, not code-free.
- `independent_sample_audit.json`: deterministic full-JSON validation of 1,000 records, 10 from each of all 100 source files.
- `verification.json`: audit summary and raw artifact SHA-256 values.

Run `climbmix-verify results` after installing the package to check all offline invariants and hashes.
