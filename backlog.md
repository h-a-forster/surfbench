# Backlog

## In progress

## Waiting on me

- polish-2026-10-30 merged to `main` without squash, and the first CI run checked (CI has never run on GitHub).
- `v0.1.0` tagged and the GitHub release published from `docs/release-notes-0.1.0.md`.
- Deribit terms re-read (Tardis.dev clause 9.3 was re-read on 2026-11-06 and allows statistics), with a decision to keep or remove the live-snapshot quote counts in REPORT.md section 4 (already on public `main`) and the Tardis data-quality medians in `results/extractor-check/` (public only after the merge).
- Decision on the 15 Co-Authored-By trailers already in public `main` history: keep them, or rewrite history with a force push.

## Decided

- 2026-10-24: filter v1 frozen with the draft thresholds; any change needs `filter_v2.toml` and a rerun of every study.

## Proposed

- Market-data quick study run and written up: `results/quick-v1/` committed, REPORT.md sections 2, 3 and 5 and the README status rewritten with real numbers (descriptive only if fewer than 12 dates are usable).
- Full 91-month study with a 60-minute horizon.
- svi-py and QuantLib oracle fixtures (DESIGN 7.5).
- eSSVI and SSVI added to the calibrator check.
