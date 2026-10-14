# Project status — 2026-10-14

Goal: publish a small, honest, interactive trajectory-generation teaching project in one day, keeping account usage increase below 10 percentage points.

Decision: fresh coefficient-space 2D implementation; preserve the legacy project outside this repository. No legacy paper numbers or unsupported robot claims. Static sample-bank website keeps hosting free and inference infrastructure unnecessary.

Completed: two models trained on 8,192 paired examples; 128 independent conditions × 64 samples evaluated; provenance hashes and checkpoint replay verified; nine UI presets; English and Chinese documentation; preview animation; Python geometry and browser-metric tests. Desktop and 390px mobile interactions checked.

Evidence: artifacts/evaluation.json; artifacts/verification.txt; docs/assets/samples.json; tests/.

Limitations: single training seed; sine-basis coefficient generation; point geometry; preset bank rather than live inference; no robotics feasibility claim.

Next: publish and verify GitHub Pages plus CI.
