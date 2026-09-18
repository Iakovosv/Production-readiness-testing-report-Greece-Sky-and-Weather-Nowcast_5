# Production Readiness Testing Report

Independent production-readiness evaluation of the
[Greece Sky and Weather Nowcast](https://github.com/Iakovosv/Greece-Sky-and-Weather-Nowcast_5)
`nowcast_ml` Home Assistant add-on.

## Contents

- **[PRODUCTION_READINESS_REPORT.md](PRODUCTION_READINESS_REPORT.md)** — full report:
  5-pillar checklist, stress-test results, performance profile, proposed patches.
- **[patches/production-hardening.patch](patches/production-hardening.patch)** —
  ready-to-apply fixes (verified to apply cleanly on baseline `ad360c0`).

## Headline findings

1. **Pre-rain claim is substantively true.** Measured lead **15–17 min** at
   precision 1.00 with zero false alarms (the notes' 19–42 min is optimistic).
2. **Critical bug found and fixed:** a single dropped sensor packet drove
   `pop_60m` to **100%** — a 100% false-alarm rate from one bad measurement.
3. **Critical bug found and fixed:** 5–30 minute data gaps caused a **100%
   false-alarm rate**; precision during dropouts rose 0.515 → 1.00.
4. **Blocking non-code item:** licence is **CC BY-NC-ND 4.0**, which conflicts
   with HA Community Store distribution and with shipping these patches.
5. Performance is excellent: **1.6 ms/step**, **47 MB** RAM, no memory leak
   over 3435 ticks.

See the report for the full checklist and evidence.

_This report and its patches were produced by an AI agent (OpenHands) on behalf of the repository owner._
