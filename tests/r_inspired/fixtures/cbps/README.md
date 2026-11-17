# Archived CBPS comparison: test-only reference bank

This bank belongs to D132255822 and the `test_cbps_r_comparison.py` tests. The complete 500-row CSV and its R capture script moved from production `balance/datasets/`; their bytes, indexes, sample/target split and stored reference weights are unchanged. The local test loader returns `treat == 1` rows as the target (254 rows) and `treat == 0` rows as the sample (246 rows), dropping only `treat`.

## Source and recorded capture

The original `sim_data_cbps.R` script reuses the simulation from R's `CBPS::CBPS` help page with an archived **ATT=1 configuration**: seed `123456`, `n = 500`, `CBPS(treat ~ X.mis, ATT = 1)`. The currently read help page instead fits this simulation with `ATT=0` and credits it to **Kang and Shafer (2007)**; this does not establish what the historical 0.23 help page said. Its recorded `sessionInfo()` names **R 3.6.1**, **CBPS 0.23**, MASS 7.3-51.4 and the remaining package versions. It retains the original simulation, printed results and source header. The reference comparison still checks correlations **>0.98** and **>0.99** on the log10 scale, not exact fitted-weight equality.

Source references:
- [CBPS help-page example](https://rdrr.io/cran/CBPS/man/CBPS.html)
- [CBPS package record](https://cran.auckland.ac.nz/web/packages/CBPS/index.html)
- Imai and Ratkovic (2014), *Covariate balancing propensity score*, DOI:10.1111/rssb.12027.

## Mixed-license attribution

The newly written Python test harness and moved R script retain their existing Meta MIT headers. Reproduced CBPS help-page material and archived reference values are attributed separately; this folder is not declared uniformly MIT by those headers.

The publicly read CBPS **0.24** record declares **GPL (>= 2)** and credits Christian Fong, Marc Ratkovic and Kosuke Imai, with contributors Chad Hazlett, Xiaolin Yang, Sida Peng and Inbeom Lee. GPL-2 and GPL-3 texts accompany this bank. This current package declaration is **not** presented as an independently verified license notice for the historical 0.23 capture: its archive could not be retrieved in this environment. No copyright year or license compatibility conclusion is inferred.

The manifest records the original file hashes at the pre-move revision, the hashes of the unchanged capture files and copied license texts, and the distinction between recorded 0.23 capture metadata and the read current declaration.

## Test-only API and reproduction

`load_cbps_reference()` lives in the R comparison test module. The production `load_cbps_data` export and `load_data("sim_data_cbps")` option are removed; the public `load_data()` / `load_data("sim_data_01")` simulation remains unchanged. The package no longer ships the CBPS CSV or R script.

This bank is a **preserved historical capture**, marked `capture_only` in its manifest. Grouped current-R regeneration excludes it by default and fails with an actionable message when explicitly asked to regenerate it. Use the recorded R script and recorded environment for a deliberate new capture, not current packages silently substituted for the old ones. Ordinary tests read the frozen CSV without R and verify its file/notice hashes.
