# fts Research Notes

Open research questions for the fts models. Merged from the old root `apuntes.md` and
`params&further_exploration.md` on 2026-11-15, plus review notes.

## Principles

- Don't overcomplicate: one CNN and one RNN (e.g. LSTM) before anything fancier.
- **Always compare against costs and a trivial classifier** (e.g. sign of the last return, or always-long). A model that doesn't beat a naive baseline after costs has no edge.
- CNN kernels to try: 3, 5, 7. Train them in parallel.

## Dates (BTC/USDT 30m, see `apps/fts/specs/components/dates/`)

| Split | Start (UTC) | End (UTC) |
|---|---|---|
| Train + validation | 2025-11-04 19:30 | 2026-05-30 19:30 |
| Embargo | 2026-05-30 19:30 | 2026-06-01 03:30 |
| Test | 2026-06-01 03:30 | 2026-06-21 23:00 |

The original notes had the train/validation end as 2025-05-30, a typo. The spec file uses 2026-05-30.
The test window is 3 weeks (about 1,000 bars). Treat its Sharpe as anecdotal until walk-forward folds exist.

## Open questions

1. **Label vs execution mismatch.** The backtest has a 1-bar execution delay (signal at bar N close, fill at N+1). Shouldn't the training target then be the return the strategy can actually earn, i.e. from N+1 to N+2, rather than the next-bar return?
2. Is the RNN teacher-forcing? Does the model see y(t−1) at t?
3. Put slippage, time delay, latency and non-fills into training, not only the backtest.
4. A portfolio of models predicting on the same bar (`PortfolioStrategy`).
5. Volatility forecasting: are there good quick tools? (See analyser backlog PR 10, EWMA.)
6. Sweep plots: only show hyperparameters that actually varied (more than one value).

## Case study: model 29 vs model 48 (LSTM, same training period)

- Model 29: close log returns only. Validation IC 0.137.
- Model 48: feature pipeline with 5 features (close return, volume change, intra-bar High/Close and Low/Close ratios, …). Validation IC 0.107.

**First check whether the gap is real.** On a single validation split with N ≈ 1,500 bars, the standard error of an IC is about 1/√N ≈ 0.026. The 0.03 gap is about 1.2 standard errors, so it's probably noise. Compare with walk-forward folds or a bootstrapped IC before explaining it. An IC around 0.14 on 30-minute BTC is also high enough to deserve a leakage check. Scalers are fit on the training split only (`plugins/nets/training/training.py`); check the targets and the feature windows too.

If the gap does turn out to be real, the usual explanations are:

- **Signal-to-noise dilution:** volume change and intra-bar ratios are noisy at 30 minutes, and an under-regularised LSTM fits that noise.
- **Mismatched distributions:** volume change is extremely fat-tailed. Even after z-scoring, outliers cause large gradient updates.
- **More capacity on little data** (about 5–7k samples) → overfitting.
- **Hyperparameters tuned for 1 feature:** re-run the search with more dropout and weight decay, or a smaller hidden dimension, for the 5-feature model.
