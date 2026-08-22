# Crypto strategy tuning — 10-year robustness grid

Optimization goal: increase cross-asset return without accepting a one-coin overfit. The same parameters are used on all 10 assets.

## Best robust configuration

donchian 4h {"entry_n": 40, "exit_n": 10} lev=1.6 | profitable=9/10 | median return=2907.85% | CAGR=76.23% | MDD=-81.19% | PF=1.203 | Sharpe=1.042

## Highest raw median-return configuration

donchian 1h {"entry_n": 160, "exit_n": 80} lev=1.6 | profitable=9/10 | median return=5697.45% | CAGR=64.41% | MDD=-83.57% | PF=1.147 | Sharpe=0.999

## Best 1x parameter configuration

supertrend 4h {"mult": 3.5, "period": 7} lev=1.0 | profitable=9/10 | median return=3107.84% | CAGR=57.17% | MDD=-72.95% | PF=1.236 | Sharpe=0.986

## Robust constraint

At least 8/10 profitable assets, median PF >= 1.05 after leverage refinement, median MDD >= -90%, and zero liquidations.

See parameter_grid.csv, leverage_grid.csv, and detail files for the complete results.