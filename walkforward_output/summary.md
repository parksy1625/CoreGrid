# Walk-forward crypto Donchian fine tuning

Parameters are selected using only the first 70% of each coin's history. The final 30% is used as out-of-sample validation.

## Train-selected #1, then evaluated OOS

Donchian 140/100 1H x1.50 | profitable=4/10 | median return=-25.38% | CAGR=-10.64% | MDD=-78.27% | PF=0.935 | Sharpe=0.239

## Best OOS diagnostic (not a clean validation pick)

Donchian 160/70 1H x1.40 | profitable=5/10 | median return=2.27% | CAGR=0.40% | MDD=-72.07% | PF=0.999 | Sharpe=0.337

See train_parameter_grid.csv, train_leverage_grid.csv, oos_leaders.csv and oos_detail.csv.