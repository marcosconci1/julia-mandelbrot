# Design notes

Why the indicator measures what it measures.

* Trend is normalized. Raw log price slope carries a unit that changes with the asset, so a 0.02 slope means nothing on its own. Dividing by the dispersion of the same window produces a score that compares across tickers and across years.
* Volatility is a percentile, not a level. A 2 percent daily move is calm for one asset and violent for another. Ranking realized volatility inside the asset's own 252 session history answers "how unusual is today" on a scale that needs no unit.
* State is a cell, not a label. Trend and volatility are separate dimensions with separate thresholds, and the pair forms six cells. Two dimensions keep each read auditable: you can see which side moved.
* Thresholds live in the past. Every boundary is an expanding quantile estimated through the previous bar. A row that set its own threshold would classify against a standard it moved.
* Odds come with a floor. The transition model speaks only when 30 or more transitions of evidence exist in its window, pools history below that, and reports nothing below the pooled floor. Thin evidence posing as odds is worse than silence.
* Membership scores stay scores. A logistic score around a threshold expresses position. Calling it a probability requires calibration evidence, and `validate` measures exactly that gap.
* The tail is unconditional. Expected shortfall and drawdown describe the asset in the trailing window and deliberately ignore the regime cell. The transition model conditions on the cell; mixing the two would answer two questions with one number.
* Roughness describes the observed series. The variogram Hurst estimate of realized log volatility says how the measured series behaves at lags 1 to 10. It is not a claim about a latent stochastic process.
* Validation demotes. Every column above earns its place through `validate`. A column that fails its check is reported as demoted, and the demotion is visible in the output.
