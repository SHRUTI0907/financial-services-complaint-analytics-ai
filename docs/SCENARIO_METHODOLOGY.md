# Scenario Methodology

The project provides Conservative, Baseline, and Aggressive scenarios. It also runs seeded Monte Carlo simulation for uncertainty analysis.

## Monte Carlo

Current simulation count: 5,000.

Distributions:

- hourly wage: normal around selected benchmark with 8% standard deviation;
- average handling time: normal around selected value with 5-minute standard deviation and lower bound of 1 minute;
- AI-addressable share: triangular distribution;
- time reduction: triangular distribution;
- adoption rate: triangular distribution;
- implementation cost: lognormal distribution;
- annual operating cost: lognormal distribution;
- success probability: triangular distribution.

Outputs include median NPV, P10/P50/P90 NPV, probability NPV > 0, and probability of payback within 3 years.

Modelled value is not realized value.
