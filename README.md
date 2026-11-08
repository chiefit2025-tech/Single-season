# Bayesian site-occupancy model in Stan: correcting for imperfect detection

A small, reproducible project that fits a **single-season site-occupancy model** in **Stan** (via **CmdStanR**) to simulated detection/non-detection data, and shows why ignoring imperfect detection gives misleading answers.

The simulated data loosely mimic **opportunistic citizen-science records**: sites receive an unequal number of visits (1–4), and the chance of recording the species depends on habitat and observer effort.

## The question

A species prefers forest, but is **harder to detect in dense forest**. If we simply ask "where was the species recorded?", do we get the right answer about where it *lives*?

## Model

For site $i = 1,\dots,M$ and visit $j = 1,\dots,n_i$:

$$
z_i \sim \text{Bernoulli}(\psi_i), \qquad \text{logit}(\psi_i) = \beta_0 + \beta_1\,\text{forest}_i
$$

$$
y_{ij} \mid z_i \sim \text{Bernoulli}(z_i\, p_{ij}), \qquad \text{logit}(p_{ij}) = \alpha_0 + \alpha_1\,\text{forest}_i + \alpha_2\,\text{effort}_{ij}
$$

The latent occupancy state $z_i$ is **marginalised out** so the model can be sampled with NUTS:

- if the species was detected at least once: $\;\log \psi_i + \sum_j \log \text{Bern}(y_{ij}\mid p_{ij})$
- if it was never detected: $\;\log\big[\psi_i \prod_j (1-p_{ij}) + (1-\psi_i)\big]$

Posterior draws of each $z_i$ are recovered afterwards in `generated quantities`, giving the finite-sample proportion of occupied sites. The **naive comparison model** is a logistic regression of "species ever recorded at the site" on forest cover.

Priors: $\beta, \alpha \sim \text{Normal}(0, 1.5)$ on the logit scale.

## Repository structure

```
stan/
  occupancy.stan          # occupancy model (marginalised likelihood, unequal visits)
  naive_logistic.stan     # comparison model that ignores detection
R/
  00_setup.R              # packages and helpers
  01_simulate_data.R      # simulate 200 sites with 1–4 visits each
  02_fit_models.R         # fit both models with CmdStanR
  03_check_results.R      # parameter recovery, model comparison, posterior predictive check
  04_visits_experiment.R  # small simulation study: number of sites × visits per site
  run_all.R               # runs scripts 01–03
data/                     # simulated data (created by script 01)
output/                   # tables and figures (created by scripts 03–04)
```

## How to run

```r
# One-time setup
install.packages("cmdstanr", repos = c("https://stan-dev.r-universe.dev", getOption("repos")))
install.packages("posterior")
cmdstanr::install_cmdstan()

# From the project root
source("R/run_all.R")              # about 1 minute
source("R/04_visits_experiment.R") # about 2-5 minutes
```

Tested with R 4.3, CmdStan 2.36 and cmdstanr 0.8.1.

## Results (seed 2026)

Of 200 simulated sites, 104 were occupied, but the species was recorded at only 64; **40 occupied sites produced no record at all**.

**1. The occupancy model recovers the true parameters.** All five true values fall inside the 90% credible intervals; all R-hat ≤ 1.001, no divergent transitions.

| Parameter | True | Posterior mean | 90% CrI |
|---|---|---|---|
| ψ: intercept | 0.30 | 0.09 | [−0.35, 0.63] |
| ψ: forest | 1.00 | 0.83 | [0.34, 1.40] |
| p: intercept | −0.50 | −0.53 | [−0.88, −0.20] |
| p: forest | −0.70 | −0.74 | [−1.06, −0.41] |
| p: effort | 0.80 | 0.98 | [0.67, 1.30] |

![Parameter recovery](output/figures/parameter_recovery.png)

**2. Ignoring detection gives the wrong picture.**

| Quantity | True | Occupancy model | Naive model |
|---|---|---|---|
| Proportion of sites occupied | 0.52 | 0.51 [0.44, 0.60] | 0.32 [0.27, 0.38] |
| Forest effect on occupancy (logit) | 1.00 | 0.83 [0.34, 1.40] | 0.21 [−0.04, 0.46] |

The naive model underestimates occupancy by about a third and almost erases the forest effect, because the species' preference for forest is cancelled out by its lower detectability there.

![Occupancy vs naive](output/figures/occupancy_vs_naive.png)

**3. Posterior predictive check.** The observed number of sites with at least one detection (64) lies well within the replicated distribution (90% interval 49–78; posterior predictive p = 0.48).

**4. Data needs (script 04, 10 replicates per scenario).** Average error and width of the 90% interval for the proportion of occupied sites:

| Sites | Visits per site | Mean error | Mean 90% width | Coverage |
|---|---|---|---|---|
| 100 | 1–2 | 0.003 | 0.35 | 0.9 |
| 300 | 1–2 | 0.018 | 0.24 | 0.9 |
| 100 | 1–4 | 0.024 | 0.21 | 1.0 |
| 300 | 1–4 | 0.005 | 0.13 | 0.8 |
| 100 | 4 | −0.014 | 0.12 | 0.9 |
| 300 | 4 | −0.004 | 0.08 | 0.9 |

Repeat visits reduce uncertainty more than extra sites do: 100 sites with four visits give narrower intervals than 300 sites with one or two. With only 10 replicates, coverage figures are rough.

## Possible extensions

- Spatial random effect on occupancy (e.g. a Hilbert-space Gaussian process approximation).
- Multi-species model with species-level random effects.
- Observer-level random effects on detection, as in citizen-science data.
- Model comparison with approximate leave-one-out cross-validation (`loo` package; `log_lik` is already computed).

## References

- MacKenzie, D. I., Nichols, J. D., Lachman, G. B., Droege, S., Royle, J. A., & Langtimm, C. A. (2002). Estimating site occupancy rates when detection probabilities are less than one. *Ecology*, 83(8), 2248–2255. [doi](https://doi.org/10.1890/0012-9658%282002%29083%5B2248%3AESORWD%5D2.0.CO%3B2)
- Altwegg, R., & Nichols, J. D. (2019). Occupancy models for citizen-science data. *Methods in Ecology and Evolution*, 10(1), 8–21. [doi:10.1111/2041-210X.13090](https://doi.org/10.1111/2041-210X.13090)
- Carpenter, B., et al. (2017). Stan: A probabilistic programming language. *Journal of Statistical Software*, 76(1). [doi:10.18637/jss.v076.i01](https://doi.org/10.18637/jss.v076.i01)
- Stan Development Team. *Stan User's Guide*, chapter on latent discrete parameters. [mc-stan.org/docs](https://mc-stan.org/docs/)

## Author

Shakeel Nouman — statistician (MSc, MPhil Statistics), Lahore, Pakistan. [github.com/chiefit2025-tech](https://github.com/chiefit2025-tech)

## License

MIT
