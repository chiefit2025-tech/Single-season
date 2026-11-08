// Single-season site-occupancy model (MacKenzie et al. 2002)
// with covariates on occupancy (psi) and detection (p).
// The latent occupancy state z is marginalised out of the likelihood,
// so the model can be fitted with Hamiltonian Monte Carlo (NUTS).
// Sites may have different numbers of visits, as in opportunistic
// citizen-science data.

functions {
  // Log-likelihood of one site's detection history, z marginalised out.
  real site_loglik(array[] int y_i, real logit_psi, vector logit_p) {
    real lp_present = log_inv_logit(logit_psi)
                      + bernoulli_logit_lpmf(y_i | logit_p);
    if (sum(y_i) > 0) {
      return lp_present;                            // detected => occupied
    }
    return log_sum_exp(lp_present,                  // occupied, never detected
                       log1m_inv_logit(logit_psi)); // or truly absent
  }
}

data {
  int<lower=1> M;                               // number of sites
  int<lower=1> J;                               // maximum visits per site
  array[M] int<lower=1, upper=J> n_visits;      // visits actually made
  array[M, J] int<lower=0, upper=1> y;          // detections (padded with 0)
  int<lower=1> K_psi;                           // occupancy covariates (incl. intercept)
  matrix[M, K_psi] X_psi;
  int<lower=1> K_p;                             // detection covariates (incl. intercept)
  array[M] matrix[J, K_p] X_p;                  // visit-level design matrices
}

parameters {
  vector[K_psi] beta;                           // occupancy coefficients (logit scale)
  vector[K_p] alpha;                            // detection coefficients (logit scale)
}

transformed parameters {
  vector[M] logit_psi = X_psi * beta;
}

model {
  // Weakly informative priors on the logit scale
  beta ~ normal(0, 1.5);
  alpha ~ normal(0, 1.5);

  for (i in 1:M) {
    int n = n_visits[i];
    target += site_loglik(y[i, 1:n], logit_psi[i], X_p[i][1:n] * alpha);
  }
}

generated quantities {
  vector[M] log_lik;                 // pointwise log-likelihood (for LOO)
  array[M] int z;                    // posterior draw of the latent occupancy state
  real psi_fs;                       // finite-sample proportion of occupied sites
  int n_detected_rep = 0;            // replicated number of sites with >= 1 detection
  real mean_p;                       // average detection probability per visit

  {
    real p_sum = 0;
    int n_tot = 0;
    for (i in 1:M) {
      int n = n_visits[i];
      vector[n] lp = X_p[i][1:n] * alpha;
      log_lik[i] = site_loglik(y[i, 1:n], logit_psi[i], lp);

      if (sum(y[i, 1:n]) > 0) {
        z[i] = 1;
      } else {
        // P(z = 1 | never detected) by Bayes' rule
        real lp1 = log_inv_logit(logit_psi[i]) + sum(log1m_inv_logit(lp));
        real lp0 = log1m_inv_logit(logit_psi[i]);
        z[i] = bernoulli_rng(exp(lp1 - log_sum_exp(lp1, lp0)));
      }

      // Posterior predictive replicate of the whole detection history
      {
        int z_rep = bernoulli_logit_rng(logit_psi[i]);
        int det = 0;
        for (j in 1:n) {
          det += z_rep * bernoulli_logit_rng(lp[j]);
        }
        n_detected_rep += (det > 0);
      }

      p_sum += sum(inv_logit(lp));
      n_tot += n;
    }
    psi_fs = mean(to_vector(z));
    mean_p = p_sum / n_tot;
  }
}
