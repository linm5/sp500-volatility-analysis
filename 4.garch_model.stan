data {
  int<lower=1> N;          // Number of observations
  vector[N] y;             // Observed returns or log-returns
}

parameters {
  real mu;                 // Mean of the returns
  real<lower=0> alpha0;    // Constant term in the GARCH model
  real<lower=0> alpha1;    // Coefficient for lagged squared residuals
  real<lower=0> beta1;     // Coefficient for lagged conditional variance
}

transformed parameters {
  vector[N] sigma2;        // Conditional variances
  
  // Initialize the first conditional variance
  sigma2[1] = alpha0 / (1.0 - alpha1 - beta1);  // Stationary assumption
  
  // Calculate the conditional variances
  for (n in 2:N) {
    sigma2[n] = alpha0 + alpha1 * square(y[n-1] - mu) + beta1 * sigma2[n-1];
    sigma2[n] = fmax(sigma2[n], 1e-8);          // Ensure positivity
  }
}

model {
  // Priors
  mu ~ normal(0, 1);                // Weakly informative prior
  alpha0 ~ normal(0.1, 0.05);       // Weakly informative prior
  alpha1 ~ beta(2, 5);              // Informative prior for ARCH term
  beta1 ~ beta(5, 2);               // Informative prior for GARCH term
  
  // Likelihood
  for (n in 1:N) {
    target += normal_lpdf(y[n] | mu, sqrt(sigma2[n]));
  }
}

generated quantities {
  vector[N] y_rep;       // Posterior predictive samples
  vector[N] log_lik;     // Log-likelihood for LOO-CV

  // Generate predictive samples
  for (n in 1:N) {
    y_rep[n] = normal_rng(mu, sqrt(sigma2[n]));
  }

  // Compute log-likelihood for each observation
  for (n in 1:N) {
    log_lik[n] = normal_lpdf(y[n] | mu, sqrt(sigma2[n]));
  }
}
