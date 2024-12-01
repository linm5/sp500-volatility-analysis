data {
  int<lower=1> N;          // Number of observations
  vector[N] y;             // Observed returns or log-returns
  real prior_mean_mu;        // Informative prior mean for mu
  real<lower=0> prior_sd_mu; // Informative prior SD for mu
}

parameters {
  real mu;                 // Mean of the returns
  real<lower=0> alpha0;    // Constant term in the GARCH model
  real<lower=0, upper=1> alpha1;    // Coefficient for lagged squared residuals
  real<lower=0, upper=1> beta1;     // Coefficient for lagged conditional variance
}

transformed parameters {
  vector[N] volatility;        // Conditional variances

  // Initialize the first conditional variance
  volatility[1] = alpha0 / (1.0 - alpha1 - beta1);  // Stationary assumption
  volatility[1] = sqrt(fmax(volatility[1], 1e-8));  // Ensure positivity

  // Calculate the conditional variances
  for (n in 2:N) {
    volatility[n] = alpha0 + alpha1 * square(y[n-1] - mu) + beta1 * square(volatility[n-1]);
    volatility[n] = sqrt(fmax(volatility[n], 1e-8));          // Ensure positivity
  }
}

model {
  // Exaggerated and nonsensical priors
  mu ~ uniform(-1000, 1000);          // Extremely wide prior for mean
  alpha0 ~ normal(100, 50);           // Unrealistically large base volatility
  alpha1 ~ beta(50, 0.5);             // Strong bias toward values near 1
  // beta1 ~ beta(0.5, 50);              // Strong bias toward values near 0

  beta1 ~ lognormal(-5, 0.5); // Lognormal with Extreme Concentration Near Zero
  // Use a lognormal prior to heavily skew the distribution:


  // Likelihood
  for (n in 1:N) {
    y[n] ~ normal(mu, volatility[n]);
  }
}

generated quantities {
  vector[N] y_rep;       // Posterior predictive samples
  vector[N] log_lik;     // Log-likelihood for LOO-CV

  // Generate predictive samples
  for (n in 1:N) {
    y_rep[n] = normal_rng(mu, volatility[n]);
  }

  // Compute log-likelihood for each observation
  for (n in 1:N) {
    log_lik[n] = normal_lpdf(y[n] | mu, volatility[n]);
  }
}
