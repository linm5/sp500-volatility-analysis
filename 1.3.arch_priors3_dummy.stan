data {
  int<lower=0> T;            // Number of observations
  vector[T] y;               // Returns data
  real prior_mean_mu;        // Informative prior mean for mu (not used in this version)
  real<lower=0> prior_sd_mu; // Informative prior SD for mu (not used in this version)
}

parameters {
  real mu;                   // Mean return
  real alpha0;               // Base volatility
  real alpha1;               // ARCH parameter
}

model {
  vector[T] h;  // Conditional variances - log volatilities

  // Stupid baseline priors
  mu ~ normal(100, 1000);      // Implausibly large prior for mean return
  alpha0 ~ normal(0, 10);      // Allows negative values, invalid for volatility
  alpha1 ~ uniform(-1, 2);     // Invalid range for ARCH parameter

  // ARCH(1) model
  h[1] = fmax(alpha0 / (1 - alpha1), 1e-8);  // Stationary variance initialization with safety check
  for (t in 2:T) {
    h[t] = fmax(alpha0 + alpha1 * square(y[t-1] - mu), 1e-8); // Ensure positivity
  }
  
  // Likelihood
  for (t in 1:T) {
    y[t] ~ normal(mu, sqrt(h[t]));
  }
}

generated quantities {
  vector[T] volatility;
  vector[T] y_rep;  // Posterior predictive samples
  vector[T] log_lik; // Log-likelihood for LOO-CV

  // Compute volatility (conditional standard deviations)
  volatility[1] = sqrt(fmax(alpha0 / (1 - alpha1), 1e-8));
  for (t in 2:T) {
    volatility[t] = sqrt(fmax(alpha0 + alpha1 * square(y[t-1] - mu), 1e-8));
  }

  // Generate predictive samples
  for (t in 1:T) {
    y_rep[t] = normal_rng(mu, volatility[t]);
  }

  // Compute log-likelihood
  for (t in 1:T) {
    log_lik[t] = normal_lpdf(y[t] | mu, volatility[t]);
  }
}
