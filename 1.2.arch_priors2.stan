data {
  int<lower=0> T;            // Number of observations
  vector[T] y;               // Returns data
  real prior_mean_mu;        // Informative prior mean for mu (not used in this version)
  real<lower=0> prior_sd_mu; // Informative prior SD for mu (not used in this version)
}

parameters {
  real mu;                   // Mean return
  real<lower=0> alpha0;      // Base volatility
  real<lower=0,upper=1> alpha1; // ARCH parameter
}

model {
  vector[T] h;  // Conditional variances - log volatilities

  // Priors informed by empirical studies
  mu ~ student_t(3, 0, 0.5);       // Heavy-tailed prior centered at 0, allowing for more uncertainty
  alpha0 ~ normal(0.2, 0.1);       // Base volatility with a mean around 0.2 and wider uncertainty
  alpha1 ~ uniform(0, 1);          // Weakly informative prior, allowing exploration of the entire valid range

  // ARCH(1) model
  h[1] = alpha0 / (1 - alpha1);    // Stationary variance initialization
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
  volatility[1] = sqrt(alpha0 / (1 - alpha1));
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
