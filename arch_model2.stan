data {
  int<lower=0> T;            // Number of observations
  vector[T] y;               // Returns data
  real prior_mean_mu;        // Informative prior mean for mu
  real<lower=0> prior_sd_mu; // Informative prior SD for mu
}

parameters {
  real mu;                   // Mean return
  real<lower=0> alpha0;      // Base volatility
  real<lower=0,upper=1> alpha1; // ARCH parameter
}

model {
  vector[T] h;  // Conditional variances
  
  // Priors
  mu ~ normal(prior_mean_mu, prior_sd_mu); // Informative prior for mean
  alpha0 ~ normal(0.1, 0.05);             // Base volatility prior
  alpha1 ~ beta(2, 5);                    // ARCH parameter prior
  
  // ARCH(1) model
  h[1] = alpha0 / (1 - alpha1); // Stationary variance initialization
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

  volatility[1] = sqrt(alpha0 / (1 - alpha1));
  for (t in 2:T) {
    volatility[t] = sqrt(fmax(alpha0 + alpha1 * square(y[t-1] - mu), 1e-8));
  }

  for (t in 1:T) {
    y_rep[t] = normal_rng(mu, volatility[t]);  // Generate predictive samples
  }
}
