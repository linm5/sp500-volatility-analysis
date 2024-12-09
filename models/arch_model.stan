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
  vector[T] h;  // Conditional variances - log volatilities
  
  // Priors

  // INFORMATIVE PRIORS - Correct - Original Priors:
  mu ~ normal(prior_mean_mu, 1.5 * prior_sd_mu); // Informative prior for mean
  alpha0 ~ normal(0.1, 0.05);             // Base volatility prior/weakly informative prior
  alpha1 ~ beta(2, 5);                    // ARCH parameter prior/ informative prior
  // https://www.shs-conferences.org/articles/shsconf/pdf/2023/18/shsconf_fems2023_01077.pdf
  // Scale parameter error can be ignored!

  // Priors based on financial theory - Alternative priors 1:
  // mu ~ normal(0, 0.2);            // Small mean, reflecting the assumption that returns hover near zero
  // alpha0 ~ lognormal(-2, 0.5);    // Lognormal ensures positivity, with a small mean and moderate uncertainty
  // alpha1 ~ beta(2, 8);            // Slightly stronger belief that alpha1 is close to zero

  // Priors informed by empirical studies - Alternative priors 2:
  // mu ~ student_t(3, 0, 0.5);       // Heavy-tailed prior centered at 0, allowing for more uncertainty
  // alpha0 ~ normal(0.2, 0.1);       // Base volatility with a mean around 0.2 and wider uncertainty
  // alpha1 ~ uniform(0, 1);          // Weakly informative prior, allowing exploration of the entire valid range

  // Dummy priors
  //mu ~ normal(100, 1000);      // Implausibly large prior for mean return
  //alpha0 ~ normal(0, 10);      // Allows negative values, invalid for volatility
  //alpha1 ~ uniform(-1, 2);     // Invalid range for ARCH parameter
  
  // ARCH(1) model
  h[1] = fmax(alpha0 / (1 - alpha1), 1e-8); // Stationary variance initialization
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
