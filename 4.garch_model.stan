data {
  int<lower=1> N;          // Number of observations
  vector[N] y;             // Observed returns or log-returns
  real prior_mean_mu;        // Informative prior mean for mu
  real<lower=0> prior_sd_mu; // Informative prior SD for mu
}

parameters {
  real mu;                 // Mean of the returns
  real<lower=0> alpha0;    // Constant term in the GARCH model
  real<lower=0> alpha1;    // Coefficient for lagged squared residuals
  real<lower=0> beta1;     // Coefficient for lagged conditional variance
}

transformed parameters {
  vector[N] volatility;        // Conditional variances
  
  // Initialize the first conditional variance
  volatility[1] = alpha0 / (1.0 - alpha1 - beta1);  // Stationary assumption
  
  // Calculate the conditional variances
  for (n in 2:N) {
    volatility[n] = alpha0 + alpha1 * square(y[n-1] - mu) + beta1 * volatility[n-1];
    volatility[n] = sqrt(fmax(volatility[n], 1e-8));          // Ensure positivity
  }
}

model {

  // Priors
  // INFORMATIVE PRIORS - CORRECT ONES DON'T DELETE:
  // mu ~ normal(prior_mean_mu, 1.5 * prior_sd_mu); // Informative prior for mean
  // alpha0 ~ normal(0.1, 0.05);             // Base volatility prior/weakly informative prior
  // alpha1 ~ beta(2, 5);                    // ARCH parameter prior/ informative prior
  // https://www.shs-conferences.org/articles/shsconf/pdf/2023/18/shsconf_fems2023_01077.pdf
  // Scale parameter error can be ignored!

  mu ~ normal(prior_mean_mu, 1.5 * prior_sd_mu);                // Informative prior for mean
  alpha0 ~ normal(0.1, 0.05);       // Weakly informative prior
  alpha1 ~ beta(2, 5);              // Informative prior for ARCH term
  beta1 ~ beta(5, 2);               // Informative prior for GARCH term
  
  // Likelihood
  for (n in 1:N) {
    y[n] ~ normal(mu, volatility[n]);
    target += normal_lpdf(y[n] | mu, volatility[n]);
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
