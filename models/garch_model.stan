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

  // Priors
  // INFORMATIVE PRIORS - CORRECT ONES DON'T DELETE:
  //mu ~ normal(prior_mean_mu, 1.5 * prior_sd_mu); // Informative prior for mean
  //alpha0 ~ normal(0.1, 0.05);             // Base volatility prior/weakly informative prior
  //alpha1 ~ beta(2, 5);                    // ARCH parameter prior/ informative prior
  //beta1 ~ beta(5, 2);               // Informative prior for GARCH term
  // https://www.shs-conferences.org/articles/shsconf/pdf/2023/18/shsconf_fems2023_01077.pdf
  // Scale parameter error can be ignored!
  
  // Priors based on financial theory - Alternative Priors 1:
  //mu ~ normal(0, 0.2);             // Small mean, reflecting the assumption that returns hover near zero
  //alpha0 ~ lognormal(-2, 0.5);     // Lognormal ensures positivity, with a small mean and moderate uncertainty
  //alpha1 ~ beta(2, 8);             // Slightly stronger belief that alpha1 is close to zero
  //beta1 ~ beta(10, 2);             // Stronger belief in high persistence

  // Priors informed by empirical studies - Alternative Priors 2:
  mu ~ student_t(3, 0, 0.5);         // Heavy-tailed prior centered at 0 for robustness
  alpha0 ~ normal(0.2, 0.1);         // Base volatility with a mean around 0.2 and moderate uncertainty
  alpha1 ~ uniform(0, 1);            // Weakly informative prior for ARCH term
  beta1 ~ uniform(0, 1);             // Weakly informative prior for GARCH term

  // Dummy priors:
  //mu ~ uniform(-1000, 1000);          // Extremely wide prior for mean
  //alpha0 ~ normal(100, 50);           // Unrealistically large base volatility
  //alpha1 ~ beta(50, 0.5);             // Strong bias toward values near 1
  //beta1 ~ lognormal(-5, 0.5); // Lognormal with Extreme Concentration Near Zero
  
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


