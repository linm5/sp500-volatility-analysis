data {
  int<lower=0> T;  // Number of observations
  vector[T] y;     // Returns data
  int prior_mean_mu; // Mean of our daily returns, we use this for building informative prior
  int prior_sd_mu; // Standard deviation of our daily returns, we use this for building informative prior
}

parameters {
  real mu;                // Mean return
  real<lower=0> alpha0;   // Base volatility
  real<lower=0,upper=1> alpha1;  // ARCH parameter
}

model {
  vector[T] h;  // Conditional variances
  
  // Priors
  mu ~ normal(prior_mean_mu, 1.5 * prior_sd_mu); // informative prior selection

  //https://www.shs-conferences.org/articles/shsconf/pdf/2023/18/shsconf_fems2023_01077.pdf
  alpha0 ~ normal(0, 0.05); // weakly informative prior
  alpha1 ~ beta(2, 3.5); // weakly informative prior
  
  // ARCH(1) model
  h[1] = alpha0;
  for (t in 2:T) {
    h[t] = alpha0 + alpha1 * square(y[t-1] - mu);
  }
  
  // Likelihood
  for (t in 1:T) {
    y[t] ~ normal(mu, sqrt(h[t]));
  }
}

generated quantities {
  vector[T] volatility;
  volatility[1] = sqrt(alpha0);
  for (t in 2:T) {
    volatility[t] = sqrt(alpha0 + alpha1 * square(y[t-1] - mu));
  }
}
