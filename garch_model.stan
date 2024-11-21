
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
  mu ~ normal(0, 1); //Weakly informative
  alpha0 ~ normal(0, 1); //Weakly informative not strong informative.
  alpha1 ~ beta(2, 5);       // Typical prior for GARCH coefficients/ informative
  beta1 ~ beta(5, 2); //informative
  // Likelihood
  for (n in 1:N) {
    target += normal_lpdf(y[n] | mu, sqrt(sigma2[n]));
  }
}

