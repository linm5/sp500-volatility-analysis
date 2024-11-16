data {
  int<lower=0> T;  // Number of observations
  vector[T] y;     // Returns data
}

parameters {
  real mu;                // Mean return
  real<lower=0> alpha0;   // Base volatility
  real<lower=0,upper=1> alpha1;  // ARCH parameter
}

model {
  vector[T] h;  // Conditional variances
  
  // Priors
  mu ~ normal(0, 0.1); // update to an informative prior
  alpha0 ~ normal(0, 0.1); // alpha0 we don't really know
  alpha1 ~ beta(2, 2); // alpha1 we don't really know
  
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
