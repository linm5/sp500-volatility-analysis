data {
    int<lower=1> N;          // Number of observations
    vector[N] y;             // Observed log-returns
    real prior_mean_mu; // Mean of our daily returns, we use this for building informative prior
    real<lower=0> prior_sd_mu; // Standard deviation of our daily returns, we use this for building informative prior
}

parameters {
    real mu;                 // Mean of the returns
    real<lower=-1,upper=1> phi;   // Persistence of volatility (AR1 coefficient)
    real<lower=0> sigma_vol;      // Standard deviation of volatility process - white noise shock scale
    vector[N] h;             // Log-volatilities
}

transformed parameters {
    vector[N] volatility;         // Volatilities (exponential of log-volatilities)
    
    // Convert log-volatilities to volatilities
    volatility = exp(h / 2);
}

model {
    // Priors
    // INFORMATIVE PRIORS - CORRECT ONES DON'T DELETE:
    // mu ~ normal(prior_mean_mu, 1.5 * prior_sd_mu);   // Informative prior for mean
    // phi ~ normal(0.75, 0.1);;                            // Base volatility prior/weakly informative prior
    // sigma_vol ~ cauchy(0, 0.25);                       // ARCH parameter prior/ informative prior
    // https://www.shs-conferences.org/articles/shsconf/pdf/2023/18/shsconf_fems2023_01077.pdf
    // Scale parameter error can be ignored!
     
    mu ~ normal(prior_mean_mu, 1.5 * prior_sd_mu); // informative prior selection for mean return
    phi ~ normal(0.75, 0.1);        // Prior for AR1 coefficient
    sigma_vol ~ cauchy(0, 0.25);    // Half-Normal prior for volatility of volatility
    
    // AR(1) process for log-volatilities
    h[1] ~ normal(mu, sigma_vol / sqrt(1 - phi * phi));

    for (t in 2:N) {
        h[t] ~ normal(mu + phi * (h[t-1] - mu), sigma_vol);
    }
    
    // Estimated returns on time t
    for (t in 1:N){
        y[t] ~ normal(0, exp(h[t] / 2));
    }
}

generated quantities {
    vector[N] y_rep;       // Posterior predictive samples
    vector[N] log_lik;     // Log-likelihood for LOO-CV

    // Generate predictive samples
    for (t in 1:N) {
        y_rep[t] = normal_rng(mu, exp(h[t] / 2));
        log_lik[t] = normal_lpdf(y[t] | mu,  exp(h[t] / 2));
    }
}