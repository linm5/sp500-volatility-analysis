data {
    int<lower=1> N;          // Number of observations
    vector[N] y;             // Observed log-returns
    real prior_mean_log_return; // Mean of our daily returns, we use this for meaningful sampling for daily log-returns
    real<lower=0> prior_sd_log_return; // Standard deviation of our daily returns (volatility), we use this for building informative prior
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
    //mu ~ normal(prior_sd_log_return, 0.025);// Informative prior for mean
    //phi ~ normal(0.5, 0.1);;                // Base volatility prior/weakly informative prior
    //sigma_vol ~ cauchy(0, 0.1);             // Volatiltiy of volatility parameter prior/ informative prior

    // Updated Dummy Priors
    mu ~ uniform(-1000, 1000);    // Extremely vague and nonsensical prior
    phi ~ uniform(-10, 10);       // Exaggerated range, allowing completely unrealistic values
    sigma_vol ~ gamma(0.01, 0.01);// Extremely vague prior, leading to highly dispersed values

    // AR(1) process for log-volatilities
    h[1] ~ normal(mu, sigma_vol / sqrt(1 - phi * phi));

    for (t in 2:N) {
        h[t] ~ normal(mu + phi * (h[t-1] - mu), sigma_vol);
    }
    
    // Estimated returns on time t
    for (t in 1:N){
        y[t] ~ normal(prior_mean_log_return, exp(h[t] / 2));
    }
}

generated quantities {
    vector[N] y_rep;       // Posterior predictive samples
    vector[N] log_lik;     // Log-likelihood for LOO-CV

    // Generate predictive samples
    for (t in 1:N) {
        y_rep[t] = normal_rng(prior_mean_log_return, exp(h[t] / 2));
        log_lik[t] = normal_lpdf(y[t] | mu,  exp(h[t] / 2));
    }
}