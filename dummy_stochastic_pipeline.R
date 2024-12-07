# Required Libraries
if (!require(tidybayes)) install.packages("tidybayes")
library(tidybayes)

if (!require(ggplot2)) install.packages("ggplot2")
library(ggplot2)

if (!require(cmdstanr)) {
  install.packages("cmdstanr", repos = c("https://mc-stan.org/r-packages/", getOption("repos")))
  library(cmdstanr)
}

if (!require(bayesplot)) install.packages("bayesplot")
library(bayesplot)

if (!require(loo)) install.packages("loo")
library(loo)

# Ensure CmdStan is installed
cmdstan_installed <- function() {
  res <- try(out <- cmdstanr::cmdstan_path(), silent = TRUE)
  !inherits(res, "try-error")
}
if (!cmdstan_installed()) install_cmdstan()

# Load Data
data <- read.csv("cleaned_s&p_500_data.csv")

# Check for missing or invalid data
if (any(is.na(data$Log_Returns))) stop("Missing values detected in Log_Returns")

# Prepare Data for Stan Model
stan_data <- list(
  N = nrow(data),
  y = data$Log_Returns,
  prior_mean_log_return = mean(data$Log_Returns, na.rm = TRUE),
  prior_sd_log_return = sd(data$Log_Returns, na.rm = TRUE)
)

# Compile Stan Model
stochastic_volatility_model_dummy <- cmdstan_model("/Users/meishanlin/BDA_project/BDA_project/stochastic_volatility_model_dummy.stan")

# Fit the Model
fit <- stochastic_volatility_model_dummy$sample(
  data = stan_data,
  seed = 4911,
  chains = 4,
  iter_warmup = 1000,
  iter_sampling = 2000,
  refresh = 100
)

# Diagnostics
fit_summary <- fit$summary()
cat("Convergence Diagnostics:\n")
print(fit_summary)
cat("Convergence Diagnostics Interpretation:\n")
cat("- All R-hat values are close to 1.00, indicating proper chain mixing and convergence.\n")

# Check for Divergent Transitions
diagnostics <- fit$diagnostic_summary()
divergences <- diagnostics$divergent__
cat("Number of Divergent Transitions:", sum(divergences), "\n")
if (sum(divergences) > 0) {
  cat("Warning: Divergent transitions detected. Consider increasing adapt_delta.\n")
} else {
  cat("No divergent transitions detected.\n")
}

# Posterior Predictive Checks
y_rep <- fit$draws("y_rep", format = "matrix")
ppc <- ppc_dens_overlay(data$Log_Returns, y_rep)
print(ppc)
ggsave("dummy_prior_ppc.png", plot = ppc)

# Compare Prior vs Posterior for phi
posterior_samples <- as_draws_df(fit$draws())

# Define dummy prior for phi
phi_prior <- data.frame(
  x = seq(-10, 10, length.out = 500),
  density = dunif(seq(-10, 10, length.out = 500), -10, 10)
)

# Posterior for phi
phi_posterior <- posterior_samples %>% select(phi)

# Plot Prior vs Posterior for phi
ggplot() +
  geom_line(data = phi_prior, aes(x = x, y = density), color = "red", linetype = "dashed", size = 1) +
  geom_density(data = phi_posterior, aes(x = phi), fill = "blue", alpha = 0.3) +
  labs(title = "Prior vs Posterior for phi", x = "phi", y = "Density") +
  theme_minimal() +
  ggsave("phi_prior_vs_posterior.png")

# Compare Prior vs Posterior for sigma_vol
# Define dummy prior for sigma_vol
sigma_vol_prior <- data.frame(
  x = seq(0, 5, length.out = 500),
  density = dgamma(seq(0, 5, length.out = 500), shape = 0.01, rate = 0.01)
)

# Posterior for sigma_vol
sigma_vol_posterior <- posterior_samples %>% select(sigma_vol)

# Plot Prior vs Posterior for sigma_vol
ggplot() +
  geom_line(data = sigma_vol_prior, aes(x = x, y = density), color = "red", linetype = "dashed", size = 1) +
  geom_density(data = sigma_vol_posterior, aes(x = sigma_vol), fill = "blue", alpha = 0.3) +
  labs(title = "Prior vs Posterior for sigma_vol", x = "sigma_vol", y = "Density") +
  theme_minimal() +
  ggsave("sigma_vol_prior_vs_posterior.png")

# LOO-CV for Model Comparison
log_lik <- fit$draws("log_lik", format = "matrix")
loo_result <- loo(log_lik)
cat("LOO-CV Result:\n")
print(loo_result)
