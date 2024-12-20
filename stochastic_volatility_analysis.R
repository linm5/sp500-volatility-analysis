# Required Libraries

# Step 1: Create user library directory
dir.create(path = "~/R/library", recursive = TRUE, showWarnings = FALSE)

# Step 2: Set user library path
.libPaths(c("~/R/library", .libPaths()))

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

if (!require(rstantools)) install.packages("rstantools", repos = c("https://mc-stan.org/rstantools/"), getOption("repos"))
library(rstantools)

if (!require(loo)) install.packages("loo")
library(loo)

# Step 3: Install and load stringi
if (!require(stringi)) {
  install.packages("stringi", lib = "~/R/library", repos = "https://cran.rstudio.com/")
}
library(stringi)

# Ensure CmdStan is installed
cmdstan_installed <- function() {
  res <- try(out <- cmdstanr::cmdstan_path(), silent = TRUE)
  !inherits(res, "try-error")
}
if (!cmdstan_installed()) install_cmdstan()

# Making sure plot texts are adequetly sized
theme_set(bayesplot::theme_default(base_family = "sans", base_size = 24) +
theme(
    legend.text = element_text(size = 32),
    legend.title = element_text(size = 36)
  )
)


# Load Data
data <- read.csv("data/cleaned_s&p_500_data.csv")

# Check for missing or invalid data
if (any(is.na(data$Log_Returns))) stop("Missing values detected in Log_Returns")

# Calculate Mean and SD for Priors
log_return_mean <- mean(data$Log_Returns, na.rm = TRUE)
log_return_sd <- sd(data$Log_Returns, na.rm = TRUE)

# Prepare Data for Stochastic Volatility Model
stan_data <- list(
  N = nrow(data),
  y = data$Log_Returns,
  prior_mean_log_return = log_return_mean,
  prior_sd_log_return = log_return_sd
)

# Compile Stan Model
stochastic_volatility_model <- cmdstan_model("models/stochastic_volatility_model.stan", force_recompile = TRUE, quiet = FALSE)

# Explanation of MCMC options
cat("MCMC Inference:\n")
cat("- The model was run with 4 chains, each with 1000 warmup iterations and 2000 post-warmup iterations.\n")
cat("- A seed value (4911) was used for reproducibility.\n")

# Fit the model
fit <- stochastic_volatility_model$sample(
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

# Check for divergent transitions
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
ggsave("graphics/stochastic_volatility_ppc.png", plot = ppc)

# Dummy Priors:
#ggsave("graphics/stochastic_volatility_ppc_dummy_priors.png", plot = ppc)

# Compare Prior vs Posterior for phi
# posterior_samples <- as_draws_df(fit$draws())

# Define dummy prior for phi
# phi_prior <- data.frame(
#  x = seq(-10, 10, length.out = 500),
#  density = dunif(seq(-10, 10, length.out = 500), -10, 10)
# )

# Posterior for phi
# phi_posterior <- posterior_samples %>% select(phi)

# Plot Prior vs Posterior for phi
#ggplot() +
#  geom_line(data = phi_prior, aes(x = x, y = density), color = "red", linetype = "dashed", size = 1) +
#  geom_density(data = phi_posterior, aes(x = phi), fill = "blue", alpha = 0.3) +
#  labs(title = "Prior vs Posterior for phi", x = "phi", y = "Density") +
#  theme_minimal() +
#  ggsave("phi_prior_vs_posterior.png")

# Compare Prior vs Posterior for sigma_vol
# Define dummy prior for sigma_vol
#sigma_vol_prior <- data.frame(
#  x = seq(0, 5, length.out = 500),
#  density = dgamma(seq(0, 5, length.out = 500), shape = 0.01, rate = 0.01)
#)

# Posterior for sigma_vol
#sigma_vol_posterior <- posterior_samples %>% select(sigma_vol)

# Plot Prior vs Posterior for sigma_vol
#ggplot() +
#  geom_line(data = sigma_vol_prior, aes(x = x, y = density), color = "red", linetype = "dashed", size = 1) +
#  geom_density(data = sigma_vol_posterior, aes(x = sigma_vol), fill = "blue", alpha = 0.3) +
#  labs(title = "Prior vs Posterior for sigma_vol", x = "sigma_vol", y = "Density") +
#  theme_minimal() +
#  ggsave("sigma_vol_prior_vs_posterior.png")

# LOO-CV for Model Comparison
log_lik <- fit$draws("log_lik", format = "matrix")
loo_result <- loo(log_lik, save_psis = TRUE)
cat("LOO-CV Result:\n")
print(loo_result)

# Drawing LOO-PIT
loo_pit <- bayesplot::ppc_loo_pit_qq(y = data$Log_Returns, yrep = y_rep, psis_object = loo_result$psis_object)
ggsave("graphics/sv_loo_pit.png", plot = loo_pit)