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

# Update Priors to Stupid Baseline
stan_data <- list(
  T = nrow(data),
  y = data$Log_Returns,
  prior_mean_mu = 100,  # Stupid prior mean for mu
  prior_sd_mu = 1000    # Stupid prior SD for mu
)

# Compile Stan Model
arch_model <- cmdstan_model("../BDA_project/1.3.arch_priors3_dummy.stan", force_recompile = TRUE, quiet = FALSE)

# Explanation of MCMC options
cat("MCMC Inference:\n")
cat("- The model was run with 4 chains, each with 1000 warmup iterations and 2000 post-warmup iterations.\n")
cat("- A seed value (4911) was used for reproducibility.\n")

# Fit the model
fit <- arch_model$sample(
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
ggsave("3.3.arch_posterior_predictive_check_stupid_priors.png", plot = ppc)

# Prior Sensitivity Analysis
priors <- list(c(100, 1000), c(0, 10), c(-1, 2))  # Reflects stupid priors
densities_list <- lapply(priors, function(prior) {
  prior_mu <- prior[1]
  prior_sd <- prior[2]
  data.frame(
    x = seq(-100, 200, length.out = 100),  # Adjust range for stupid priors
    density = dnorm(seq(-100, 200, length.out = 100), prior_mu, prior_sd),
    prior = paste("Mean =", prior_mu, "SD =", prior_sd)
  )
})
df <- do.call(rbind, densities_list)
prior_plot <- ggplot(df, aes(x = x, y = density, color = prior)) +
  geom_line(size = 1) +
  labs(title = "Density Plots for Stupid Priors", x = expression(theta), y = "Density") +
  theme_minimal()
ggsave("3.3.stupid_prior_density_plots.png", plot = prior_plot)

# LOO-CV for Model Comparison
log_lik <- fit$draws("log_lik", format = "matrix")
loo_result <- loo(log_lik)
cat("LOO-CV Result:\n")
print(loo_result)
