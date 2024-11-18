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

# Calculate Mean and SD for Priors
log_return_mean <- mean(data$Log_Returns, na.rm = TRUE)
log_return_sd <- sd(data$Log_Returns, na.rm = TRUE)

stan_data <- list(
  T = nrow(data),
  y = data$Log_Returns,
  prior_mean_mu = log_return_mean,
  prior_sd_mu = log_return_sd
)

# Compile Stan Model
arch_model <- cmdstan_model("../BDA_project/arch_model2.stan", force_recompile = TRUE, quiet = FALSE)

# Fit Model
fit <- arch_model$sample(
  data = stan_data,
  seed = 4911,
  chains = 4,
  iter_warmup = 1000,
  iter_sampling = 2000,
  refresh = 100
)

# Diagnostics
cat("Diagnostics Output:\n")
fit_diagnostics <- fit$cmdstan_diagnose()
print(fit_diagnostics)

# Posterior Summary
cat("Posterior Summary:\n")
posterior_summary <- fit$summary()
print(posterior_summary)

# Check Convergence with Trace Plot
cat("Trace Plot:\n")
mcmc_trace(fit$draws(), pars = c("mu", "alpha0", "alpha1"))

# Extract Posterior Predictive Samples
cat("Extracting Posterior Predictive Samples:\n")
if (!"y_rep" %in% fit$metadata()$variables) {
  stop("The 'y_rep' variable is not present in the model. Check the 'generated quantities' block in the .stan file.")
}

y_rep <- fit$draws("y_rep", format = "matrix")  # Extract as a matrix

# Validate Dimensions
if (ncol(y_rep) != length(data$Log_Returns)) {
  stop("Mismatch between dimensions of y_rep and observed data. Check the Stan model or extraction.")
}

# Posterior Predictive Check
cat("Posterior Predictive Check:\n")
ppc_dens_overlay(data$Log_Returns, y_rep) +
  ggtitle("Posterior Predictive Check")

# Save Results for Future Analysis (Optional)
saveRDS(fit, file = "arch_model_fit.rds")
