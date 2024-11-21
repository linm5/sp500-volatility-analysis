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
arch_model <- cmdstan_model("../BDA_project/arch_model3.stan", force_recompile = TRUE, quiet = FALSE)

# Command Used for MCMC Inference
fit <- arch_model$sample(
  data = stan_data,
  seed = 4911, # Ensures reproducibility
  chains = 4, # Standard number of chains for robust diagnostics
  iter_warmup = 1000, # Warm-up iterations for tuning
  iter_sampling = 2000, # Post-warm-up iterations for inference
  refresh = 100 # Display progress updates
)

cat("Command used for MCMC inference:\n")
cat("arch_model$sample(data = stan_data, seed = 4911, chains = 4, iter_warmup = 1000, iter_sampling = 2000, refresh = 100)\n\n")
cat("Explanation of options: \n")
cat("- `seed`: Ensures reproducibility\n")
cat("- `chains`: Number of chains (4 is standard for good diagnostics)\n")
cat("- `iter_warmup`: Number of warm-up iterations\n")
cat("- `iter_sampling`: Number of iterations for sampling the posterior\n")
cat("- `refresh`: Frequency of progress updates\n\n")

# Convergence Diagnostic
fit_summary <- fit$summary()
cat("Convergence Diagnostics:\n")

rhat <- fit_summary$rhat
ess_bulk <- fit_summary$ess_bulk
ess_tail <- fit_summary$ess_tail

cat("R-hat values:\n")
print(rhat)
cat("Interpretation: Values close to 1 indicate convergence. If not, the chains need more iterations.\n\n")

cat("Effective Sample Size (ESS):\n")
cat("Bulk ESS:\n")
print(ess_bulk)
cat("Tail ESS:\n")
print(ess_tail)
cat("Interpretation: High ESS values indicate effective sampling. Low values suggest poor mixing.\n\n")

# Convergence Check
# if (any(rhat > 1.1) || any(ess_bulk < 100, na.rm = TRUE)) {
  # cat("Warning: Convergence diagnostics indicate potential issues.\n")
# } else {
  # cat("Diagnostics indicate good convergence.\n")
# }

# Save Diagnostics Summary
write.csv(fit_summary, "diagnostics_summary.csv", row.names = FALSE)

# Visualization of R-hat and ESS
plot(rhat, type = "h", main = "R-hat Diagnostics", ylab = "R-hat")
plot(ess_bulk, type = "h", main = "Bulk ESS", ylab = "ESS (Bulk)")
plot(ess_tail, type = "h", main = "Tail ESS", ylab = "ESS (Tail)")

# Posterior Predictive Checks
if (!"y_rep" %in% fit$metadata()$variables) {
  stop("The 'y_rep' variable is not present in the model. Check the 'generated quantities' block in the .stan file.")
}

y_rep <- fit$draws("y_rep", format = "matrix")

if (ncol(y_rep) != length(data$Log_Returns)) {
  stop("Mismatch between dimensions of y_rep and observed data. Check the Stan model or extraction.")
}

cat("Posterior Predictive Check:\n")
ppc <- ppc_dens_overlay(data$Log_Returns, y_rep)
print(ppc)
ggsave("posterior_predictive_check.png", plot = ppc)

cat("Interpretation:\n")
cat("- If the simulated density does not match the observed density, it indicates model misspecification.\n")
cat("- Possible improvements:\n")
cat("  - Revise the model's likelihood or priors.\n")
cat("  - Add or remove parameters to better capture data behavior.\n")

# Diagnostics
cat("Diagnostics Output:\n")
fit_diagnostics <- fit$cmdstan_diagnose()
print(fit_diagnostics)

# Posterior Summary
cat("Posterior Summary:\n")
posterior_summary <- fit$summary()
print(posterior_summary)

# Trace Plot
cat("Trace Plot:\n")
mcmc_trace(fit$draws(), pars = c("mu", "alpha0", "alpha1"))
