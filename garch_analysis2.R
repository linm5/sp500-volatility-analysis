# ------------------------
# Ensure Libraries are Loaded
# ------------------------
if (!require(cmdstanr)) {
  install.packages("cmdstanr", repos = c("https://mc-stan.org/r-packages/", getOption("repos")))
  library(cmdstanr)
}
# ------------------------
# Check and Install CmdStan
# ------------------------
if (is.null(cmdstanr::cmdstan_path()) || !file.exists(cmdstanr::cmdstan_path())) {
  message("CmdStan not found or improperly installed. Installing CmdStan...")
  cmdstanr::install_cmdstan(overwrite = TRUE)
}

# ------------------------
# Read and Validate Data
# ------------------------
data <- read.csv("cleaned_s&p_500_data.csv")

# Inspect the data
print(head(data))
str(data)

# Ensure 'Log_Returns' column exists and is numeric
if (!"Log_Returns" %in% colnames(data)) {
  stop("The column 'Log_Returns' is missing in the dataset.")
}
if (!is.numeric(data$Log_Returns)) {
  stop("'Log_Returns' must be numeric.")
}

# Standardize Log Returns
data$Log_Returns <- (data$Log_Returns - mean(data$Log_Returns, na.rm = TRUE)) / 
  sd(data$Log_Returns, na.rm = TRUE)

# Calculate log returns' mean and standard deviation
log_return_mean <- mean(data$Log_Returns, na.rm = TRUE)
log_return_sd <- sd(data$Log_Returns, na.rm = TRUE)

# ------------------------
# Compile the Stan Model
# ------------------------
stan_model_path <- "garch_model.stan"
garch_model <- cmdstan_model(stan_model_path)


# Prepare Data for Stan
# ------------------------
stan_data <- list(
  N = nrow(data),  # Number of observations
  y = data$Log_Returns,
  prior_mean_mu = log_return_mean,
  prior_sd_mu = log_return_sd,
  prior_mean_alpha0 = 1e-4,
  prior_sd_alpha0 = 1e-5,
  prior_mean_alpha1 = 0.2,
  prior_sd_alpha1 = 0.1
)


# ------------------------
# Run Sampling
# ------------------------
fit <- garch_model$sample(
  data = stan_data,
  seed = 4911,
  chains = 4,
  iter_warmup = 1000,
  iter_sampling = 2000,
  refresh = 500
)

# ------------------------
# Print Diagnostics
# ------------------------
print(fit$diagnostic_summary())
fit_summary <- fit$summary()
print(fit_summary)

# ------------------------
# Residual Analysis
# ------------------------
posterior_draws <- fit$draws(variables = c("mu"))
fitted_values <- apply(posterior_draws, 2, mean)  # Average across chains
residuals <- data$Log_Returns - fitted_values

# Plot Residuals
plot(data$Exchange_Date, residuals, type = "l", col = "blue", 
     main = "Residuals Over Time", xlab = "Date", ylab = "Residuals")
abline(h = 0, col = "red", lty = 2)

# ------------------------
# Visualize Volatility
# ------------------------
volatility <- fit$draws(variables = "volatility")  # Adjust variable name as needed
realized_volatility <- data$Log_Returns^2

# Plot Model-Implied and Realized Volatility
plot(data$Exchange_Date, apply(volatility, 2, mean), type = "l", col = "blue",
     main = "Volatility Over Time", xlab = "Date", ylab = "Volatility")
lines(data$Exchange_Date, realized_volatility, col = "red")
legend("topright", legend = c("Model-Implied Volatility", "Realized Volatility"),
       col = c("blue", "red"), lty = 1)

# ------------------------
# Posterior Predictive Check
# ------------------------
posterior_predictive <- fit$draws(variables = "y_pred")  # Adjust variable name as needed
simulated_returns <- apply(posterior_predictive, 2, mean)

# Plot Observed vs Simulated Returns
plot(data$Log_Returns, simulated_returns, main = "Observed vs Simulated Returns",
     xlab = "Observed Returns", ylab = "Simulated Returns")
abline(0, 1, col = "red")

# ------------------------
# Save Results for Future Analysis
# ------------------------
write.csv(fit_summary, "garch_model_fit_summary.csv", row.names = FALSE)
posterior_draws_df <- as.data.frame(posterior_draws)
write.csv(posterior_draws_df, "garch_model_posterior_draws.csv", row.names = FALSE)
