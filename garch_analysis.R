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
# Load the cleaned data
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

# Calculate log returns' mean and standard deviation
log_return_mean <- mean(data$Log_Returns, na.rm = TRUE)
log_return_sd <- sd(data$Log_Returns, na.rm = TRUE)

# ------------------------
# Compile the Stan Model
# ------------------------
# Define the path to the updated GARCH(1,1) Stan model
stan_model_path <- "garch_model.stan"

# Compile the Stan model
garch_model <- cmdstan_model(stan_model_path)

# ------------------------
# Prepare Data for Stan
# ------------------------
stan_data <- list(
  T = nrow(data),
  y = data$Log_Returns,
  prior_mean_mu = log_return_mean,
  prior_sd_mu = log_return_sd
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

# Summarize results
fit_summary <- fit$summary()
print(fit_summary)

# ------------------------
# Analyze Posterior Draws
# ------------------------
# Extract posterior draws for key parameters
posterior_draws <- fit$draws(variables = c("mu", "alpha0", "alpha1"))
print(head(posterior_draws))

# ------------------------
# Save Results for Future Analysis
# ------------------------
# Save fit summary to a CSV
write.csv(fit_summary, "garch_model_fit_summary.csv", row.names = FALSE)

# Save posterior draws to a CSV (optional for deeper analysis)
posterior_draws_df <- as.data.frame(posterior_draws)
write.csv(posterior_draws_df, "garch_model_posterior_draws.csv", row.names = FALSE)

sum(is.na(data$Log_Returns))
sum(is.infinite(data$Log_Returns))

