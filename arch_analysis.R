# Importing necessary libraries
if (!require(tidybayes)) {
  install.packages("tidybayes")
}
library(tidybayes)

if(!require(ggplot2)){
  install.packages("ggplot2")
}
library(ggplot2)

if(!require(cmdstanr)){
  install.packages("cmdstanr", repos = c("https://mc-stan.org/r-packages/", getOption("repos")))
  library(cmdstanr)
}
cmdstan_installed <- function(){
  res <- try(out <- cmdstanr::cmdstan_path(), silent = TRUE)
  !inherits(res, "try-error")
}
if(!cmdstan_installed()){
  install_cmdstan()
}

# Here we are reading the data from CSV file
data <- read.csv("cleaned_s&p_500_data.csv")

# Calculating the log returns' mean and standard deviation
# These will be used in the ARCH model for prior mu
log_return_mean <- mean(data$Log_Returns)
log_return_sd <- sd(data$Log_Returns)

# Defining and compiling the Stan model
arch_model <- cmdstan_model("../BDA_project/arch_model.stan")

# Preparing data for Stan model
stan_data <- list(
    T = nrow(data),
    y = data$Log_Returns,
    prior_mean_mu = log_return_mean,
    prior_sd_mu = log_return_sd
)

print(log_return_sd)

# Estimate model
out <- capture.output(
  # Sampling from the posterior distribution happens here:
  fit <- arch_model$sample(data=stan_data, refresh=0, show_messages=FALSE, seed = 4911, chains = 4, iter_warmup = 1000, 
                    iter_sampling = 2000)
)

# Print the summary of the fit
fit$diagnostic_summary()
