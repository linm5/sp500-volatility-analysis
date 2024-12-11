# S\&P 500 Volatility Analysis: Bayesian Inference with ARCH, GARCH, and Stochastic Volatility Models

Understanding market volatility is a fundamental aspect of financial modelling. The S\&P 500 index, a benchmark for U.S. financial markets, serves as a critical indicator of economic health and investor sentiment. 
This project aims to analyze the volatility of the S\&P 500 by applying existing ARCH, GARCH and stochastic volatility models, complemented by Bayesian inference to enhance parameter estimation and uncertainty quantification.

The motivation for this project is that financial markets are inherently volatile, with periods of high uncertainty significantly impacting investment strategies and economic policies. Accurately analyzing volatility is essential for risk management, option pricing, and portfolio optimization. We employ ARCH, GARCH and stochastic volatility models, widely used in financial econometrics, which are effective for capturing volatility clustering,  working by non-linearly modelling heteroscedasticity, which is used for heterogeneity of variance.

The initial challenge or problem for this project lies in determining whether relatively simple models, such as ARCH and GARCH, can adequately capture periods of high volatility and their underlying patterns in the S\&P 500 data. To investigate this, we first applied these models to evaluate their performances. Once these models showed fairly good performances in exploring the data, we tested a more complex approach, the stochastic volatility model, to see if it is overly complicated in the context of this project or if it can make any additional improvements in terms of the inference. 

This project utilizes the following models: ARCH, GARCH, and stochastic volatility. These models are applied to capture volatility clustering in the S\&P 500 index data; Bayesian inference is used for parameter estimation, allowing us to incorporate prior knowledge and quantify the uncertainty in the model outputs. By applying these established techniques, we aim to provide insights into the dynamic behavior of market volatility.

You can read the full report [here](report.pdf) and check presentation slides [here](presentation.pdf).
