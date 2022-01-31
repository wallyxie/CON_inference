library(cmdstanr)
library(posterior)
library(tidyverse)
library(bayesplot)

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

num_chains <- 2

#Data to be passed to Stan.
state_dim <- 3
temp_ref <- 283
temp_rise <- 5 #High estimate of 5 celsius temperature rise by 2100.
prior_scale_factor <- 0.25
obs_error_scale <- 0.1
obs_every <- 5 #Observations every 10 hours.
t <- 1000 #Total time span of ODE simulation.
x_hat0 <- c(54.014866, 2.595037, 4.777019) #Originally sampled values used for Euler-Maruyama solution.
y_full <- read_csv('generated_data/SCON-C_CO2_logit_short_2022_01_20_08_53_sample_y_t_5000_dt_0-01_sd_scale_0-25.csv')
y <- y_full %>% filter(hour <= t) %>% tail(-1)
ts <- y$hour
N_t <- length(ts)
y <- y %>% select(-hour)
#y <- split(y, 1:nrow(y)) #Convert data observations to list of rows to correspond to Stan's array of vectors type.
y <- as.list(y) #Convert data observations to list of columns to correspond to Stan's array of vectors type.

#Parameter prior means
u_M_prior_dist_params <- c(0.0016, 0, 0.1)
a_SD_prior_dist_params <- c(0.5, 0, 1)
a_DS_prior_dist_params <- c(0.5, 0, 1)
a_M_prior_dist_params <- c(0.5, 0, 1)
a_MSC_prior_dist_params <- c(0.5, 0, 1)
k_S_ref_prior_dist_params <- c(0.0005, 0, 0.1)
k_D_ref_prior_dist_params <- c(0.0008, 0, 0.1)
k_M_ref_prior_dist_params <- c(0.0007, 0, 0.1)
Ea_S_prior_dist_params <- c(20, 5, 80)
Ea_D_prior_dist_params <- c(20, 5, 80)
Ea_M_prior_dist_params <- c(20, 5, 80)

#Create list of lists to pass prior means as initial theta values in Stan corresponding to four chains.
init_theta_single = list(
                  u_M = u_M_prior_dist_params[1],
                  a_SD = a_SD_prior_dist_params[1],
                  a_DS = a_DS_prior_dist_params[1],
                  a_M = a_M_prior_dist_params[1],
                  a_MSC = a_MSC_prior_dist_params[1],
                  k_S_ref = k_S_ref_prior_dist_params[1],
                  k_D_ref = k_D_ref_prior_dist_params[1],
                  k_M_ref = k_M_ref_prior_dist_params[1],
                  Ea_S = Ea_S_prior_dist_params[1],
                  Ea_D = Ea_D_prior_dist_params[1],
                  Ea_M = Ea_M_prior_dist_params[1]
                  )
init_theta = list(init_theta_single)[rep(1, num_chains)] #num_chains copies of initial theta proposals for each of the HMC chains to be used.

data_list = list(
    state_dim = state_dim,
    N_t = N_t,
    ts = ts,
    y = y,
    temp_ref = temp_ref,
    temp_rise = temp_rise,
    prior_scale_factor = prior_scale_factor,
    obs_error_scale = obs_error_scale,
    x_hat0 = x_hat0,
    u_M_prior_dist_params = u_M_prior_dist_params,
    a_SD_prior_dist_params = a_SD_prior_dist_params,
    a_DS_prior_dist_params = a_DS_prior_dist_params,
    a_M_prior_dist_params = a_M_prior_dist_params,
    a_MSC_prior_dist_params = a_MSC_prior_dist_params,
    k_S_ref_prior_dist_params = k_S_ref_prior_dist_params,
    k_D_ref_prior_dist_params = k_D_ref_prior_dist_params,
    k_M_ref_prior_dist_params = k_M_ref_prior_dist_params,
    Ea_S_prior_dist_params = Ea_S_prior_dist_params,
    Ea_D_prior_dist_params = Ea_D_prior_dist_params,
    Ea_M_prior_dist_params = Ea_M_prior_dist_params
    )

file_path <- 'CON_CO2_cont_time.stan' #Read in Stan model code.
lines <- readLines(file_path, encoding = "ASCII")
for (n in 1:length(lines)) cat(lines[n],'\n')
model <- cmdstan_model(file_path)

CON_stan_fit_CO2 <- model$sample(data = data_list, seed = 1234, refresh = 100, init = init_theta, iter_sampling = 5000, iter_warmup = 1000, chains = num_chains, parallel_chains = num_chains, adapt_delta = 0.95)

#Save Stan fit object and NUTS inference results.
CON_stan_fit_CO2$save_object(file = "NUTS_results/CON_CO2_NUTS_inference_SCON-C_data.rds")
CON_stan_fit_CO2_post <- as_tibble(CON_stan_fit_CO2$draws(c("u_M", "a_SD", "a_DS", "a_M", "a_MSC", "k_S_ref", "k_D_ref", "k_M_ref", "Ea_S", "Ea_D", "Ea_M")))
write_csv(CON_stan_fit_CO2_post, "NUTS_results/CON_CO2_NUTS_inference_SCON-C_data_post.csv")
CON_stan_fit_CO2_post_pred <- as_tibble(CON_stan_fit_CO2$draws("y_hat_post_pred"))
write_csv(CON_stan_fit_CO2_post_pred, "NUTS_results/CON_CO2_NUTS_inference_SCON-C_data_post_pred.csv")
CON_stan_fit_CO2_summary <- as_tibble(CON_stan_fit_CO2$summary(c("u_M", "a_SD", "a_DS", "a_M", "a_MSC", "k_S_ref", "k_D_ref", "k_M_ref", "Ea_S", "Ea_D", "Ea_M", "y_hat_post_pred")))
write_csv(CON_stan_fit_CO2_summary, "NUTS_results/CON_CO2_NUTS_inference_SCON-C_data_post_and_post_pred_summary.csv")
