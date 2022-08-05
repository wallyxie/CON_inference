functions {

  // Temperature function for ODE forcing.
  real temp_func(real t, real temp_ref, real temp_rise) {
    return temp_ref + (temp_rise * t) / (80 * 24 * 365) + 10 * sin((2 * pi() / 24) * t) + 10 * sin((2 * pi() / (24 * 365)) * t);
  }

  vector temp_func_vec(vector ts, real temp_ref, real temp_rise) {
    return temp_ref + (temp_rise * ts) / (80 * 24 * 365) + 10 * sin((2 * pi() / 24) * ts) + 10 * sin((2 * pi() / (24 * 365)) * ts);
  }

  // Exogenous SOC input function.
  real i_s_func(real t) {
    return 0.001 + 0.0005 * sin((2 * pi() / (24 * 365)) * t);
  }

  // Exogenous DOC input function.
  real i_d_func(real t) {
    return 0.0001 + 0.00005 * sin((2 * pi() / (24 * 365)) * t);
  }

  // Function for enforcing Arrhenius temperature dependency of ODE parameter.
  real arrhenius_temp(real input, real temp, real Ea, real temp_ref) {
    return input * exp(-Ea / 0.008314 * (1 / temp - 1 / temp_ref));
  }

  vector arrhenius_temp_vec(real input, vector temps, real Ea, real temp_ref) {
    return input * exp(-Ea / 0.008314 * (1 ./ temps - 1 ./ temp_ref));
  }

  // Function for enforcing linear temperature dependency of ODE parameter.
  real linear_temp(real input, real temp, real a_SD, real temp_ref) {
    return input - a_SD * (temp - temp_ref);
  }

  vector linear_temp_vec(real input, vector temps, real a_SD, real temp_ref) {
    return input - a_SD * (temps - temp_ref);
  }

  /*
  Deterministic CON model as in Allison et al., 2010.
  C[1] is soil organic carbon (SOC) density.
  C[2] is dissolved organic carbon (DOC) density.
  C[3] is microbial biomass carbon (MBC) density.
  */

  vector CON_ODE(real t, vector C,
                 real u_M, // Microbial DOC uptake rate.
                 real a_SD, // SOC to DOC transfer coefficient.
                 real a_DS, // DOC to SOC transfer coefficient.
                 real a_M, // MBC transfer coefficient.
                 real a_MSC, // MBC decomposition to SOC fraction.
                 real k_S_ref, // SOC decay rate.
                 real k_D_ref, // DOC decay rate.
                 real k_M_ref, // MBC decay rate.
                 real Ea_S, // SOC decay energy of activation.
                 real Ea_D, // DOC decay energy of activation.
                 real Ea_M, // MBC decay energy of activation.
                 real temp_ref,
                 real temp_rise) {

    // Initiate exogenous input and forcing variables for future assignment.
    real temp;
    real i_s;
    real i_d;
    real k_S; // Forced k_S.
    real k_D; // Forced k_D.
    real k_M; // Forced k_M.

    // Assign input and forcing variables to appropriate value at time t.
    temp = temp_func(t, temp_ref, temp_rise); // x_r[1] is temp_ref 283.
    i_s = i_s_func(t);
    i_d = i_d_func(t);

    // Force temperature dependent parameters.
    k_S = arrhenius_temp(k_S_ref, temp, Ea_S, temp_ref);
    k_D = arrhenius_temp(k_D_ref, temp, Ea_D, temp_ref);
    k_M = arrhenius_temp(k_M_ref, temp, Ea_M, temp_ref);

    // Initiate vector for storing derivatives.
    vector[3] dCdt;

    // Compute derivatives.
    dCdt[1] = i_s + a_DS * k_D * C[2] + a_M * a_MSC * k_M * C[3] - k_S * C[1];
    dCdt[2] = i_d + a_SD * k_S * C[1] + a_M * (1 - a_MSC) * k_M * C[3] - (u_M + k_D) * C[2];
    dCdt[3] = u_M * C[2] - k_M * C[3];
    return dCdt;
  }

  // Calculate model output CO2 observations from states x.
  vector calc_CON_CO2(real[] ts,
                      array[] vector C,
                      real a_SD, // SOC to DOC transfer coefficient.
                      real a_DS, // DOC to SOC transfer coefficient.
                      real a_M, // MBC transfer coefficient.
                      real k_S_ref, // SOC decay rate.
                      real k_D_ref, // DOC decay rate.
                      real k_M_ref, // MBC decay rate.
                      real Ea_S, // SOC decay energy of activation.
                      real Ea_D, // DOC decay energy of activation.
                      real Ea_M, // MBC decay energy of activation.
                      real temp_ref,
                      real temp_rise) {
    
    vector[size(ts)] ts_vec;
    vector[size(ts)] k_S;
    vector[size(ts)] k_D;
    vector[size(ts)] k_M;
    vector[size(ts)] temp_vec;
    vector[size(ts)] CO2;
    
    ts_vec = to_vector(ts);
    //print("ts_vec", ts_vec);
    temp_vec = temp_func_vec(ts_vec, temp_ref, temp_rise);
    //print("temp_vec", temp_vec);
    k_S = arrhenius_temp_vec(k_S_ref, temp_vec, Ea_S, temp_ref);
    k_D = arrhenius_temp_vec(k_D_ref, temp_vec, Ea_D, temp_ref);
    k_M = arrhenius_temp_vec(k_M_ref, temp_vec, Ea_M, temp_ref);
    CO2 = (k_S .* C[1,] * (1 - a_SD)) + (k_D .* C[2,] * (1 - a_DS)) + (k_M .* C[3,] * (1 - a_M));
  
    return CO2;    
  }

  // From https://discourse.mc-stan.org/t/rng-for-truncated-distributions/3122/12.
  real normal_lb_ub_rng(real mu, real sigma, real lb, real ub) {
      real p1 = normal_cdf(lb, mu, sigma);  // cdf with lower bound
      real p2 = normal_cdf(ub, mu, sigma);  // cdf with upper bound
      real u = uniform_rng(p1, p2);
      return (sigma * inv_Phi(u)) + mu;  // inverse cdf 
  }
}

data {
  int<lower=1> state_dim; // Number of state dimensions (4 for AWB).
  int<lower=1> N_t; // Number of observations.
  array[N_t] real<lower=0> ts; // Univariate array of observation time steps.
  array[state_dim+1] vector<lower=0>[N_t] y; // Multidimensional array of state observations and CO2 bounded at 0. y in [state_dim, N_t] shape to facilitate likelihood sampling.
  real<lower=0> temp_ref; // Reference temperature for temperature forcing.
  real<lower=0> temp_rise; // Assumed increase in temperature (°C/K) over next 80 years.
  real<lower=0> prior_scale_factor; // Factor multiplying parameter means to obtain prior standard deviations.
  real<lower=0> obs_error_scale; // Observation noise factor multiplying observations of model output x_hat.
  vector<lower=0>[state_dim] x_hat0; // Initial ODE conditions.
  // [1] is prior mean, [2] is prior lower bound, [3] is prior upper bound.
  array[3] real<lower=0> u_M_prior_dist_params;
  array[3] real<lower=0> a_SD_prior_dist_params;
  array[3] real<lower=0> a_DS_prior_dist_params;
  array[3] real<lower=0> a_M_prior_dist_params;
  array[3] real<lower=0> a_MSC_prior_dist_params;
  array[3] real<lower=0> k_S_ref_prior_dist_params;
  array[3] real<lower=0> k_D_ref_prior_dist_params;
  array[3] real<lower=0> k_M_ref_prior_dist_params;
  array[3] real<lower=0> Ea_S_prior_dist_params;
  array[3] real<lower=0> Ea_D_prior_dist_params;
  array[3] real<lower=0> Ea_M_prior_dist_params;
}

transformed data {
  real t0 = 0; // Initial time.
}

parameters {
  real<lower = u_M_prior_dist_params[2], upper = u_M_prior_dist_params[3]> u_M; // SOC to DOC transfer coefficient.
  real<lower = a_SD_prior_dist_params[2], upper = a_SD_prior_dist_params[3]> a_SD; // SOC to DOC transfer coefficient.
  real<lower = a_DS_prior_dist_params[2], upper = a_DS_prior_dist_params[3]> a_DS; // DOC to SOC transfer coefficient.
  real<lower = a_M_prior_dist_params[2], upper = a_M_prior_dist_params[3]> a_M; // MBC transfer coefficient
  real<lower = a_MSC_prior_dist_params[2], upper = a_MSC_prior_dist_params[3]> a_MSC; // MBC decomposition to SOC fraction.
  real<lower = k_S_ref_prior_dist_params[2], upper = k_S_ref_prior_dist_params[3]> k_S_ref; // SOC decay rate.
  real<lower = k_D_ref_prior_dist_params[2], upper = k_D_ref_prior_dist_params[3]> k_D_ref; // DOC decay rate.
  real<lower = k_M_ref_prior_dist_params[2], upper = k_M_ref_prior_dist_params[3]> k_M_ref; // MBC decay rate.
  real<lower = Ea_S_prior_dist_params[2], upper = Ea_S_prior_dist_params[3]> Ea_S; // SOC decay energy of activation.
  real<lower = Ea_D_prior_dist_params[2], upper = Ea_D_prior_dist_params[3]> Ea_D; // DOC decay energy of activation.
  real<lower = Ea_M_prior_dist_params[2], upper = Ea_M_prior_dist_params[3]> Ea_M; // MBC decay energy of activation.
}

transformed parameters {
  vector<lower=0>[N_t] x_hat_CO2;
  array[state_dim+1] vector<lower=0>[N_t] x_hat_add_CO2;

  // Solve ODE.
  array[N_t] vector<lower=0>[state_dim] x_hat_intmd = ode_rk45(CON_ODE, x_hat0, t0, ts, u_M, a_SD, a_DS, a_M, a_MSC, k_S_ref, k_D_ref, k_M_ref, Ea_S, Ea_D, Ea_M, temp_ref, temp_rise);

  // Transform model output to match observations y in shape, [state_dim, N_t].
  array[state_dim] vector<lower=0>[N_t] x_hat;
  for (i in 1:N_t) {
    for (j in 1:state_dim) {
      x_hat[j, i] = x_hat_intmd[i, j];
    }
  }

  // Compute CO2.
  x_hat_CO2 = calc_CON_CO2(ts, x_hat, a_DS, a_SD, a_M, k_S_ref, k_D_ref, k_M_ref, Ea_S, Ea_D, Ea_M, temp_ref, temp_rise);   

  // Add CO2 vector to x_hat.
  x_hat_add_CO2[1:3,] = x_hat;
  x_hat_add_CO2[4,] = x_hat_CO2;
  
  //print("Leapfrog x: ", x_hat);
  //print("Leapfrog CO2: ", x_hat_CO2);
  //print("Leapfrog x add CO2: ", x_hat_add_CO2);  
}

model {
  u_M ~ normal(u_M_prior_dist_params[1], u_M_prior_dist_params[1] * prior_scale_factor) T[u_M_prior_dist_params[2], u_M_prior_dist_params[3]];
  a_SD ~ normal(a_SD_prior_dist_params[1], a_SD_prior_dist_params[1] * prior_scale_factor) T[a_SD_prior_dist_params[2], a_SD_prior_dist_params[3]];
  a_DS ~ normal(a_DS_prior_dist_params[1], a_DS_prior_dist_params[1] * prior_scale_factor) T[a_DS_prior_dist_params[2], a_DS_prior_dist_params[3]];
  a_M ~ normal(a_M_prior_dist_params[1], a_M_prior_dist_params[1] * prior_scale_factor) T[a_M_prior_dist_params[2], a_M_prior_dist_params[3]];
  a_MSC ~ normal(a_MSC_prior_dist_params[1], a_MSC_prior_dist_params[1] * prior_scale_factor) T[a_MSC_prior_dist_params[2], a_MSC_prior_dist_params[3]];
  k_S_ref ~ normal(k_S_ref_prior_dist_params[1], k_S_ref_prior_dist_params[1] * prior_scale_factor) T[k_S_ref_prior_dist_params[2], k_S_ref_prior_dist_params[3]];
  k_D_ref ~ normal(k_D_ref_prior_dist_params[1], k_D_ref_prior_dist_params[1] * prior_scale_factor) T[k_D_ref_prior_dist_params[2], k_D_ref_prior_dist_params[3]];
  k_M_ref ~ normal(k_M_ref_prior_dist_params[1], k_M_ref_prior_dist_params[1] * prior_scale_factor) T[k_M_ref_prior_dist_params[2], k_M_ref_prior_dist_params[3]];
  Ea_S ~ normal(Ea_S_prior_dist_params[1], Ea_S_prior_dist_params[1] * prior_scale_factor) T[Ea_S_prior_dist_params[2], Ea_S_prior_dist_params[3]];
  Ea_D ~ normal(Ea_D_prior_dist_params[1], Ea_D_prior_dist_params[1] * prior_scale_factor) T[Ea_D_prior_dist_params[2], Ea_D_prior_dist_params[3]];
  Ea_M ~ normal(Ea_M_prior_dist_params[1], Ea_M_prior_dist_params[1] * prior_scale_factor) T[Ea_M_prior_dist_params[2], Ea_M_prior_dist_params[3]];
  //print("Leapfrog theta: ", "u_M = ", u_M, ", a_SD = ", a_SD, ", a_DS = ", a_DS, ", a_M = ", a_M, ", a_MSC = ", a_MSC, ", k_S_ref = ", k_S_ref, ", k_D_ref = ", k_D_ref, ", k_M_ref = ", k_M_ref, ", Ea_S = ", Ea_S, ", Ea_D = ", Ea_D, ", Ea_M = ", Ea_M);

  // Likelihood evaluation.
  for (i in 1:state_dim+1) {
    y[i,] ~ normal(x_hat_add_CO2[i,], obs_error_scale * mean(x_hat_add_CO2[i,]));
  }
}

generated quantities {
  array[N_t] vector<lower=0>[state_dim] x_hat_post_pred_intmd;
  array[state_dim] vector<lower=0>[N_t] x_hat_post_pred;
  vector<lower=0>[N_t] x_hat_post_pred_CO2;
  array[state_dim+1] vector<lower=0>[N_t] x_hat_post_pred_add_CO2;
  array[state_dim+1, N_t] real<lower=0> y_hat_post_pred;

  print("Iteration theta: ", "u_M = ", u_M, ", a_SD = ", a_SD, ", a_DS = ", a_DS, ", a_M = ", a_M, ", a_MSC = ", a_MSC, ", k_S_ref = ", k_S_ref, ", k_D_ref = ", k_D_ref, ", k_M_ref = ", k_M_ref, ", Ea_S = ", Ea_S, ", Ea_D = ", Ea_D, ", Ea_M = ", Ea_M);

  x_hat_post_pred_intmd = ode_rk45(CON_ODE, x_hat0, t0, ts, u_M, a_SD, a_DS, a_M, a_MSC, k_S_ref, k_D_ref, k_M_ref, Ea_S, Ea_D, Ea_M, temp_ref, temp_rise);
  // Transform posterior predictive model output to match observations y in dimensions, [state_dim, N_t].
  for (i in 1:N_t) {
    for (j in 1:state_dim) {
      x_hat_post_pred[j, i] = x_hat_post_pred_intmd[i, j];
    }
  }

  // Compute posterior predictive CO2.
  x_hat_post_pred_CO2 = calc_CON_CO2(ts, x_hat_post_pred, a_SD, a_DS, a_M, k_S_ref, k_D_ref, k_M_ref, Ea_S, Ea_D, Ea_M, temp_ref, temp_rise);   

  // Append CO2 vector to posterior predictive x_hat.
  x_hat_post_pred_add_CO2[1:3,] = x_hat_post_pred;
  x_hat_post_pred_add_CO2[4,] = x_hat_post_pred_CO2;

  // Add observation noise to posterior predictive model output to obtain posterior predictive samples.
  for (i in 1:state_dim+1) {
    y_hat_post_pred[i,] = normal_rng(x_hat_post_pred_add_CO2[i,], obs_error_scale * mean(x_hat_post_pred_add_CO2[i,]));
  }
  print("Iteration posterior predictive y observation: ", y_hat_post_pred);

  // Obtain prior predictive samples. 
  real u_M_prior_pred = normal_lb_ub_rng(u_M_prior_dist_params[1], u_M_prior_dist_params[1] * prior_scale_factor, u_M_prior_dist_params[2], u_M_prior_dist_params[3]);
  real a_SD_prior_pred = normal_lb_ub_rng(a_SD_prior_dist_params[1], a_SD_prior_dist_params[1] * prior_scale_factor, a_SD_prior_dist_params[2], a_SD_prior_dist_params[3]);
  real a_DS_prior_pred = normal_lb_ub_rng(a_DS_prior_dist_params[1], a_DS_prior_dist_params[1] * prior_scale_factor, a_DS_prior_dist_params[2], a_DS_prior_dist_params[3]);
  real a_M_prior_pred = normal_lb_ub_rng(a_M_prior_dist_params[1], a_M_prior_dist_params[1] * prior_scale_factor, a_M_prior_dist_params[2], a_M_prior_dist_params[3]);
  real a_MSC_prior_pred = normal_lb_ub_rng(a_MSC_prior_dist_params[1], a_MSC_prior_dist_params[1] * prior_scale_factor, a_MSC_prior_dist_params[2], a_MSC_prior_dist_params[3]);
  real k_S_ref_prior_pred = normal_lb_ub_rng(k_S_ref_prior_dist_params[1], k_S_ref_prior_dist_params[1] * prior_scale_factor, k_S_ref_prior_dist_params[2], k_S_ref_prior_dist_params[3]);
  real k_D_ref_prior_pred = normal_lb_ub_rng(k_D_ref_prior_dist_params[1], k_D_ref_prior_dist_params[1] * prior_scale_factor, k_D_ref_prior_dist_params[2], k_D_ref_prior_dist_params[3]);
  real k_M_ref_prior_pred = normal_lb_ub_rng(k_M_ref_prior_dist_params[1], k_M_ref_prior_dist_params[1] * prior_scale_factor, k_M_ref_prior_dist_params[2], k_M_ref_prior_dist_params[3]);
  real Ea_S_prior_pred = normal_lb_ub_rng(Ea_S_prior_dist_params[1], Ea_S_prior_dist_params[1] * prior_scale_factor, Ea_S_prior_dist_params[2], Ea_S_prior_dist_params[3]);
  real Ea_D_prior_pred = normal_lb_ub_rng(Ea_D_prior_dist_params[1], Ea_D_prior_dist_params[1] * prior_scale_factor, Ea_D_prior_dist_params[2], Ea_D_prior_dist_params[3]);
  real Ea_M_prior_pred = normal_lb_ub_rng(Ea_M_prior_dist_params[1], Ea_M_prior_dist_params[1] * prior_scale_factor, Ea_M_prior_dist_params[2], Ea_M_prior_dist_params[3]);
}
