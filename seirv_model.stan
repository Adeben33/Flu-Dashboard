functions {
  array[] real seirv_ode(real t, array[] real y, array[] real theta, array[] real x_r, array[] int x_i) {
    real S = y[1];
    real V = y[2];
    real E = y[3];
    real I = y[4];
    real R = y[5];
    real Cum_Inc = y[6];

    real beta = theta[1];
    real eta = theta[2];
    real phi = theta[3];
    real mu = theta[4];
    real d = theta[5];
    real k1 = theta[6];
    real delta = theta[7];
    real sigma = theta[8];

    real inf_force = beta * (I + eta * E);

    real dS_dt = - inf_force * S - (phi + mu) * S;
    real dV_dt = phi * S - (1 - sigma) * inf_force * V - mu * V;
    real dE_dt = inf_force * S + (1 - sigma) * inf_force * V - k1 * E - mu * E;
    real dI_dt = k1 * E - d * I - delta * I - mu * I;
    real dR_dt = delta * I - mu * R;
    real dCum_Inc_dt = k1 * E;

    return {dS_dt, dV_dt, dE_dt, dI_dt, dR_dt, dCum_Inc_dt};
  }
}

data {
  int<lower=1> T;
  array[T] real ts;
  real t0;
  array[6] real initial_state;
  array[7] real parameter;
  array[T] int<lower=0> incidence;
}

transformed data {
  array[0] real x_r;
  array[0] int x_i;
}

parameters {
  real<lower=0.1, upper=0.25> sigma;
  real<lower=0> phi_b;
  real<lower=0.1, upper=0.4> delta;
  real<lower=0.2, upper=0.3> k1;
  real<lower=0.001, upper=0.002> d;
  real<lower=0.0003, upper=0.0006> phi;
  real<lower=0.2, upper=0.4> eta;
}

transformed parameters {
  array[8] real theta;
  theta[1] = parameter[1];
  theta[2] = eta;
  theta[3] = phi;
  theta[4] = parameter[4];
  theta[5] = d;
  theta[6] = k1;
  theta[7] = delta;
  theta[8] = sigma;

  real phi_inv = 1. / phi_b;

  array[T, 6] real y_hat = integrate_ode_rk45(seirv_ode, initial_state, t0, ts, theta, x_r, x_i);
  array[T] real Incidence_total;

  for (t in 1:T)
    Incidence_total[t] = (t == 1) ? y_hat[t, 6] : y_hat[t, 6] - y_hat[t - 1, 6];
}

model {
  sigma ~ lognormal(-0.2544, 0.003);
  phi_b ~ exponential(30);
  delta ~ lognormal(log(0.15), 0.001);
  k1 ~ lognormal(log(0.2), 0.005);
  d ~ lognormal(log(0.001), 0.0001);
  phi ~ lognormal(log(0.0002), 0.00002);
  eta ~ lognormal(log(0.33), 0.2);

  incidence ~ neg_binomial_2(Incidence_total, 1. / phi_b);
}

generated quantities {
  array[T] int pred_cases;
  for (t in 1:T)
    pred_cases[t] = neg_binomial_2_rng(Incidence_total[t], 1. / phi_b);
}
