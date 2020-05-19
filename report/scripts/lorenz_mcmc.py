import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from lorenz import Lorenz96


def avg_moment_function(y, K, J):
    """
    Average the moment function over the values given in y

    y.shape = ((J+1)*K, n_timesteps)
    """
    n_t = y.shape[1]

    f = np.empty((5 * K), n_t)

    # this could be a lot faster and a lot less clear
    for t in range(n_t):
        f[:K, t] = y[:K, t]

        for k in range(K):
            f[K + k, t] = np.mean(y[K + k*J, t])
            
        f[2*K:3*K, t] = y[:K]**2

        f[3*K:4*K, t] = y[:K, t] * f[K:2*K, t]

        f[4*K:5*K, t] = f[K:2*K, t]**2

    return np.mean(f, axis=1)


class LorenzObservationOperator:
    """
    Observation operator for MCMC based on Lorenz96
    """
    def __init__(self, K, J, T):
        self.K = K
        self.J = J

        self.IC = np.random.default_rng(1).random((J + 1) * K)

        self.T = T

    def __call__(self, u):
        """
        u: np.array(dtype=float)
             parameter vector, u = [F, h, c, b]

        evolve a Lorenz96-ODE with self.K, self.J and the parameters
        given in u from the initial conditions of the last
        run over time [0, self.T)
        """
        y = self._solve_ODE(Lorenz96(self.K, self.J, *u))
        self.IC = y
        return avg_moment_function(y, self.K, self.J)

    def _solve_ODE(self, dy_dt):
        return solve_ivp(fun=f, t_span=(0, self.T), y0=self.IC, method='RK45').y[-1]


def main():
    K, J = 6, 4
    moment_function_variances = np.random.random(5 * K)

    noise = GaussianDistribution(covariance=np.diag(moment_function_variances))

    # WHAT IS THIS, I WAS TOLD PRIORS WILL ALWAYS BE GAUSSIANS
    # AND HERE THEY USE A LOGNORMAL REEEEEE
    prior_means = np.array([10, 0, ?, 5])
    prior_variance = np.array([10, 1, ?, 10])
    prior = GaussianDistribution(mean=prior_means, covariance=np.diag(prior_variance))

    
