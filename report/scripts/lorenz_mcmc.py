import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from ip_mcmc import (MCMCSampler,
                     pCNAccepter, CountedAccepter,
                     pCNProposer,
                     EvolutionPotential,
                     GaussianDistribution, LogNormalDistribution, IndependentDistributions)

import sys
sys.path.append("/home/david/fs20/thesis/code/report/scripts/")
from lorenz import Lorenz96


def moment_function(y, K, J):
    """
    Average the moment function over the values given in y

    y.shape = ((J+1)*K, n_timesteps)
    """
    n_t = y.shape[1]

    f = np.empty(((5 * K), n_t))

    # this could be a lot faster and a lot less clear
    for t in range(n_t):
        f[:K, t] = y[:K, t]

        for k in range(K):
            f[K + k, t] = np.mean(y[K + k*J, t])

        f[2*K:3*K, t] = y[:K, t]**2

        f[3*K:4*K, t] = y[:K, t] * f[K:2*K, t]

        f[4*K:5*K, t] = f[K:2*K, t]**2

    return f


class LorenzObservationOperator:
    """
    Observation operator for MCMC based on Lorenz96
    """
    def __init__(self, K, J, T, IC):
        self.K = K
        self.J = J
        self.IC = IC
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
        self.IC = y[:, -1]
        return np.mean(moment_function(y, self.K, self.J), axis=1)

    def _solve_ODE(self, l):
        return solve_ivp(fun=l, t_span=(0, self.T), y0=self.IC, method='RK45').y


def run_lorenz96(K, J, theta, T):
    l = Lorenz96(K, J, *theta)

    IC = np.random.random(K * (J + 1))

    return solve_ivp(fun=l, t_span=(0, T), y0=np.random.default_rng(1).random((J + 1) * K)).y


def main():
    rng = np.random.default_rng(1)
    data_dir = "/home/david/fs20/thesis/code/report/data/"

    # Parameters of simulation
    K, J = 6, 4
    sim_length = 2

    # True Theta
    theta = [10, 10, 1, 10]  # F, h, c, b

    # Characteristics of system
    T = 100
    try:
        Y = np.load(data_dir + f"Y_{K=}_{J=}_{T=}.npy")
        print("Loaded existing simulation results")
    except FileNotFoundError:
        print("Running simulation to generate fake measurements")
        Y = run_lorenz96(K, J, theta, T)
        np.save(data_dir + f"Y_{K=}_{J=}_{T=}", Y)

    print(f"{Y.shape=}")
    data = moment_function(Y[:, -1].reshape((J + 1) * K, 1), K, J).flatten()
    print(f"{data.shape=}")
    moment_function_variances = np.var(moment_function(Y, K, J), axis=1)
    print(f"{moment_function_variances.shape=}")

    noise = GaussianDistribution(mean=np.zeros_like(moment_function_variances),
                                 covariance=np.diag(moment_function_variances))

    F_prior = GaussianDistribution(10, np.sqrt(10))
    h_prior = GaussianDistribution(0, np.sqrt(1))
    c_prior = LogNormalDistribution(2, np.sqrt(.1))
    b_prior = GaussianDistribution(5, np.sqrt(10))

    prior = IndependentDistributions((F_prior, h_prior, c_prior, b_prior))

    potential = EvolutionPotential(LorenzObservationOperator(K, J, sim_length, Y[:, -1]),
                                   data,
                                   noise)

    # don't need huge array anymore
    del Y

    proposer = pCNProposer(beta=0.25, prior=prior)
    accepter = CountedAccepter(pCNAccepter(potential=potential))

    sampler = MCMCSampler(proposer, accepter, rng)

    try:
        samples = np.load(data_dir + f"S_{K=}_{J=}_T={sim_length}.npy")
        print("Loaded existing sampling results")
    except FileNotFoundError:
        print("Generating samples")
        samples = sampler.run(u_0=np.zeros((4,)),
                              n_samples=100,
                              burn_in=10,
                              sample_interval=2)
        np.save(data_dir + f"S_{K=}_{J=}_T={sim_length}", samples)

    print(f"{samples.shape=}")

    # to conform with the output-shape of
    # solve_ivp. Might be worthwile to change it in the sampler,
    # but then I break older scripts
    samples = samples.T

    plt.hist(samples[0, :], density=True)
    x_range = np.linspace(0, 20)
    plt.plot(x_range, [F_prior(x) for x in x_range])
    plt.show()


if __name__ == '__main__':
    main()
