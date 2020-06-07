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
    Time-average the moment function over the values given in y

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
    def __init__(self, K, J, T, c, prior_means, IC):
        self.K = K
        self.J = J
        self.T = T
        self.c = c
        self.prior_means = prior_means
        self.IC = IC

    def __call__(self, u):
        """
        u: np.array(dtype=float)
             parameter vector, u = [F, h, b]

        evolve a Lorenz96-ODE with self.K, self.J and the parameters
        given in u from the initial conditions of the last
        run over time [0, self.T)
        """
        F, h, b = self.prior_means + u
        y = self._solve_ODE(Lorenz96(self.K, self.J, F, h, self.c, b))
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
    sim_length = 20

    # True Theta
    theta = np.array([10, 10, 1, 10])  # F, h, c, b
    r = 0.5  # noise level

    # Characteristics of system
    T = 500
    try:
        Y = np.load(data_dir + f"Y_{K=}_{J=}_{T=}.npy")
        print("Loaded existing simulation results")
    except FileNotFoundError:
        print("Running simulation to generate fake measurements")
        Y = run_lorenz96(K, J, theta, T)
        np.save(data_dir + f"Y_{K=}_{J=}_{T=}", Y)

    print(f"{Y.shape=}")

    moment_function_values = moment_function(Y, K, J)
    moment_function_means = np.mean(moment_function_values, axis=1)
    moment_function_variances = np.var(moment_function_values, axis=1)
    print(f"{moment_function_variances.shape=}")

    noise = GaussianDistribution(mean=np.zeros_like(moment_function_variances),
                                 covariance=r**2 * np.diag(moment_function_variances))

    prior_means = np.array([12, 8, 9])  # F, h, b
    prior_covariance = np.diag([10, 1, 10])

    # From theory: prior is always assumed to be centered,
    # so I do MCMC over pertubations from given prior means
    prior = GaussianDistribution(np.zeros_like(prior_means), prior_covariance)

    observation_operator = LorenzObservationOperator(K, J,
                                                     sim_length,
                                                     theta[2],
                                                     prior_means,
                                                     Y[:, -1])

    # don't need huge array anymore
    del Y

    potential = EvolutionPotential(observation_operator,
                                   moment_function_means,
                                   noise)

    proposer = pCNProposer(beta=0.5, prior=prior)
    accepter = CountedAccepter(pCNAccepter(potential=potential))

    sampler = MCMCSampler(proposer, accepter, rng)

    u_0 = np.array([-1.9, 1.9, 0.9])  # start close to true theta
    n_samples = 2000
    try:
        samples = np.load(data_dir + f"S_{K=}_{J=}_T={sim_length}_{r=}_{n_samples=}.npy")
        print("Loaded existing sampling results")
    except FileNotFoundError:
        print("Generating samples")
        samples = sampler.run(u_0=u_0,
                              n_samples=n_samples,
                              burn_in=100,
                              sample_interval=1)
        np.save(data_dir + f"S_{K=}_{J=}_T={sim_length}_{r=}_{n_samples=}", samples)

    print(f"{samples.shape=}")

    # to conform with the output-shape of
    # solve_ivp. Might be worthwile to change it in the sampler,
    # but then I break older scripts
    samples = samples.T

    # Add pertubations to means
    for i in range(len(samples[0, :])):
        samples[:, i] += prior_means

    # Plot densities
    priors = [GaussianDistribution(mu, np.sqrt(sigma_sq))
              for mu, sigma_sq in zip(prior_means, np.diag(prior_covariance))]
    intervals = [(-5, 25)] * 3
    names = ["F", "h", "b"]

    fig, plts = plt.subplots(1, 3, figsize=(20,10))

    plot_info = zip(priors,
                    intervals,
                    [theta[0], theta[1], theta[3]],
                    names,
                    plts)

    for i, (prior, interval, true_val, name, ax) in enumerate(plot_info):
        ax.hist(samples[i, :], density=True)
        x_range = np.linspace(*interval)
        ax.plot(x_range, [prior(x) for x in x_range])
        ax.axvline(true_val, c='r')
        ax.set_title(f"Prior and posterior for {name}")
        ax.set(xlabel=name, ylabel="Probability")

    fig.suptitle("Posteriors and priors")
    store_figure(f"combined_{K=}_{J=}_T={sim_length}_{r=}")

    # Average the autocorrelation
    ac = np.zeros((3, 100))
    n = 10
    for i in range(1,1+n):
        for var in range(3):
            ac[var, :] += MCMCSampler.autocorr(samples[var, i*100:(i+1)*100])
    ac /= n
    plt.plot(ac[0, :], label="F")
    plt.plot(ac[1, :], label="h")
    plt.plot(ac[2, :], label="b")
    plt.title("Autocorrelation")
    plt.xlabel("Lag")
    plt.legend()
    store_figure(f"lorenz_ac_avg_{K=}_{J=}_T={sim_length}_{r=}")


def store_figure(name):
    """
    Store a figure in the figures directory.
    Assumes there is an active pyplot-Plot and clears it after
    """
    plt.savefig("/home/david/fs20/thesis/code/report/figures/" + name + ".svg", format='svg')
    plt.clf()


if __name__ == '__main__':
    main()
