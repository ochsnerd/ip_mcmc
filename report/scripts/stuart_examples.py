import numpy as np
import matplotlib.pyplot as plt

from ip_mcmc import (MCMCSampler,
                     pCNAccepter, CountedAccepter,
                     pCNProposer,
                     EvolutionPotential,
                     GaussianDistribution)

from helpers import store_figure


# not sure yet if this should be part of the package, so
# keep it in script for now
class SyntheticModel():
    """
    Model based on eq 2.5 in Stuart 2010:
    y = G(u) + eta
    with eta ~ noise

    noise: GaussianDistribution
    """
    def __init__(self, observation_operator, noise, posterior):
        self.G = observation_operator
        self.eta = noise
        self.posterior = posterior

    def evolve(self, u):
        """
        Evolve the state u according to the model equation
        """
        return self.G(u)

    def observe(self, u, rng):
        """
        Do one observation of the model, ie.
        y = self.evolve(u) + eta,
        where eta ~ self.noise
        """
        self.y = self.G(u) + self.eta.sample(rng) 
        return self.y

    def noise_pdf(self, x):
        return self.eta(x)

    def posterior_pdf(self, u):
        return self.posterior(u, self.y)


def build_evolution_pCN_sampler(observation_operator, u, data, noise, prior, posterior, rng):
    potential = EvolutionPotential(observation_operator, data, noise)

    proposer = pCNProposer(beta=0.25, prior=prior)
    accepter = CountedAccepter(pCNAccepter(potential=potential))
    return MCMCSampler(proposer, accepter, rng)


def stuart_example_21():
    """
    Example 2.1 on page 460 in Stuart 2010
    """
    n = 1
    # take first n digits of pi
    g = np.array([int(x) for x in str(np.pi) if x != '.'])[:n]

    # take first n digits of e
    u = np.array([int(x) for x in str(np.e) if x != '.'])[:n]

    def G(u):
        return np.dot(g, u)

    prior_covariance = np.identity(n)
    prior = GaussianDistribution(mean=np.zeros_like(u),
                                 covariance=prior_covariance)

    gamma = 0.5
    noise = GaussianDistribution(mean=0, covariance=gamma**2)

    # missing normalisation
    def posterior(u, y):
        likelihood_term = - 0.5 / gamma**2 * np.linalg.norm(y - G(u))**2
        prior_term = - 0.5 * np.dot(u, prior.apply_precision(u))
        return np.exp(likelihood_term + prior_term)

    model = SyntheticModel(observation_operator=G,
                           noise=noise,
                           posterior=posterior)

    rng = np.random.default_rng(1)

    data = model.observe(u, rng)
    print(f"y = {data}")

    sampler = build_evolution_pCN_sampler(observation_operator=G,
                                          u=u,
                                          data=data,
                                          noise=noise,
                                          prior=prior,
                                          posterior=posterior,
                                          rng=rng)

    u_0 = np.zeros_like(u)
    n_samples = 5000
    samples = sampler.run(u_0=u_0,
                          n_samples=n_samples)

    plt.hist(samples, bins=20, density=True)

    store_figure(f"stuart_example_21_n={n}_N={n_samples}")


def stuart_example_22():
    """
    Example 2.2 on page 461 in Stuart 2010
    """
    q = 2
    # take first q digits of pi
    g = np.array([int(x) for x in str(np.pi) if x != '.'])[:q]

    u = np.array([0.5])

    beta = 0
    def G(u):
        return g * (u + beta * np.array([u[0]**3]))

    prior_covariance = np.identity(1)
    prior = GaussianDistribution(mean=np.zeros_like(u),
                                 covariance=prior_covariance)

    gamma = 0.5
    noise = GaussianDistribution(mean=np.zeros(q), covariance=np.identity(q)*gamma**2)

    # missing normalisation
    def posterior(u, y):
        likelihood_term = - 0.5 / gamma**2 * np.linalg.norm(y - G(u))**2
        prior_term = - 0.5 * np.dot(u, prior.apply_precision(u))
        return np.exp(likelihood_term + prior_term)

    model = SyntheticModel(observation_operator=G,
                           noise=noise,
                           posterior=posterior)

    rng = np.random.default_rng(1)

    data = model.observe(u, rng)
    print(f"y = {data}")

    sampler = build_evolution_pCN_sampler(observation_operator=G,
                                          u=u,
                                          data=data,
                                          noise=noise,
                                          prior=prior,
                                          posterior=posterior,
                                          rng=rng)

    u_0 = np.zeros_like(u)
    n_samples = 5000
    samples = sampler.run(u_0=u_0,
                          n_samples=n_samples)

    plt.hist(samples, bins=20, density=True)

    store_figure(f"stuart_example_22q={q}_N={n_samples}")


def main():
    stuart_example_21()
    stuart_example_22()


if __name__ == '__main__':
    main()
