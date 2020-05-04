import numpy as np

from abc import ABC, abstractmethod


class PotentialBase(ABC):
    """
    Potential used to express the likelihood;
    d mu(u; y) / d mu_0(u) \propto L(u; y)
    Write L(u; y) as exp(-potential(u; y))
    """
    @abstractmethod
    def __call__(self, u):
        ...

    @abstractmethod
    def exp_minus_potential(self, u):
        """
        Return exp(-potential(u))
        This is sometimes easier to compute
        """
        ...


class AnalyticPotential(PotentialBase):
    """
    Potential used to sample from an analytical pdf using
    a Bayes' formulation for Accepters
    (i.e. we know the posterior but not the likelihood)
    """
    def __init__(self, posterior, prior):
        self.posterior = posterior
        self.prior = prior

    def __call__(self, u):
        return -np.log(self.exp_minus_potential(u))

    def exp_minus_potential(self, u):
        return self.posterior(u) / self.prior(u)


class EvolutionPotential(PotentialBase):
    """
    Potential resulting from a model equation
    data = observation_operator(u) + eta,
    where the noise-term eta ~ noise_distribution
    """
    def __init__(self, observation_operator, data, noise_distribution):
        self.G = observation_operator
        self.y = data
        self.rho = noise_distribution

    def __call__(self, u):
        return -np.log(self.exp_minus_potential(u))

    def exp_minus_potential(self, u):
        return self.rho(self.y - self.G(u))
