import numpy as np

from abc import ABC, abstractmethod

from .distribution import GaussianDistribution


class ProposerBase(ABC):
    @abstractmethod
    def __call__(self, u, rng):
        ...


class StandardRWProposer(ProposerBase):
    """Propose a new state as
    v = u + sqrt(2*delta) * w, w ~ N(0,C)

    w has the same covariance as the prior, but is mean 0

    (4.3) in MCMCMF
    """
    def __init__(self, delta, prior):
        self.prefactor = np.sqrt(2*delta)

        self.w = GaussianDistribution(mean=np.zeros_like(prior.mean),
                                      covariance=prior.covariance)

    def __call__(self, u, rng):
        return u + self.prefactor * self.w.sample(rng)


class pCNProposer(ProposerBase):
    """Propose a new state as 
    v = sqrt(1-beta^2) * u + beta * w, w ~ N(0,C)

    w has the same covariance as the prior, but is mean 0

    (4.8) in MCMCMF
    """
    def __init__(self, beta, prior):
        """
        beta: float
        prior: GaussianDistribution

        Only the covariance of the prior is used, a non-zero mean is ignored
        """
        assert 0 <= beta <= 1, "beta has to be in [0,1]"
        self.beta = beta
        self.contraction = np.sqrt(1 - beta ** 2)
        self.w = GaussianDistribution(mean=np.zeros_like(prior.mean),
                                      covariance=prior.covariance)

    def __call__(self, u, rng):
        return self.contraction * u + self.beta * self.w.sample(rng)
