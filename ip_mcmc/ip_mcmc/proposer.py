import numpy as np

from abc import ABC, abstractmethod

from .distribution import GaussianDistribution


class ProposerBase(ABC):
    @abstractmethod
    def __call__(self, u, rng):
        ...


class ConstStepStandardRWProposer(ProposerBase):
    """Propose a new state as
    v = u + sqrt(2*delta) * w, w ~ N(0,C)

    w has the same covariance as the prior, but is mean 0.
    delta is constant.

    (4.3) in MCMCMF
    """
    def __init__(self, delta, prior):
        self.prefactor = np.sqrt(2*delta)

        self.w = GaussianDistribution(mean=np.zeros_like(prior.mean),
                                      covariance=prior.covariance)

    def __call__(self, u, rng):
        return u + self.prefactor * self.w.sample(rng)


class VarStepStandardRWProposer(ProposerBase):
    """Propose a new state as
    v = u + sqrt(2*delta) * w, w ~ N(0,C)

    w has the same covariance as the prior, but is mean 0.
    delta is based on how many proposals have been made up to
    now (should equal the iteration of the MCMC algorithm,
    so don't reuse proposers!)

    (4.3) in MCMCMF
    """
    def __init__(self, delta, prior):
        self.prefactor = np.sqrt(2)
        self.delta = delta

        self.i = 0

        self.w = GaussianDistribution(mean=np.zeros_like(prior.mean),
                                      covariance=prior.covariance)

    def __call__(self, u, rng):
        self.i += 1
        stepsize = self.prefactor * np.sqrt(self.delta(self.i))
        return u + stepsize * self.w.sample(rng)


class ConstSteppCNProposer(ProposerBase):
    """Propose a new state as
    v = sqrt(1-beta^2) * u + beta * w, w ~ N(0,C)

    w has the same covariance as the prior, but is mean 0.
    beta is constant.

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


class VarSteppCNProposer(ProposerBase):
    """Propose a new state as
    v = sqrt(1-beta^2) * u + beta * w, w ~ N(0,C)

    w has the same covariance as the prior, but is mean 0.
    beta is based on how many proposals have been made up to
    now (should equal the iteration of the MCMC algorithm,
    so don't reuse proposers!)

    (4.8) in MCMCMF
    """
    def __init__(self, beta, prior):
        """
        beta: callable
        prior: GaussianDistribution

        Only the covariance of the prior is used, a non-zero mean is ignored
        """
        self.beta = beta

        self.i = 0

        self.w = GaussianDistribution(mean=np.zeros_like(prior.mean),
                                      covariance=prior.covariance)

    def __call__(self, u, rng):
        self.i += 1
        b = self.beta(self.i)
        contraction = np.sqrt(1 - b**2)

        return contraction * u + b * self.w.sample(rng)
