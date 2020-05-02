import numpy as np

from abc import ABC, abstractmethod

from .distribution import GaussianDistribution


class ProposerBase(ABC):
    @abstractmethod
    def __call__(self, u, rng):
        ...


class StandardRWProposer(ProposerBase):
    """Propose a new state as v = u + sqrt(2*delta*covariance) * epsilon,
    with epsilon ~ N(0,I)

    (4.3) in MCMCMF
    """
    def __init__(self, delta, dims, sqrt_covariance=None):
        self.prefactor = np.sqrt(2*delta)

        # TODO: Can we use GaussianDistribution better?
        self.epsilon = GaussianDistribution(mean=np.zeros(dims),
                                            covariance=np.identity(dims))

        if sqrt_covariance is not None:
            self.prefactor *= sqrt_covariance

    def __call__(self, u, rng):
        # Scalar case is uncommon enough to not warrant if-statement
        try:
            return u + self.prefactor @ self.epsilon.sample(rng)
        except (ValueError, TypeError):
            return u + self.prefactor * self.epsilon.sample(rng)


class pCNProposer(ProposerBase):
    """Propose a new state as 
    v = sqrt(1-beta^2) * u + beta * w, w ~ N(0,C) (= prior)

    (4.8) in MCMCMF
    """
    def __init__(self, beta, prior):
        """
        beta: float
        prior: GaussianDistribution
        """
        assert 0 <= beta <= 1, "beta has to be in [0,1]"
        self.beta = beta
        self.contraction = np.sqrt(1 - beta ** 2)
        self.w = prior

    def __call__(self, u, rng):
        return self.contraction * u + self.beta * self.w.sample(rng)
