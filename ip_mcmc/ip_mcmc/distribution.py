import numpy as np
import scipy.linalg as la

from abc import ABC, abstractmethod
from scipy.stats import multivariate_normal, lognorm


class DistributionBase(ABC):
    # Classes deriving from this are expected to define a
    # attribute k for the dimension of the distribution.
    # This would be propertly implemented in python
    # by using the property attribute.
    @abstractmethod
    def sample(self, rng):
        """Return a point sampled from this distribution"""
        ...

    @abstractmethod
    def __call__(self, x):
        """Return the value of the distribution at x"""
        ...

    @abstractmethod
    def logpdf(self, x):
        """Return the log of value of the distribution at x"""
        ...


class IndependentDistributions(DistributionBase):
    """
    Multiple (possibly different) distributions
    (could for example build a multivariate normal with
    block-diagonal covaraince matrix by combining multiple
    GaussianDistribution instances)
    """
    def __init__(self, distributions):
        self.distributions = distributions

        self.k = sum(dist.k for dist in distributions)

    def __call__(self, x):
        a = 1
        k = 0
        for dist in self.distributions:
            a *= dist(x[k:k+dist.k])
            k += dist.k
        return a

    def logpdf(self, x):
        a = 1
        k = 0
        for dist in self.distributions:
            a *= dist.logpdf(x[k:k+dist.k])
            k += dist.k
        return a

    def sample(self, rng):
        # Convert scalars to arrays before concatenation
        return np.concatenate([np.array(dist.sample(rng), ndmin=1) for dist in self.distributions])


class LogNormalDistribution(DistributionBase):
    """
    1D lognormal distribution
    """
    def __init__(self, mu, sigma):
        """
        Cosntruct distribution of Y st.
        Y ~ lognormal with Y = exp(X), where X ~ N(mu, sigma^2)
        """
        # parameters of the underlying normal
        self.mu = mu
        self.s = sigma

        self.dist = lognorm(scale=np.exp(self.mu), s=self.s)

        # Scalar distribution
        self.k = 1

    def __call__(self, x):
        return self.dist.pdf(x)

    def logpdf(self, x):
        return self.dist.logpdf(x)

    def sample(self, rng):
        return rng.lognormal(mean=self.mu, sigma=self.s)


class GaussianDistribution(DistributionBase):
    # This really needs to be reworked
    # - Seperate multivariate and scalar case
    # - maybe even throw in a scalar baseclass
    def __init__(self, mean=0, covariance=1):
        mean = self._ensure_array(mean, ndim=1)
        covariance = self._ensure_array(covariance, ndim=2)
        self.k = mean.shape[0]

        assert covariance.shape == (self.k, self.k), (
            "dimension error")

        self.mean = mean
        self.covariance = covariance
        self.L, _ = la.cho_factor(covariance, lower=True)

        self.dist = multivariate_normal(mean=self.mean, cov=self.covariance)

    def __call__(self, x):
        return self.dist.pdf(x)

    def logpdf(self, x):
        return self.dist.logpdf(x)

    def sample(self, rng):
        # this is actually the function that scipy.stats.multivariate_normal
        # calls to generate its realizations
        return rng.multivariate_normal(mean=self.mean,
                                       cov=self.covariance)

    def apply_covariance(self, x):
        x = self._ensure_array(x)
        return self.covariance @ x

    def apply_sqrt_covariance(self, x):
        x = self._ensure_array(x)
        return self.L @ x

    def apply_precision(self, x):
        x = self._ensure_array(x)
        return la.cho_solve((self.L, True), x)

    def apply_sqrt_precision(self, x):
        x = self._ensure_array(x)
        return la.solve_triangular(self.L.T, x, lower=False)

    def _ensure_array(self, x, ndim=1):
        if np.isscalar(x):
            return np.array([x], ndmin=ndim)

        if isinstance(x, list):
            x = np.array(x)

        assert len(x.shape) == ndim, (
            f"Dimension error: {len(x.shape)} instead of {ndim}.")

        return x
