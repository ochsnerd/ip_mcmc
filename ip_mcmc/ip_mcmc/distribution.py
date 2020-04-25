import numpy as np
import scipy.linalg as la

from abc import ABC, abstractmethod
from scipy.stats import multivariate_normal


class DistributionBase(ABC):
    @abstractmethod
    def sample(self, rng):
        """Return a point sampled from this distribution"""
        ...

    @abstractmethod
    def __call__(self, x):
        """Return the value of the distribution at x"""
        ...


class GaussianDistribution(DistributionBase):
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
