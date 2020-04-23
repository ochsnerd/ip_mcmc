import numpy as np
import scipy.linalg as la

from abc import ABC, abstractmethod


class DistributionBase(ABC):
    @abstractmethod
    def sample(self, rng):
        ...


class GaussianDistribution(DistributionBase):
    def __init__(self, mean=0, covariance=1):
        mean = self._ensure_array(mean, ndim=1)
        covariance = self._ensure_array(covariance, ndim=2)

        assert covariance.shape == (mean.shape[0], mean.shape[0]), (
            "dimension error")

        self.mean = mean
        self.covariance = covariance
        self.L, _ = la.cho_factor(covariance, lower=True)

    def sample(self, rng):
        return rng.normal(loc=self.mean, scale=self.covariance)

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
