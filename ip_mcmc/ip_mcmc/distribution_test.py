import numpy as np

from .distribution import GaussianDistribution
from .test_utilities import MockRNG


def test_GaussianDistributionScalar():
    rng = MockRNG(0)
    g = GaussianDistribution(mean=1, covariance=2)

    assert np.isclose(g.sample(rng), 0), ""
    assert np.isclose(g.apply_covariance(1), 2), ""
    assert np.isclose(g.apply_sqrt_covariance(1), np.sqrt(2)), ""
    assert np.isclose(g.apply_precision(1), .5), ""
    assert np.isclose(g.apply_sqrt_precision(1), np.sqrt(.5)), ""


def test_GaussianDistributionMultivariate():
    g = GaussianDistribution(mean=np.array([1,1,1]), covariance=np.diagflat([1,2,3]))

    assert all(np.isclose([1,2,3], g.apply_covariance([1,1,1]))), ""
    assert all(np.isclose([1,np.sqrt(2),np.sqrt(3)], g.apply_sqrt_covariance([1,1,1]))), ""
    assert all(np.isclose([1,1/2,1/3], g.apply_precision([1,1,1]))), ""
    assert all(np.isclose([1,1/np.sqrt(2),1/np.sqrt(3)], g.apply_sqrt_precision([1,1,1]))), ""
