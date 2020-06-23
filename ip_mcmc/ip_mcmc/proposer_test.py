import numpy as np

from .proposer import ConstStepStandardRWProposer, ConstSteppCNProposer
from .distribution import GaussianDistribution
from .test_utilities import MockRNG


def test_ConstStepStandardRWProposer_Scalar():
    rng = MockRNG(-np.pi)
    delta = np.e
    prior = GaussianDistribution(100, np.sin(np.e))

    p = ConstStepStandardRWProposer(delta, prior)

    assert np.isclose(p.prefactor, np.sqrt(2*delta)), ""
    assert np.isclose(p(-np.e, rng), -np.e - p.prefactor * np.pi), ""


def test_ConstStepStandardRWProposer_Multivariate():
    rng = MockRNG(np.array([1,2]))
    delta = np.pi
    prior = GaussianDistribution(np.array([-1,2]), np.array([[1, 0],[0, 4]]))

    p = ConstStepStandardRWProposer(delta, prior)

    assert np.isclose(
        p.prefactor,
        np.sqrt(2*delta)
    ).all(), ""
    assert np.isclose(
        p(np.array([-1, 2]), rng),
        np.array([-1 + p.prefactor, 2 + 2 * p.prefactor])
    ).all(), ""


def test_ConstSteppCNProposer_Scalar():
    rng = MockRNG(-np.pi)
    beta = 1 / np.e
    prior = GaussianDistribution(mean=0, covariance=2)

    p = ConstSteppCNProposer(beta, prior)

    assert np.isclose(p.contraction, np.sqrt(1 - beta**2)), ""
    assert np.isclose(p(-np.pi, rng),
                      np.sqrt(1 - beta**2) * (-np.pi) + beta * -np.pi), ""


def test_ConstSteppCNProposer_Multivariate():
    rng = MockRNG(np.array([2, -1]))
    beta = 0.5
    prior = GaussianDistribution(mean=np.array([0, 0]),
                                 covariance=np.array([[2, 0.5], [0.5, 1]]))

    p = ConstSteppCNProposer(beta, prior)

    u = np.array([1, 1])
    assert np.isclose(p(u, rng),
                      np.sqrt(1 - beta**2) * u + beta * np.array([2, -1])).all(), ""
