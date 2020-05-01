import numpy as np

from .proposer import StandardRWProposer, pCNProposer
from .test_utilities import MockRNG


def test_StandardRWProposer_Scalar():
    rng = MockRNG(-np.pi)
    delta = np.e
    sqrt_cov = np.sin(np.e)

    p = StandardRWProposer(delta, 1, sqrt_cov)

    assert np.isclose(p.prefactor, np.sqrt(2*delta) * sqrt_cov), ""
    assert np.isclose(p(-np.e, rng), -np.e - p.prefactor * np.pi), ""


def test_StandardRWProposer_Multivariate():
    rng = MockRNG(np.array([1,2]))
    delta = np.pi
    sqrt_cov = np.array([[1, 0],[0, 2]])

    p = StandardRWProposer(delta, 2, sqrt_cov)

    assert np.isclose(
        p.prefactor,
        np.diagflat([np.sqrt(2 * delta), 2 * np.sqrt(2 * delta)])
    ).all(), ""
    assert np.isclose(
        p(np.array([-1, 2]), rng),
        np.array([-1 + p.prefactor[0, 0], 2 + 2 * p.prefactor[1, 1]])
    ).all(), ""


def test_pCNProposer_Scalar():
    rng = MockRNG(-np.pi)
    beta = 1 / np.e
    covariance = np.array([2], ndmin=2)

    p = pCNProposer(beta, covariance)

    assert np.isclose(p.contraction, np.sqrt(1 - beta**2)), ""
    assert np.isclose(p(-np.pi, rng),
                      np.sqrt(1 - beta**2) * (-np.pi) + beta * -np.pi), ""



def test_pCNProposer_Scalar():
    rng = MockRNG(np.array([2, -1]))
    beta = 0.5
    covariance = np.array([[2, 0.5], [0.5, 1]])

    p = pCNProposer(beta, covariance)

    u = np.array([1, 1])
    assert np.isclose(p(u, rng),
                      np.sqrt(1 - beta**2) * u + beta * np.array([2, -1])).all(), ""
