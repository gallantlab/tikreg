import numpy as np

from tikypy import spatial_priors as sp


def test_prior_from_penalty():
    tmp = np.random.randn(10, 10)
    raw_penalty = np.dot(tmp, tmp.T)
    prior = sp.PriorFromPenalty(raw_penalty)
    prior.get_prior()
