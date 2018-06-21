import numpy as np

import tikreg.temporal_priors as tp
import tikreg.utils as tikutils

##############################
#
##############################



def test_base_prior():
    tmp = np.random.randn(10, 10)
    raw_prior = np.dot(tmp, tmp.T)
    prior = tp.BasePrior(raw_prior, dodetnorm=False)

    # basic defaults
    assert prior.detnorm == 1.0
    assert np.allclose(prior.asarray, raw_prior)
    # return penalty
    penalty = np.linalg.inv(raw_prior)
    assert np.allclose(prior.prior2penalty(dodetnorm=False), penalty)


    #####
    prior = tp.BasePrior(raw_prior, dodetnorm=True)
    # test determinant normalizer computation
    detnorm = tikutils.determinant_normalizer(raw_prior)
    assert prior.detnorm == detnorm
    # test determinant normalization
    rr = raw_prior / tikutils.determinant_normalizer(raw_prior)
    assert np.allclose(prior.asarray, rr)

    #####
    prior = tp.BasePrior(raw_prior, dodetnorm=False)
    # test penalty computation
    penalty = np.linalg.inv(raw_prior)
    assert np.allclose(prior.prior2penalty(dodetnorm=False), penalty)
    assert prior.penalty_detnorm == 1.0
    # test penalty with normalization
    penalty_detnorm = prior.prior2penalty(dodetnorm=True)
    assert prior.penalty_detnorm == 1.0

    # update class
    prior.normalize_penalty()
    pdetnorm = tikutils.determinant_normalizer(penalty)
    assert prior.penalty_detnorm == pdetnorm
    assert np.allclose(penalty_detnorm, penalty / pdetnorm)

def test_temporal_prior():
    ndelays = 10
    tmp = np.random.randn(ndelays, ndelays)
    raw_prior = np.dot(tmp, tmp.T)
    prior = tp.TemporalPrior(raw_prior)
    delays = np.arange(ndelays)

    assert hasattr(prior, 'ndelays')
    assert hasattr(prior, 'delays')
    assert isinstance(prior.asarray, np.ndarray)
    assert prior.ndelays == ndelays
    assert len(prior.delays) == prior.ndelays
    assert np.allclose(prior.delays, delays)


def test_spherical_prior():
    prior = tp.SphericalPrior()
    delays = range(10)
    ndelays = len(delays)
    assert prior.asarray.shape[0] == prior.ndelays

    prior = tp.SphericalPrior(delays)
    a,b = prior.asarray.shape
    assert a == b == len(prior.delays)
    assert np.allclose(prior.asarray, np.eye(ndelays))

    delays = [1,2,3,8,10]
    prior = tp.SphericalPrior(delays)
    assert np.allclose(prior.asarray, np.eye(len(delays)))


def test_hrf_prior():

    H = tikutils.hrf_default_basis(dt=2.0, duration=20.)
    raw_prior = np.dot(H, H.T).astype(np.float64)

    # test initialization
    prior = tp.HRFPrior()
    assert np.allclose(prior.asarray, raw_prior)

    # test delay seletion
    delays = np.arange(10)
    tt = tikutils.fast_indexing(raw_prior, delays, delays)
    prior = tp.HRFPrior()
    assert np.allclose(prior.asarray, tt)


    # change temporal resolution
    H = tikutils.hrf_default_basis(dt=1.0, duration=20.)
    raw_prior = np.dot(H, H.T).astype(np.float64)
    prior = tp.HRFPrior(dt=1.0)
    assert np.allclose(prior.asarray, raw_prior)

    # change total duration
    H = tikutils.hrf_default_basis(dt=2.0, duration=40.)
    raw_prior = np.dot(H, H.T).astype(np.float64)
    prior = tp.HRFPrior(duration=40)
    assert np.allclose(prior.asarray, raw_prior)

    # change resolution and duratoin
    H = tikutils.hrf_default_basis(dt=1.0, duration=40.)
    raw_prior = np.dot(H, H.T).astype(np.float64)
    prior = tp.HRFPrior(dt=1.0, duration=40)
    assert np.allclose(prior.asarray, raw_prior)

    # change resolution and duration and subselect delays
    H = tikutils.hrf_default_basis(dt=1.0, duration=40.)
    raw_prior = np.dot(H, H.T).astype(np.float64)
    delays = np.arange(1, 40)
    tt = tikutils.fast_indexing(raw_prior, delays, delays)
    prior = tp.HRFPrior(dt=1.0, duration=40, delays=delays)
    assert np.allclose(prior.asarray, tt)

    # change resolution and duration and subselect disjoint delays
    H = tikutils.hrf_default_basis(dt=1.0, duration=40.)
    raw_prior = np.dot(H, H.T).astype(np.float64)
    delays = np.asarray([1,2,10,30,35])
    tt = tikutils.fast_indexing(raw_prior, delays, delays)
    prior = tp.HRFPrior(dt=1.0, duration=40, delays=delays)
    assert np.allclose(prior.asarray, tt)


    # grab continuous delays
    H = tikutils.hrf_default_basis(dt=2.0, duration=20.)
    raw_prior = np.dot(H, H.T).astype(np.float64)

    delays = np.arange(1,10)
    tt = tikutils.fast_indexing(raw_prior, delays, delays)
    prior = tp.HRFPrior(delays)
    assert np.allclose(prior.asarray, tt)
    assert np.allclose(prior.delays, delays)
    assert np.allclose(prior.ndelays, len(delays))

    # grab disjoint delays

    delays = np.asarray([1,3,6,9])
    tt = tikutils.fast_indexing(raw_prior, delays, delays)
    prior = tp.HRFPrior(delays)
    assert np.allclose(prior.asarray, tt)
    assert np.allclose(prior.delays, delays)
    assert np.allclose(prior.ndelays, len(delays))


def test_prior_from_penalty():
    tmp = np.random.randn(10, 10)
    raw_penalty = np.dot(tmp, tmp.T)
    prior = tp.PriorFromPenalty(raw_penalty, dodetnorm=False)

    assert np.allclose(raw_penalty, prior.penalty)

    penalty = prior.prior2penalty(dodetnorm=False)
    assert np.allclose(raw_penalty, penalty)

    penalty = prior.prior2penalty(dodetnorm=True)
    pdetnorm = tikutils.determinant_normalizer(raw_penalty)
    assert np.allclose(raw_penalty / pdetnorm, penalty)

    # this must break because the prior hasn't been updated
    try:
        penalty = prior.prior2penalty()
    except AssertionError:
        pass

    # generate the prior from penalty
    prior.get_prior(dodetnorm=False)
    raw_prior = np.linalg.inv(raw_penalty)
    assert np.allclose(raw_prior, prior.asarray)
    assert np.allclose(prior.penalty, raw_penalty)

    # generate a regularized prior from penalty
    penalty = prior.prior2penalty(dodetnorm=True)
    reg_penalty = np.linalg.inv(raw_prior)
    reg_pdetnorm = tikutils.determinant_normalizer(reg_penalty)
    assert np.allclose(prior.penalty, raw_penalty)
    assert np.allclose(reg_penalty / reg_pdetnorm, penalty)

    # subselect delays
    delays = np.asarray([1,2,3,4])
    prior = tp.PriorFromPenalty(raw_penalty, delays=delays)
    assert np.allclose(prior.delays, delays)

    # generate prior
    prior.get_prior(dodetnorm=False)
    raw_prior = np.linalg.inv(raw_penalty)
    raw_prior = tikutils.fast_indexing(raw_prior, delays, delays)
    # prior should only contain delays of interest
    assert np.allclose(raw_prior, prior.asarray)
    # penalty should be kept as original
    assert np.allclose(prior.penalty, raw_penalty)

    # check default wishart covariance
    assert np.allclose(prior.wishart, np.eye(raw_penalty.shape[0]))

    # regularize penalty before inverting
    oprior = prior.get_prior(hhparam=2.0, dodetnorm=False)
    raw_prior = np.linalg.inv(raw_penalty + 2.0*np.eye(raw_penalty.shape[0]))
    raw_prior = tikutils.fast_indexing(raw_prior, delays, delays)
    assert np.allclose(raw_prior, oprior)

    # regularize
    oprior = prior.get_prior(hhparam=2.0, dodetnorm=True)
    detnorm = tikutils.determinant_normalizer(raw_prior )
    assert np.allclose(raw_prior / detnorm, oprior)

    # set a non-diagonal wishart prior
    a = np.random.randn(10,10)
    W = np.dot(a, a.T)
    prior.set_wishart(W)
    assert np.allclose(prior.wishart, W)
    # check the update works
    oprior = prior.get_prior(hhparam=2.0, dodetnorm=True)
    raw_prior = np.linalg.inv(raw_penalty + 2.0*W)
    raw_prior = tikutils.fast_indexing(raw_prior, delays, delays)
    detnorm = tikutils.determinant_normalizer(raw_prior )
    assert np.allclose(raw_prior / detnorm, oprior)


def test_smoothness_prior():
    delays = range(10)
    ndelays = len(delays)

    for order in [1,2,4]:
        C = tikutils.difference_operator(order, ndelays)
        raw_penalty = np.dot(C, C.T)

        # create object
        prior = tp.SmoothnessPrior(delays=delays, order=order)
        assert np.allclose(prior.asarray.shape, (ndelays,ndelays))
        assert np.allclose(prior.penalty, raw_penalty)

        # create prior
        prior.get_prior(dodetnorm=False)
        raw_prior = np.linalg.inv(raw_penalty)
        assert np.allclose(raw_prior, prior.asarray)

        # update regularizer
        oprior = prior.get_prior(hhparam=2.0, dodetnorm=False)
        raw_prior = np.linalg.inv(raw_penalty + 2.0*np.eye(ndelays))
        assert np.allclose(raw_prior, oprior)

    # check delay sub-selection
    delays = np.asarray([1,3,6,8])
    ndelays = len(delays)
    fulldelays = range(max(delays)+1)

    C = tikutils.difference_operator(order, len(fulldelays))
    raw_penalty = np.dot(C, C.T)

    # create object
    prior = tp.SmoothnessPrior(delays=delays, order=order)
    assert np.allclose(prior.get_prior().shape, (ndelays,ndelays))
    assert np.allclose(prior.penalty.shape, len(fulldelays), len(fulldelays))
    assert np.allclose(prior.penalty, raw_penalty)

    # create prior
    prior.get_prior(dodetnorm=False)
    raw_prior = np.linalg.inv(raw_penalty)
    raw_prior = tikutils.fast_indexing(raw_prior, delays, delays)
    assert np.allclose(raw_prior, prior.asarray)

    # update regularizer
    oprior = prior.get_prior(hhparam=2.0, dodetnorm=False)
    raw_prior = np.linalg.inv(raw_penalty + 2.0*np.eye(len(fulldelays)))
    raw_prior = tikutils.fast_indexing(raw_prior, delays, delays)
    assert np.allclose(raw_prior, oprior)


def test_gaussian_kernel_prior():
    # TODO
    prior = tp.GaussianKernelPrior(delays=range(10))
