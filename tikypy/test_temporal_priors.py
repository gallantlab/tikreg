import numpy as np

import temporal_priors as tp
import utils as tikutils

##############################
#
##############################



def test_base_prior():
    tmp = np.random.randn(10, 10)
    raw_prior = np.dot(tmp, tmp.T)
    prior = tp.BasePrior(raw_prior, dodetnorm=False)
    assert prior.detnorm == 1.0
    assert np.allclose(prior.asarray, raw_prior)
    penalty = np.linalg.inv(prior.asarray + np.eye(prior.asarray.shape[0]))
    assert np.allclose(prior.prior2penalty(regularizer=1.0), penalty)


    tmp = np.random.randn(10, 10)
    raw_prior = np.dot(tmp, tmp.T)
    prior = tp.BasePrior(raw_prior, dodetnorm=True)
    detnorm = tikutils.determinant_normalizer(raw_prior)
    assert prior.detnorm == detnorm

    rr = raw_prior / tikutils.determinant_normalizer(raw_prior)
    assert np.allclose(prior.asarray, rr)
    penalty = np.linalg.inv(prior.asarray + np.eye(prior.asarray.shape[0]))
    penalty /= tikutils.determinant_normalizer(penalty)
    assert np.allclose(prior.prior2penalty(regularizer=1.0), penalty)


def test_temporal_prior():
    ndelays = 10
    tmp = np.random.randn(ndelays, ndelays)
    raw_prior = np.dot(tmp, tmp.T)
    prior = tp.TemporalPrior(raw_prior)
    assert hasattr(prior, 'ndelays')
    assert hasattr(prior, 'delays')


def test_spherical_prior():
    prior = tp.SphericalPrior()
    assert len(prior.delays) == prior.ndelays
    assert isinstance(prior.asarray, np.ndarray)
    assert prior.asarray.shape[0] == prior.ndelays

    delays = range(10)
    ndelays = len(delays)
    prior = tp.SphericalPrior(delays)
    a,b = prior.asarray.shape
    assert a == b == len(prior.delays)
    assert np.allclose(prior.asarray, np.eye(ndelays))

    delays = [1,2,3,8,10]
    prior = tp.SphericalPrior(delays)
    assert np.allclose(prior.asarray, np.eye(len(delays)))


def test_hrf_prior():

    from aone.fmri import handler, utils as fmriutils
    H = fmriutils.hrf_default_basis(dt=2.0, duration=20.)
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
    H = fmriutils.hrf_default_basis(dt=1.0, duration=20.)
    raw_prior = np.dot(H, H.T).astype(np.float64)
    prior = tp.HRFPrior(dt=1.0)
    assert np.allclose(prior.asarray, raw_prior)

    # change total duration
    H = fmriutils.hrf_default_basis(dt=2.0, duration=40.)
    raw_prior = np.dot(H, H.T).astype(np.float64)
    prior = tp.HRFPrior(duration=40)
    assert np.allclose(prior.asarray, raw_prior)

    # change resolution and duratoin
    H = fmriutils.hrf_default_basis(dt=1.0, duration=40.)
    raw_prior = np.dot(H, H.T).astype(np.float64)
    prior = tp.HRFPrior(dt=1.0, duration=40)
    assert np.allclose(prior.asarray, raw_prior)

    # change resolution and duration and subselect delays
    H = fmriutils.hrf_default_basis(dt=1.0, duration=40.)
    raw_prior = np.dot(H, H.T).astype(np.float64)
    delays = np.arange(1, 40)
    tt = tikutils.fast_indexing(raw_prior, delays, delays)
    prior = tp.HRFPrior(dt=1.0, duration=40, delays=delays)
    assert np.allclose(prior.asarray, tt)

    # change resolution and duration and subselect delays
    H = fmriutils.hrf_default_basis(dt=1.0, duration=40.)
    raw_prior = np.dot(H, H.T).astype(np.float64)
    delays = np.asarray([1,2,10,30,35])
    tt = tikutils.fast_indexing(raw_prior, delays, delays)
    prior = tp.HRFPrior(dt=1.0, duration=40, delays=delays)
    assert np.allclose(prior.asarray, tt)


    # grab continuous delays
    H = fmriutils.hrf_default_basis(dt=2.0, duration=20.)
    raw_prior = np.dot(H, H.T).astype(np.float64)

    delays = np.arange(1,10)
    tt = tikutils.fast_indexing(raw_prior, delays, delays)
    prior = tp.HRFPrior(delays)
    assert np.allclose(prior.asarray, tt)
    assert np.allclose(prior.delays, delays)
    assert np.allclose(prior.ndelays, len(delays))

    # grab disjoint delays
    H = fmriutils.hrf_default_basis(dt=2.0, duration=20.)
    raw_prior = np.dot(H, H.T).astype(np.float64)

    delays = np.asarray([1,3,6,9])
    tt = tikutils.fast_indexing(raw_prior, delays, delays)
    prior = tp.HRFPrior(delays)
    assert np.allclose(prior.asarray, tt)
    assert np.allclose(prior.delays, delays)
    assert np.allclose(prior.ndelays, len(delays))
