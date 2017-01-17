import numpy as np

import tikypy.utils as tikutils

def test_fast_indexing():
    D = np.random.randn(1000, 1000)
    rows = np.random.randint(0, 1000, (400))
    cols = np.random.randint(0, 1000, (400))

    a = tikutils.fast_indexing(D, rows, cols)
    b = D[rows, :][:, cols]
    assert np.allclose(a, b)
    a = tikutils.fast_indexing(D, rows)
    b = D[rows, :]
    assert np.allclose(a, b)
    a = tikutils.fast_indexing(D.T, cols).T
    b = D[:, cols]
    assert np.allclose(a, b)


def test_generators(N=100, testpct=0.2, nchunks=5, nfolds=5):
    ntest = int(N*(1./nfolds))
    ntrain = N - ntest
    alltrn = []
    folds = tikutils.generate_trntest_folds(N, 'cv', testpct=testpct,
                                            nfolds=nfolds, nchunks=nchunks)
    for idx, (trn, val) in enumerate(folds):
        # none of the trn is in the val
        assert np.in1d(trn, val).sum() == 0
        assert np.in1d(val, trn).sum() == 0
        assert len(np.unique(np.r_[val, trn])) == N
        assert ntrain + nchunks >= len(trn) >= ntrain - nchunks

    ntest = int(N*testpct)
    ntrain = int(np.ceil(N - ntest))
    remainder = np.mod(ntrain, nchunks)
    nfolds = 10
    folds = tikutils.generate_trntest_folds(N, 'nbb', nfolds=nfolds,
                                            testpct=testpct, nchunks=nchunks)
    for idx, (trn, val) in enumerate(folds):
        # none of the trn is in the val
        assert np.in1d(trn, val).sum() == 0
        assert np.in1d(val, trn).sum() == 0
        assert (len(trn) == ntrain - remainder) or (len(trn) == ntrain - nchunks)
    assert idx+1 == nfolds

    nfolds = 100
    folds = tikutils.generate_trntest_folds(N, 'mbb', nfolds=nfolds,
                                            testpct=testpct, nchunks=nchunks)
    for idx, (trn, val) in enumerate(folds):
        # none of the trn is in the val
        assert np.in1d(trn, val).sum() == 0
        assert np.in1d(val, trn).sum() == 0
        assert len(trn) == (ntrain - remainder) or (len(trn) == ntrain - nchunks)
    assert idx+1 == nfolds


def test_noise_ceiling_correction():
    # Based on Schoppe, et al. (2016)
    # Author's Sample MATLAB code
    # https://github.com/OSchoppe/CCnorm/blob/master/calc_CCnorm.m
    from scipy.stats import zscore
    signal = np.random.randn(50)
    repeats = np.asarray([signal + np.random.randn(len(signal))*1. for t in range(10)])
    nreps, ntpts = repeats.shape

    repeats = zscore(repeats, 1)
    ymean = np.mean(repeats,0) # mean time-course
    yhat = zscore(ymean + np.random.randn(len(ymean))*0.5)

    Vy = np.var(ymean)
    Vyhat = 1.
    Cyyhat = np.cov(ymean, yhat)[0,1]
    mcov = ((ymean - ymean.mean(0))*yhat).sum(0)/(ntpts - 1) # sample covariance
    assert np.allclose(Cyyhat, mcov)

    top = np.var(np.sum(repeats,0)) - np.sum(np.var(repeats, 1))
    SP = top/(nreps*(nreps-1)) # THIS IS EXPLAINABLE VARIANCE
    # same as
    top2 = (nreps**2)*np.var(np.mean(repeats,0)) - nreps
    SP2 = top2/(nreps*(nreps -1))
    assert np.allclose(top, top2)
    assert np.allclose(SP, SP2)
    # same as
    top3 = nreps*np.var(np.mean(repeats,0)) - 1
    SP3 = top3/(nreps-1)
    assert np.allclose(top2/nreps, top3)
    assert np.allclose(SP2, SP3)
    # same as
    ev = np.var(np.mean(repeats,0)) # same as R2 := SSreg/SStot
    ev = ev - ((1 - ev) / np.float((repeats.shape[0] - 1))) # adjusted
    assert np.allclose(ev, SP)
    # same as (1 - residual variance)
    assert np.allclose(tikutils.explainable_variance(repeats[...,None],
                                                     dozscore=False, ncorrection=True),
                       SP)
    assert np.allclose(tikutils.explainable_variance(repeats[...,None],
                                                     dozscore=True, ncorrection=True),
                       SP)
    # measures
    CCabs	= Cyyhat/np.sqrt(Vy*Vyhat)
    CCnorm	= Cyyhat/np.sqrt(SP*Vyhat)
    CCmax	= np.sqrt(SP/Vy)
    corrected = CCabs/CCmax
    eqn27 = Cyyhat/np.sqrt(SP) # eqn 27 from Schoppe, et al. (2016) paper

    res = tikutils.noise_ceiling_correction(repeats, yhat, dozscore=True)
    assert np.allclose(eqn27, res)
    assert np.allclose(corrected, res)
    assert np.allclose(CCnorm, res)
