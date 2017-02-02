from collections import defaultdict as ddict
import itertools

import numpy as np
from scipy import linalg as LA
from scipy.linalg import cho_factor, cho_solve


from tikypy.utils import SVD
from tikypy.kernels import lazy_kernel
import tikypy.utils as tikutils

METHOD = 'SVD'

def nan_to_num(*args, **kwargs):
    return np.nan_to_num(*args, **kwargs)


def zscore(*args, **kwargs):
    from scipy.stats import zscore
    return zscore(*args, **kwargs)


def atleast_2d(arr):
    if arr.ndim < 2:
        arr = np.atleast_2d(arr).T
    return arr


def _ols(X, Y, rcond=1e-08):
    '''Perform OLS fit, return weight estimates
    '''
    return np.dot(LA.pinv(X, rcond=rcond), Y)


def ols(X, Y, rcond=1e-08):
    '''Perform OLS fit, return weight estimates
    '''
    U, S, Vt = LA.svd(X, full_matrices=False)
    gdx = S > rcond
    S = S[gdx]
    U = U[:, gdx]
    V = Vt.T[:, gdx]
    XTY  = np.dot(X.T, Y)
    XTXinv = np.dot(tikutils.mult_diag(1.0/S**2, V, left=False), V.T)
    return np.dot(XTXinv, XTY)


def loo_ols(xtrain_samples, ytrain_samples, rcond=1e-08):
    '''Leave-one out OLS
    Return the mean weight across head-out folds

    Parameters
    ----------
    ytrain_samples : np.ndarray (nfolds, time, voxels)
    xtrain_samples : np.ndarray (nfolds, time, features)

    '''
    B = 0
    nreps = len(ytrain_samples)
    assert nreps == len(xtrain_samples)
    samples = np.arange(nreps)
    for left_out in xrange(nreps):
        train = samples != left_out
        X = np.vstack(xtrain_samples[train])
        Y = np.vstack(ytrain_samples[train])
        B += ols(X, Y)/nreps
    return B


def olspred(X, Y, Xtest=False):
    '''Fit OLS, return predictions ``Yhat``
    '''
    U, S, Vt = SVD(X)
    V = Vt.T
    del Vt
    UTY = np.dot(U.T, Y)
    if (Xtest is False) or (Xtest is None):
        LH = U
    else:
        LH = np.dot(Xtest, tikutils.mult_diag(1.0/S, V, left=False))
    return np.dot(LH, UTY)


def check_response_dimensionality(train, test, allow_test_none=True):
    '''Make sure matrices are 2D arrays before running models
    '''
    if train.ndim == 1:
        train = train[...,None]
    if test is not None:
        if test.ndim == 1:
            test = test[...,None]
    else:
        if not allow_test_none:
            test = train
    return train, test


def should_solve_dual(X, kernel):
    '''Answer whether we should solve the regression
    problem in the dual (kernel) space.
    '''
    n, p = X.shape
    solve_dual = False
    if p > n*0.75:
        solve_dual = True
    if (kernel is not None) and (kernel is not 'linear'):
        solve_dual = True
    return solve_dual


def clean_results_dict(results):
    '''Make sure we return arrays, and ndim is at least 2D
    '''
    for k,v in results.items():
        v = np.asarray(v)
        v = v.squeeze() if k != 'performance' else v
        if v.ndim <= 1: v = v[...,None]
        # Update
        results[k] = v
    return results





def solve_l2_primal(Xtrain, Ytrain,
                    Xtest=None, Ytest=None,
                    ridges=[0], method=METHOD,
                    zscore_ytrain=False, zscore_ytest=False,
                    EPS=1e-10, verbose=False,
                    performance=False, predictions=False, weights=False):
    '''Solve the (primal) L2 regression problem for each L2 parameter.
    '''
    results = ddict(list)
    Ytrain = atleast_2d(Ytrain)

    if predictions:
        assert Xtest is not None
    if performance:
        assert (Ytest is not None) and (Xtest is not None)
        Ytest = atleast_2d(Ytest)

    if zscore_ytrain:
        Ytrain = zscore(Ytrain)
    if zscore_ytest:
        Ytest = zscore(Ytest)

    if method == 'SVD':
        U, S, Vt = SVD(Xtrain, full_matrices=False)
        V = Vt.T
        del Vt
        gidx = S > EPS
        S = S[gidx]
        U = U[:, gidx]
        V = V[:, gidx]
        UTY = np.dot(U.T, Ytrain)
        del(U)

        if predictions or performance:
            XtestV = np.dot(Xtest, V)

    elif method == 'Chol':
        XtY = np.dot(Xtrain.T, Ytrain)
        XtX = np.dot(Xtrain.T, Xtrain)


    for lidx, rlambda in enumerate(ridges):
        if method == 'SVD':
            D = S / (S**2 + rlambda**2)
        elif method == 'Chol':
            XtXI = XtX + rlambda**2 * np.eye(Xtrain.shape[-1])
            L, lower = cho_factor(XtXI, lower=True)
            del XtXI

        if performance:
            # Compute performance
            if method == 'SVD':
                XVD = tikutils.mult_diag(D, XtestV, left=False)
                Ypred = np.dot(XVD, UTY)
            elif method == 'Chol':
                Ypred = np.dot(Xtest, cho_solve((L, lower), XtY))
            cc = tikutils.columnwise_correlation(Ypred, Ytest, axis=0)
            results['performance'].append(cc)

            if verbose:
                perf = nan_to_num(cc)
                contents = (lidx +1, rlambda, np.mean(perf),
                            np.percentile(perf, 25), np.median(perf), np.percentile(perf, 75),
                            np.sum(perf < 0.25), np.sum(perf > 0.75))
                txt = "lambda %02i: %8.03f, mean=%0.04f, (25,50,75)pctl=(%0.04f,%0.04f,%0.04f),"
                txt += "(0.2<r>0.8): (%03i,%03i)"
                print(txt % contents)

        if predictions and performance:
            results['predictions'].append(Ypred)
        elif predictions:
            # only predictions
            if method == 'SVD':
                XVD = tikutils.mult_diag(D, XtestV, left=False)
                Ypred = np.dot(XVD, UTY)
            elif method == 'Chol':
                Ypred = np.dot(Xtest, cho_solve((L, lower), XtY))
            results['predictions'].append(Ypred)
        if weights:
            # weights
            if method == 'SVD':
                betas = np.dot(tikutils.mult_diag(D, V, left=False), UTY)
            elif method == 'Chol':
                betas = cho_solve((L, lower), XtY)
            results['weights'].append(betas)

    return clean_results_dict(dict(results))


def solve_l2(Xtrain, Ytrain,
             Xtest=None, Ytest=None,
             ridge=0.0, verbose=False,
             kernel_name='linear', kernel_param=None,
             kernel_weights=False,
             **kwargs):
    '''Solve L2 regularized regression problem
    '''
    n, p = Xtrain.shape
    if kernel_name is None: kernel_name = 'linear'

    # figure out how to solve the problem
    solve_dual = should_solve_dual(Xtrain, kernel_name) # Should's I?
    if solve_dual:
        ktrain_object = lazy_kernel(Xtrain, kernel_type=kernel_name)
        ktrain_object.update(kernel_param, verbose=verbose)

        if (Xtest is None) and (Ytest is None):
            # with-in set fit
            ktest_object = ktrain_object
            Ytest = Ytrain
        else:
            # project test data to kernel
            ktest_object = lazy_kernel(Xtest, Xtrain, kernel_type=kernel_name)
            ktest_object.update(kernel_param, verbose=verbose)

        fit = solve_l2_dual(ktrain_object.kernel, Ytrain,
                            ktest_object.kernel, Ytest,
                            ridges=[ridge],
                            **kwargs)

        if (kernel_name == 'linear') and ('weights' in fit) and (kernel_weights is False):
            fit['weights'] = np.dot(Xtrain.T, fit['weights'])
    else:
        if (Xtest is None) and (Ytest is None):
            Xtest, Ytest = Xtrain,Ytrain

        fit = solve_l2_primal(Xtrain, Ytrain,
                              Xtest, Ytest,
                              ridges=[ridge],
                              **kwargs)


    return clean_results_dict(dict(fit))


def solve_l2_dual(Ktrain, Ytrain,
                  Ktest=None, Ytest=None,
                  ridges=[0.0], method=METHOD, EPS=1e-10, verbose=False,
                  performance=False, predictions=False, weights=False):
    '''Solve the dual (kernel) L2 regression problem for each L2 parameter.
    '''
    results = ddict(list)

    if predictions:
        assert Ktest is not None
    if performance:
        assert (Ytest is not None) and (Ktest is not None)
        Ytest = atleast_2d(Ytest)


    if method == 'SVD':
        L, Q = LA.eigh(Ktrain)
        gidx = L > EPS
        L = L[gidx]
        Q = Q[:, gidx]
        QTY = np.dot(Q.T, Ytrain)

        if predictions or performance:
            KtestQ = np.dot(Ktest, Q)

    for rdx, rlambda in enumerate(ridges):
        if method == 'SVD':
            D = 1.0 / (L + rlambda**2)
        elif method == 'Chol':
            KtKI = Ktrain + rlambda**2 * np.eye(Ktrain.shape[0])
            L, lower = cho_factor(KtKI, lower=True, check_finite=False)
            del KtKI

        if performance:
            if method == 'SVD':
                KtestQD = tikutils.mult_diag(D, KtestQ, left=False)
                Ypred = np.dot(KtestQD, QTY)
            elif method == 'Chol':
                Ypred = np.dot(Ktest, cho_solve((L, lower), Ytrain))

            cc = tikutils.columnwise_correlation(Ypred, Ytest)
            results['performance'].append(cc)

            if verbose:
                perf = nan_to_num(cc)
                contents = (rdx +1, rlambda, np.mean(perf),
                            np.percentile(perf, 25), np.median(perf), np.percentile(perf, 75),
                            np.sum(perf < 0.25), np.sum(perf > 0.75))
                txt = "lambda %02i: %8.03f, mean=%0.04f, (25,50,75)pctl=(%0.04f,%0.04f,%0.04f),"
                txt += "(0.2<r>0.8): (%03i,%03i)"
                print(txt % contents)

        if predictions and performance:
            results['predictions'].append(Ypred)
        elif predictions:
            if method == 'SVD':
                KtestQD = tikutils.mult_diag(D, KtestQ, left=False)
                Ypred = np.dot(KtestQD, QTY)
            elif method == 'Chol':
                Ypred = np.dot(Ktest, cho_solve((L, lower), Ytrain))
            results['predictions'].append(Ypred)

        if weights:
            if method == 'SVD':
                QD = tikutils.mult_diag(D, Q, left=False)
                kernel_weights = np.dot(QD, QTY)
            elif method == 'Chol':
                kernel_weights = cho_solve((L, lower), Ytrain)
            results['weights'].append(kernel_weights)

    return clean_results_dict(dict(results))


def kernel_spatiotemporal_prior(Xtrain, temporal_prior, spatial_prior,
                                Xtest=None, delays=[1,2,3,4]):
    '''Compute the kernel matrix of a model with a spatio-temporal prior

    temporal_prior (d, d): d = len(delays)
    '''
    matrix_mult = np.dot
    # if tikutils.isdiag(spatial_prior):
    #     # TODO: this is slower for some reason.... must debug
    #     def matrix_mult(xx,yy):
    #         di = np.diag(yy)
    #         if np.allclose(di, di[0]):
    #             # constant diagonal
    #             res = xx*di[0]
    #         else:
    #             res = tikutils.mult_diag(di, xx, left=False)
    #         return res


    if Xtest is None:
        Xtest = Xtrain
    kernel = np.zeros((Xtest.shape[0], Xtrain.shape[0]))
    for jdx, jdelay in enumerate(delays):
        Xj = Xtrain[tikutils.delay2slice(jdelay)]
        for idx, idelay in enumerate(delays):
            if temporal_prior[idx,jdx] == 0:
                continue
            Xi = Xtest[tikutils.delay2slice(idelay)]
            tmp = np.dot(temporal_prior[idx,jdx]*matrix_mult(Xi, spatial_prior), Xj.T)
            kernel[idelay:,jdelay:] += tmp
    return kernel


def kernel_cvridge(Ktrain, Ytrain,
                   Ktest=None, Ytest=None,
                   ridges=[0.0],
                   folds='cv', nfolds=5, blocklen=5, trainpct=0.8,
                   performance=False, predictions=False, weights=False,
                   verbose=True, EPS=1e-10,
                   ):
    import time
    start_time = time.time()

    n = Ktrain.shape[0]
    if not isinstance(folds, list):
        folds = tikutils.generate_trnval_folds(n, sampler=folds,
                                            nfolds=nfolds,
                                            testpct=1-trainpct,
                                            nchunks=blocklen)
    else:
        nfolds = len(folds)
    nridges = len(ridges)

    if verbose:
        txt = (nridges, nfolds)
        intro = 'Fitting *%i* ridges, across *%i* folds'%txt
        print(intro)

    results = np.empty((nfolds, nridges, Ytrain.shape[-1]))
    for fdx, fold in enumerate(folds):
        trn, test = fold
        ntrn, ntest = len(trn), len(test)
        if verbose:
            txt = (fdx+1,nfolds,ntrn,ntest)
            print('train ridge fold  %i/%i: ntrain=%i, ntest=%i'%txt)

        Ktrn = tikutils.fast_indexing(Ktrain, trn, trn)
        Ktest = tikutils.fast_indexing(Ktrain, test, trn)

        res = solve_l2_dual(Ktrn, Ytrain[trn],
                            Ktest, zscore(Ytrain[test]),
                            ridges, EPS=EPS,
                            weights=False,
                            predictions=False,
                            performance=True,
                            verbose=verbose)

        results[fdx,:,:] = res['performance']

    # We done, otherwise fit and predict the held-out set
    if (predictions is False) and (performance is False) and (weights is False):
        if verbose: print('Duration %0.04f[mins]' % ((time.time()-start_time)/60.))
        return {'cvresults' : results}


    # Find best parameters across responses
    surface = np.nan_to_num(results.mean(0)).mean(-1).squeeze()
    # find the best point in the 2D space
    max_point = np.where(surface.max() == surface)
    # make sure it's unique (conservative-ish biggest ridge/parameter)
    max_point = map(max, max_point)
    ridgeopt = ridges[max_point]

    if verbose:
        desc = 'held-out' if (Ktest is not None) else 'within'
        outro = 'Predicting {d} set:\ncvperf={cc},ridge={alph}'
        outro = outro.format(d=desc,cc=surface.max(),alph=ridgeopt)
        print(outro)

    fit = solve_l2_dual(Ktrain, Ytrain,
                        Ktest, Ytest,
                        ridges=[ridgeopt],
                        performance=performance,
                        predictions=predictions,
                        weights=weights,
                        EPS=EPS,
                        verbose=verbose,
                        )

    if verbose: print('Duration %0.04f[mins]' % ((time.time()-start_time)/60.))
    fit = clean_results_dict(dict(fit))
    fit['cvresults'] = results
    return fit


def cvridge(Xtrain, Ytrain,
            Xtest=None, Ytest=None,
            ridges=[0.0],
            Li=None,
            kernel_name='linear', kernel_params=None,
            folds='cv', nfolds=5, blocklen=5, trainpct=0.8,
            verbose=True, EPS=1e-10,
            withinset_test=False,
            performance=False, predictions=False, weights=False,
            kernel_weights=False):
    """Cross-validation procedure for tikhonov regularized regression.

    Parameters
    ----------
    Xtrain (n, p):
        Design matrix
    Ytrain (n, v):
        Training set responses
    Xtest (None, (m, p)):
        Design matrix for held-out set
    Ytest (None, (m, v)):
        Held-out set responses
    ridges (r,):
        Ridge parameters to evaluate
    Li (q,p):
        Generalized tikhonov regression. This solves the problem with a
        prior on $\beta$ determined by $L^\top L$. The problem is solved
        in the standard form and in kernel space if necessary.
    kernel_name (str):
        Kernel to use
    kernel_params (None, (k,)):
        Kernel parameters to cross-validate
    folds (str, list):
        * (str) Type of cross-validation
           - 'cv'  - cross-validation with chunks of size ``blocklen``
           - 'nbb' - block boostrap with chunks of size ``blocklen``
           - 'mbb' - moving/overlapping block bootstrap chunks of size ``blocklen``
        * (list) Can also be a list of (train, test) pairs: [(trn1, test1),...]
    nfolds (int):
        Number of learning folds
    blocklen (int):
        Chunk data into blocks of this size, and sample these blocks
    trainpct (float 0-1):
        Percentage of data to use in training if using a bootstrap sampler.
    withinset_test (bool):
        If no ``Xtest`` or ``Ytest`` is given and ``predictions`` and/or
        ``performance`` are requested, compute these testues based on training set.
    performance (bool):
        Held-out prediction performance
    predictions (bool):
        Held-out timecourse predictions
    weights (bool):
        Weight estimates on training set (does not depend on (``Xtest``,``Ytest``)
    kernel_weights (bool):
        Whether to project kernel weights into feature weights.
        If True, the kernel weights are returned. This is useful when fitting
        large models and storing the feature weights is expensive.
    verbose (bool):
        Verbosity
    EPS (float):
        Testue used to threshold small eigentestues

    Returns
    -------
    fit (optional; dict):
        cross-validation results per response for each fold, kernel and L2 parameters
            * cvresults (``nfolds``, len(``kernel_params``), len(``ridges``), nresponses)
        If a held-out set (``Xtest`` and ``Ytest``) is specified, performs the
        fit on the full training set with the optimal L2 and kernel parameters.
        It returns, as requested, any of:
           * predictions (m, v) ``Ytest`` prediction for each voxel
           * performance (v,) correlation coefficient of predictions and ``Ytest``
           * weights: (p, v) for linear models, (n by v) for non-linear models

    """
    import time
    start_time = time.time()

    if kernel_name is None: raise TestueError('Say linear if linear')
    kernel_params = [None] if (kernel_name == 'linear') else kernel_params
    nkparams = len(kernel_params)


    Ytrain, Ytest = check_response_dimensionality(Ytrain, Ytest, allow_test_none=True)
    Xtrain, Xtest = check_response_dimensionality(Xtrain, Xtest, allow_test_none=True)

    # Check for generalized tikhonov
    if Li is not None:
        Xtrain = np.dot(Xtrain, Li)

    n, p = Xtrain.shape

    if not isinstance(folds, list):
        folds = tikutils.generate_trnval_folds(n, sampler=folds,
                                            nfolds=nfolds,
                                            testpct=1-trainpct,
                                            nchunks=blocklen)
        if kernel_name != 'linear':
            # if kernel is not linear we need to get a list to re-use folds
            folds = [(trn,test) for trn,test in folds]

    else:
        nfolds = len(folds)

    nridges = len(ridges)

    if verbose:
        txt = (nridges, nfolds, nkparams, kernel_name)
        intro = 'Fitting *%i* ridges, across *%i* folds, and *%i* "%s" kernel parameters'%txt
        print(intro)

    # figure out how to solve the problem
    solve_dual = should_solve_dual(Xtrain, kernel_name)
    if solve_dual:
        if verbose: print('Caching *%s* kernel'%kernel_name)
        ktrain_object = lazy_kernel(Xtrain, kernel_type=kernel_name)

    results = np.empty((nfolds, nkparams, nridges, Ytrain.shape[-1]))
    for kdx, kernel_param in enumerate(kernel_params):
        if solve_dual:
            ktrain_object.update(kernel_param)
            kernel = ktrain_object.kernel
            if verbose:
                txt = (kernel_name,kdx+1,nkparams,str(kernel_param))
                print('Updating *%s* kernel %i/%i:%s'%txt)

        for fdx, fold in enumerate(folds):
            trn, test = fold
            ntrn, ntest = len(trn), len(test)
            if verbose:
                txt = (fdx+1,nfolds,ntrn,ntest)
                print('train ridge fold  %i/%i: ntrain=%i, ntest=%i'%txt)

            if solve_dual is False:
                res = solve_l2_primal(Xtrain[trn], Ytrain[trn],
                                      Xtrain[test], zscore(Ytrain[test]),
                                      ridges, EPS=EPS,
                                      weights=False,
                                      predictions=False,
                                      performance=True,
                                      verbose=verbose)
            else:
                Ktrain = tikutils.fast_indexing(kernel,trn, trn)
                Ktest = tikutils.fast_indexing(kernel,test, trn)
                res = solve_l2_dual(Ktrain, Ytrain[trn],
                                    Ktest, zscore(Ytrain[test]),
                                    ridges, EPS=EPS,
                                    weights=False,
                                    predictions=False,
                                    performance=True,
                                    verbose=verbose)
            # Store results
            results[fdx,kdx,:,:] = res['performance']
            del(res)

    # We done, otherwise fit and predict the held-out set
    if (predictions is False) and (performance is False) and (weights is False):
        if verbose: print('Duration %0.04f[mins]' % ((time.time()-start_time)/60.))
        return {'cvresults' : results}

    # Find best parameters across responses
    surface = np.nan_to_num(results.mean(0)).mean(-1)
    # find the best point in the 2D space
    max_point = np.where(surface.max() == surface)
    # make sure it's unique (conservative-ish biggest ridge/parameter)
    max_point = map(max, max_point)
    # The maximum point
    kernmax, ridgemax = max_point
    kernopt, ridgeopt = kernel_params[kernmax], ridges[ridgemax]
    if verbose:
        desc = 'held-out' if (Xtest is not None) else 'within'
        outro = 'Predicting {d} set:\ncvperf={cc},ridge={alph},kernel={kn},kernel_param={kp}'
        outro = outro.format(d=desc,cc=surface.max(),alph=ridgeopt,
                             kn=kernel_name,kp=kernopt)
        print(outro)

    if solve_dual:
        # Set the parameter to the optimal
        ktrain_object.update(kernopt, verbose=verbose)

        if Ytest is not None:
            Ytest = zscore(Ytest)

        if Xtest is not None:
            if Li is not None: Xtest = np.dot(Xtest, Li)
            # project test data to kernel
            ktest_object = lazy_kernel(Xtest, Xtrain, kernel_type=kernel_name)
            ktest_object.update(kernopt, verbose=verbose)
            ktest = ktest_object.kernel
        elif withinset_test:
            # predict within set if so desired
            ktest = ktrain_object.kernel
            Ytest = zscore(Ytrain)
        else:
            ktest = None

        fit = solve_l2_dual(ktrain_object.kernel, Ytrain,
                            ktest, Ytest,
                            ridges=[ridgeopt],
                            performance=performance,
                            predictions=predictions,
                            weights=weights,
                            EPS=EPS,
                            verbose=verbose,
                            )
        # Project to linear space if we can
        if (kernel_name == 'linear') and ('weights' in fit) and (not kernel_weights):
            fit['weights'] = np.dot(Xtrain.T, fit['weights'])
    else:
        if Ytest is not None:
            Ytest = zscore(Ytest)
        if Xtest is not None:
            if Li is not None: Xtest = np.dot(Xtest, Li)
        elif withinset_test:
            Xtest = Xtrain
            Ytest = zscore(Ytrain)

        fit = solve_l2_primal(Xtrain, Ytrain,
                              Xtest, Ytest,
                              ridges=[ridgeopt],
                              performance=performance,
                              predictions=predictions,
                              weights=weights,
                              verbose=verbose,
                              EPS=EPS,
                              )
    if (Li is not None) and ('weights' in fit):
        # project back
        fit['weights'] = np.dot(Li, fit['weights'])

    if verbose: print('Duration %0.04f[mins]' % ((time.time()-start_time)/60.))
    fit = clean_results_dict(dict(fit))
    fit['cvresults'] = results
    return fit


def simple_ridge_dual(X, Y, ridge=10.0):
    '''Return weights for linear kernel ridge regression'''
    K = np.dot(X, X.T)
    kinv = np.linalg.inv(K + ridge*np.eye(K.shape[0]))
    return np.dot(X.T, np.dot(kinv, Y))


def simple_ridge_primal(X, Y, ridge=10.0):
    '''Return weights for ridge regression'''
    XTY = np.dot(X.T, Y)
    XTXinv = np.linalg.inv(np.dot(X.T, X) + ridge*np.eye(X.shape[-1]))
    return np.dot(XTXinv, XTY)







def simple_generalized_tikhonov(X, Y, L, ridge=10.0):
    '''Direct implementation of generalized tikhonov regression
    '''
    XTXLTL = np.dot(X.T, X) + ridge*np.dot(L.T, L)
    XTY = np.dot(X.T, Y)
    betas = np.dot(LA.inv(XTXLTL), XTY)
    return betas


def generalized_tikhonov(X, Y, Li, ridge=10.0):
    '''Implementation fo tikhonov regression using the
    standard transform (cf. Hansen, 1998).
    '''
    A = np.dot(X, Li)
    ATY = np.dot(A.T, Y)
    ATAIi = LA.inv(np.dot(A.T, A) + ridge*np.identity(A.shape[-1]))
    weights = np.dot(ATAIi, ATY)
    betas = np.dot(Li, weights)
    return betas


def _generalized_tikhonov_dual(X, Y, Li, ridge=10.0):
    '''check kernel representation also works
    '''
    A = np.dot(X, Li)
    AATIi = LA.inv(np.dot(A, A.T) + ridge*np.identity(A.shape[0]))
    rlambdas = np.dot(AATIi, Y)
    weights = np.dot(A.T, rlambdas)
    betas = np.dot(Li, weights)
    return betas



def find_optimum_mvn(response_cvmean,
                     temporal_hhparams,
                     spatial_hyparams,
                     ridge_hyparams):
    '''
    '''
    optimum = np.unravel_index(np.argmax(response_cvmean),
                               response_cvmean.shape)
    temporal_argmax = optimum[0]
    temporal_optimum = temporal_hhparams[temporal_argmax]

    spatial_argmax = optimum[1]
    spatial_optimum = spatial_hyparams[spatial_argmax]

    ridge_argmax = optimum[2]
    ridge_optimum = ridge_hyparams[ridge_argmax]

    return temporal_optimum, spatial_optimum, ridge_optimum


def crossval_stem_wmvnp(features_train,
                        responses_train,
                        ridges=np.logspace(0,3,10),
                        normalize_kernel=True,
                        temporal_prior=None,
                        feature_priors=None,
                        performance=True,
                        mean_cv_only=False,
                        folds=(1,5),
                        method='SVD',
                        verbosity=1,
                        ):
    '''Cross-validation procedure for
    spatio-temporal encoding models with MVN priors.
    '''
    import time
    start_time = time.time()

    if isinstance(verbosity, bool):
        verbosity = 1 if verbosity else 0
    if isinstance(features_train, np.ndarray):
        features_train = [features_train]

    nridges = len(ridges)
    delays = temporal_prior.delays
    ndelays = len(delays)
    nfeatures = [fs.shape[1] for fs in features_train]
    nresponses = responses_train.shape[-1]
    ntrain = responses_train.shape[0]
    kernel_normalizer = 1.0

    #### handle cross-validation folds options
    if isinstance(folds, list):
        # pre-defined folds
        nfolds = len(folds)
    elif np.isscalar(folds):
        # do k-fold cross-validation only once
        nfolds = (1, folds)
    else:
        # do k-fold cross-validation N times
        assert isinstance(folds, tuple)

    if isinstance(folds, tuple):
        # get cv folds (train, val) indeces
        nfolds = folds
        folds = tikutils.generate_trnval_folds(ntrain,
                                               sampler='bcv',
                                               nfolds=nfolds)
        nfolds = np.prod(nfolds)

    folds = list(folds)

    # get temporal hyper-prior hyper-parameters from object
    all_temporal_hhparams = [temporal_prior.get_hhparams()]
    # get feature prior hyper parameters
    all_spatial_hyparams= [t.get_hyperparameters() for t in feature_priors]
    # all combinations
    all_hyperparams = list(itertools.product(*(all_temporal_hhparams + all_spatial_hyparams)))
    nall_cvparams = len(all_hyperparams)

    # count parametres
    ntemporal_hhparams = np.prod([len(t) for t in all_temporal_hhparams])
    nspatial_hyparams = np.prod([len(t) for t in all_spatial_hyparams])

    # store cross-validation performance
    results = np.zeros((nfolds,
                        ntemporal_hhparams,
                        nspatial_hyparams,
                        nridges,
                        1 if mean_cv_only else responses_train.shape[-1]),
                       )

    sp_hyparams = []
    scaled_ridges = np.atleast_1d(ridges).copy()

    # start iterating through spatio-temporal hyperparameters
    for hyperidx, spatiotemporal_hyperparams in enumerate(all_hyperparams):
        temporal_hhparam = spatiotemporal_hyperparams[0]
        spatial_hyparams = spatiotemporal_hyperparams[1:]

        # map hyperparameters to surface of sphere
        spatial_hyparams /= np.linalg.norm(spatial_hyparams)
        sp_hyparams.append(spatial_hyparams)

        # apply the hyperparameter to the hyper-prior on the temporal prior
        this_temporal_prior = temporal_prior.get_prior(hhparam=temporal_hhparam)

        # get spatial and temporal parameter indeces
        shyperidx = np.mod(hyperidx, nspatial_hyparams)
        thyperidx = int(hyperidx // nspatial_hyparams)

        if verbosity:
            hyperdesc = (hyperidx+1, nall_cvparams,
                         thyperidx+1, ntemporal_hhparams, temporal_hhparam,
                         shyperidx+1, nspatial_hyparams) + tuple(spatial_hyparams)
            hypertxt = "%i/%i: temporal %i/%i=%0.03f, "
            hypertxt += "features %i/%i=(%0.04f, "
            hypertxt += ', '.join(["%0.04f"]*(len(spatial_hyparams)-1)) + ')'
            print(hypertxt % hyperdesc)

        Ktrain = 0.0
        # iterate through feature  matrices, priors, and hyperparameters
        # to construct spatio-temporal kernel for full training set
        for fdx, (fs_train, fs_prior, fs_hyper) in enumerate(zip(features_train,
                                                                 feature_priors,
                                                                 spatial_hyparams)):
            # compute spatio-temporal kernel for this feature space given
            # spatial prior hyperparameters, and temporal prior hyper-prior hyperparameters
            kernel_train = kernel_spatiotemporal_prior(fs_train,
                                                       this_temporal_prior,
                                                       fs_prior.get_prior(fs_hyper),
                                                       delays=delays)
            # store this feature space spatio-temporal kernel
            Ktrain += kernel_train


        if (normalize_kernel is True) and (hyperidx == 0):
            # normalize all ridges by the determinant of the first kernel
            kernel_normalizer = tikutils.determinant_normalizer(Ktrain)
            if np.allclose(kernel_normalizer, 0):
                # invalid determinant, do not scale
                kernel_normalizer = 1.0
            scaled_ridges *= np.sqrt(kernel_normalizer)

        Ktrain /= kernel_normalizer

        # perform cross-validation procedure
        for ifold, (trnidx, validx) in enumerate(folds):
            # extract training and validation sets from full kernel
            ktrn = tikutils.fast_indexing(Ktrain, trnidx, trnidx)
            kval = tikutils.fast_indexing(Ktrain, validx, trnidx)

            if verbosity > 1:
                txt = (ifold+1,nfolds,len(trnidx),len(validx))
                print('train fold  %i/%i: ntrain=%i, ntest=%i'%txt)

            # solve the regression problem
            fit = solve_l2_dual(ktrn, responses_train[trnidx],
                                kval, responses_train[validx],
                                ridges=ridges,
                                performance=True,
                                verbose=verbosity > 1,
                                method=method,
                                )

            if mean_cv_only:
                # only keep mean population performance on the validation set
                cvfold = np.nan_to_num(fit['performance']).mean(-1)[...,None]
            else:
                # keep all individual responses' performance on validtion set
                cvfold = fit['performance']
            results[ifold, thyperidx, shyperidx] = cvfold

        if verbosity:
            # print performance for this spatio-temporal hyperparameter set
            perf = nan_to_num(results[:,thyperidx,shyperidx].mean(0))
            group_ridge = ridges[np.argmax(perf.mean(-1))]
            contents = (group_ridge, np.mean(perf),
                        np.percentile(perf, 25), np.median(perf), np.percentile(perf, 75),
                        np.sum(perf < 0.25), np.sum(perf > 0.75))
            txt = "pop.cv.best: %6.03f, mean=%0.04f, (25,50,75)pctl=(%0.04f,%0.04f,%0.04f),"
            txt += "(0.2<r>0.8): (%03i,%03i)\n"
            print(txt % contents)

    #### dimensions explored
    dtype = np.dtype([('nfolds', np.int),
                      ('ntemporal_hhparams', np.int),
                      ('nspatial_hyparams', np.int),
                      ('nridges', np.int),
                      ('nresponses', np.int),
                      ('nfspaces', np.int)])

    dims = np.recarray(shape=(1), dtype=dtype)
    dims[0] = np.asarray([nfolds,
                          ntemporal_hhparams,
                          nspatial_hyparams,
                          nridges,
                          results.shape[-1],
                          len(features_train)])

    # spatial hyperparameters. all the same across temporal
    sp_hyparams = np.asarray(sp_hyparams)[:nspatial_hyparams]

    if verbosity:
        print('Duration %0.04f[mins]' % ((time.time()-start_time)/60.))

    return {'cvresults' : results,
            'dims' : dims,
            'spatial' : sp_hyparams,
            'ridges' : scaled_ridges,
            'temporal' : temporal_prior.get_hhparams(),
            }


def estimate_stem_wmvnp(features_train,
                        responses_train,
                        features_test=None,
                        responses_test=None,
                        ridges=np.logspace(0,3,10),
                        normalize_kernel=True,
                        temporal_prior=None,
                        feature_priors=None,
                        weights=False,
                        predictions=False,
                        performance=True,
                        # noise_ceiling_correction=False,
                        mean_cv_only=False,
                        folds=(1,5),
                        method='SVD',
                        verbosity=1,
                        cvresults=None,
                        population_optimal=False
                        ):
    '''
    '''
    delays = temporal_prior.delays
    ndelays = len(delays)

    if features_test is None:
        features_test = [features_test]*len(features_train)

    if cvresults is None:
        # find optimal hyperparamters via Nx k-fold cross-validation
        cvresults = crossval_stem_wmvnp(features_train,
                                        responses_train,
                                        ridges=ridges,
                                        normalize_kernel=normalize_kernel,
                                        temporal_prior=temporal_prior,
                                        feature_priors=feature_priors,
                                        mean_cv_only=mean_cv_only,
                                        folds=folds,
                                        method=method,
                                        verbosity=verbosity,
                                        )

    if (weights is False) and (performance is False) and (prediction is False):
        return cvresults

    dims = cvresults['dims']
    # find optima across cross-validation folds
    cvmean = cvresults['cvresults'].mean(0)

    if population_optimal is True and (dims.nresponses > 1):
        cvmean = np.nan_to_num(cvmean).mean(-1)[...,None]

    nresponses = int(dims.nresponses)
    nfspaces = int(dims.nfspaces)
    ntspaces = 1
    optima = np.zeros((nresponses, nfspaces + ntspaces + 1))
    for idx in range(nresponses):
        # find response optima
        temporal_opt, spatial_opt, ridge_opt = find_optimum_mvn(cvmean[...,idx],
                                                                cvresults['temporal'],
                                                                cvresults['spatial'],
                                                                cvresults['ridges'])
        optima[idx] = tuple([temporal_opt])+tuple(spatial_opt)+tuple([ridge_opt])

    cvresults['optima'] = optima                                 # store optima
    unique_optima = np.vstack(set(tuple(row) for row in optima)) # get unique rows

    # estimate solutions
    solutions = [[]]*nresponses
    for idx in range(unique_optima.shape[0]):
        # get hyper parameters
        uopt = unique_optima[idx][0], unique_optima[idx][1:-1], unique_optima[idx][-1]
        temporal_opt, spatial_opt, ridge_opt = uopt

        # fit responses that have this optimum
        responses_mask = np.asarray([np.allclose(row, unique_optima[idx]) for row in optima])
        test_responses = None if responses_test is None else responses_test[:, responses_mask]
        response_solution = estimate_simple_stem_wmvnp(features_train,
                                                       responses_train[:, responses_mask],
                                                       features_test=features_test,
                                                       responses_test=test_responses,
                                                       temporal_prior=temporal_prior,
                                                       temporal_hhparam=temporal_opt,
                                                       feature_priors=feature_priors,
                                                       feature_hyparams=spatial_opt,
                                                       weights=weights,
                                                       performance=weights,
                                                       predictions=predictions,
                                                       ridge_scale=ridge_opt,
                                                       verbosity=verbosity,
                                                       method=method,
                                                       )
        # store the solutions
        for rdx, response_index in enumerate(responses_mask.nonzero()[0]):
            solutions[response_index] = {k:v[...,rdx] for k,v in response_solution.items()}

        if verbosity:
            itxt = '%i responses:'%(responses_mask.sum())
            ttxt = "ridge=%9.03f, temporal=%0.03f," % (ridge_opt, temporal_opt)
            stxt = "spatial=("
            stxt += ', '.join(["%0.03f"]*(len(spatial_opt)))
            stxt = stxt%tuple(spatial_opt) + ')'
            perf = 'perf=%0.04f'%response_solution['performance'].mean()
            print(' '.join([itxt, ttxt, stxt, perf]))

    fits = ddict(list)
    for solution in solutions:
        for k,v in solution.items():
            fits[k].append(v)
    for k,v in fits.items():
        v = np.asarray(v).T
        cvresults[k] = v
    del fits, solutions
    return cvresults


def dual2primal_weights(kernel_weights,
                        feature_matrices,
                        feature_priors,
                        feature_hyparams,
                        delays,
                        ):
    '''
    '''
    # WIP: need to fix for kron kernel
    weights = []
    for fi, features in enumerate(feature_matrices):
        Xi = tikutils.delay_signal(features, delays)
        Wi = np.dot(Xi.T, kernel_weights)
        weights.append(Wi)
    return weights


def estimate_simple_stem_wmvnp(features_train,
                               responses_train,
                               features_test=None,
                               responses_test=None,
                               temporal_prior=None,
                               temporal_hhparam=1.0,
                               feature_priors=None,
                               feature_hyparams=None,
                               weights=False,
                               performance=False,
                               predictions=False,
                               ridge_scale=1.0,
                               verbosity=2,
                               method='SVD',
                               ):
    '''
    '''
    if feature_hyparams is None:
        feature_hyparams = [1.0]*len(features_train)

    if features_test is None:
        features_test = [features_test]*len(features_train)

    # we're only using one set in this function
    assert len(feature_hyparams) == len(features_train)

    Ktrain = 0.
    Ktest = 0.
    this_temporal_prior = temporal_prior.get_prior(hhparam=temporal_hhparam)
    for fdx, (fs_train, fs_test, fs_prior, fs_hyper) in enumerate(zip(features_train,
                                                                      features_test,
                                                                      feature_priors,
                                                                      feature_hyparams)):
        Ktrain += kernel_spatiotemporal_prior(fs_train,
                                              this_temporal_prior,
                                              fs_prior.get_prior(fs_hyper),
                                              delays=temporal_prior.delays)

        if fs_test is not None:
            Ktest += kernel_spatiotemporal_prior(fs_train,
                                                 this_temporal_prior,
                                                 fs_prior.get_prior(fs_hyper),
                                                 delays=temporal_prior.delays,
                                                 Xtest=fs_test)

    if np.allclose(Ktest, 0.0):
        Ktest = None

    # solve for this response
    response_solution = solve_l2_dual(Ktrain, responses_train,
                                      Ktest=Ktest,
                                      Ytest=responses_test,
                                      ridges=[ridge_scale],
                                      performance=performance,
                                      predictions=predictions,
                                      weights=weights,
                                      verbose=verbosity > 1,
                                      method=method)

    return response_solution




def hyperopt_estimate_stem_wmvnp(features_train,
                                 responses_train,
                                 features_test=None,
                                 responses_test=None,
                                 normalize_kernel=True,
                                 temporal_prior=None,
                                 feature_priors=None,
                                 weights=False,
                                 predictions=False,
                                 performance=True,
                                 # noise_ceiling_correction=False,
                                 mean_cv_only=False,
                                 folds=(1,5),
                                 method='SVD',
                                 verbosity=1,
                                 cvresults=None,
                                 population_optimal=False,
                                 ntrials=100,
                                 ridge_limits=(-7, 7),
                                 feature_limits=[(0,7)],
                                 spatial_sampler=None,
                                 temporal_sampler=None,
                                 ridge_sampler=None,
                                 ):
    import pickle
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

    if spatial_sampler is None:
        spatial_sampler = hp.loguniform

    if ridge_sampler is None:
        ridge_sampler = hp.loguniform

    delays = temporal_prior.delays
    ndelays = len(delays)

    if features_test is None:
        features_test = [features_test]*len(features_train)

    if len(feature_limits) == 1:
        feature_limits = feature_limits*len(features_train)

    def objective(params):
        # temporal_hhparam = params[0]
        feature_hyparams = params[:-1]
        scale_hyparam = params[-1]

        temporal_prior.set_hhparameters(1.)#temporal_hhparam)

        for fi, feature_prior in enumerate(feature_priors):
            feature_prior.set_hyperparameters(feature_hyparams[fi])

        res = crossval_stem_wmvnp(features_train,
                                  responses_train,
                                  ridges=np.asarray([scale_hyparam]),
                                  normalize_kernel=False,
                                  temporal_prior=temporal_prior,
                                  feature_priors=feature_priors,
                                  performance=True,
                                  folds=folds,
                                  method=method,
                                  verbosity=verbosity,
                                  )

        print params
        cvres = res['cvresults'].mean(0).mean(-1).mean()
        print 'features:', feature_hyparams
        print 'ridges:', scale_hyparam
        print res['spatial'], res['temporal'], res['ridges']
        print cvres
        return {'loss' : (1 - cvres)**2,
                'attachments' : {'internals' : pickle.dumps({'temporal' : res['temporal'],
                                                             'spatial' : res['spatial'],
                                                             'ridges' : res['ridges']}),
                                 },
                'status': STATUS_OK,
                }



    spaces = []
    for i in range(len(features_train)):
        sampler = spatial_sampler('x%0i'%i, feature_limits[i][0], feature_limits[i][1])
        spaces.append(sampler)

    spaces.append(ridge_sampler('ridge_scale', ridge_limits[0], ridge_limits[1]))


    trials = Trials()
    best_params = fmin(objective,
                       space=spaces,
                       algo=tpe.suggest,
                       max_evals=ntrials,
                       trials=trials)



    print best_params
    return trials

if __name__ == '__main__':
    pass
