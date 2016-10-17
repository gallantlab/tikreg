import numpy as np
from scipy import linalg as LA
from scipy.stats import zscore
from collections import defaultdict as ddic

from . import SVD
from .kernels import lazy_kernel
from . import utils

def fit_cv_individual_optimal_kernel(Ktrain, Ktest, cvresults):
    pass

def fit_cv_individual_optimal(Xtrain, Xtest, cvresults,
                              dim1_params,
                              dim2_params):
    dim1, dim2 = argmax_2d(cvresults)

    pass


def argmax_2d(cvmean):
    '''Find the 2D maximum for each sample in the
    last dimension

    Parameters
    ----------
    cvmean: 3D np.array, (x,y,z)
        First two dimensions are parameters,
        last dimension is the response

    Returns
    -------
    dim1: 1D np.array, (z,)
    dim2: 1D np.array, (z,)
        The argmax for that value along the first
        and second dimensions respsectively
    '''
    nresp = cvmean.shape[-1]
    # dim1 = np.ones(nresp).astype(np.int)*-1
    # dim2 = np.ones(nresp).astype(np.int)*-1

    dim1 = np.zeros(nresp).astype(np.int)
    dim2 = np.zeros(nresp).astype(np.int)

    for respidx in xrange(nresp):
        resp_results = cvmean[...,respidx]
        max_val = np.nanmax(resp_results)
        if (max_val == 0) or np.isnan(max_val):
            continue

        argmax1, argmax2 = np.where(resp_results == max_val)
        dim1[respidx] = max(argmax1)
        dim2[respidx] = max(argmax2)
    return dim1, dim2



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
    XTXinv = np.dot(utils.mult_diag(1.0/S**2, V, left=False), V.T)
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


def test_ols():
    B, X, Y = utils.generate_data(noise=0, dozscore=False)
    Bh = ols(X, Y)
    assert np.allclose(Bh, B)
    Bh = _ols(X, Y)
    assert np.allclose(Bh, B)


def olspred(X, Y, Xval=False):
    '''Fit OLS, return predictions ``Yhat``
    '''
    U, S, Vt = SVD(X)
    V = Vt.T
    del Vt
    UTY = np.dot(U.T, Y)
    if (Xval is False) or (Xval is None):
        LH = U
    else:
        LH = np.dot(Xval, utils.mult_diag(1.0/S, V, left=False))
    return np.dot(LH, UTY)


def test_olspred():
    B, (Xtrn, Xval), (Ytrn, Yval) = utils.generate_data(noise=0, valsize=20, dozscore=False)
    Bh = ols(Xtrn, Ytrn)
    Yval_direct = np.dot(Xval, Bh)    # Explicit predictions
    Yval_tricks = olspred(Xtrn, Ytrn, Xval=Xval) # implicit predictions
    assert np.allclose(Yval_tricks, Yval_direct)
    # implicit within-set predictions
    Ytrn_hat = olspred(Xtrn, Ytrn)
    assert np.allclose(Ytrn_hat, Ytrn)


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
    for k,v in results.iteritems():
        v = np.asarray(v)
        v = v.squeeze() if k != 'performance' else v
        if v.ndim <= 1: v = v[...,None]
        # Update
        results[k] = v
    return results


def solve_l2_primal(Xtrain, Ytrain, Xval=None, Yval=None,
                    ridges=[0], EPS=1e-10, verbose=False,
                    performance=False, predictions=False, weights=False):
    '''Solve the (primal) L2 regression problem for each L2 parameter.
    '''
    results = ddic(list)

    if predictions:
        assert Xval is not None
    if performance:
        assert (Yval is not None) and (Xval is not None)
        if Yval.ndim == 1:
            Yval = Yval[...,None]

    U, S, Vt = SVD(Xtrain, full_matrices=False)
    V = Vt.T
    del Vt
    gidx = S > EPS
    S = S[gidx]
    U = U[:, gidx]
    V = V[:, gidx]

    UTY = np.dot(U.T, Ytrain)
    if predictions or performance:
        XvalV = np.dot(Xval, V)

    for lidx, alpha in enumerate(ridges):
        D = S / (S**2 + alpha**2)
        if performance:
            # Compute performance
            XVD = utils.mult_diag(D, XvalV, left=False)
            Ypred = np.dot(XVD, UTY)
            cc = utils.columnwise_correlation(Ypred, Yval, zscoreb=False, axis=0)
            results['performance'].append(cc)
            perf = np.nan_to_num(cc)
            if verbose: print('''alpha %02i: %8.03f, mean=%0.04f, (25,50,75)pctl=(%0.04f,%0.04f,%0.04f), (0.2<r>0.8): (%03i,%03i)''' % (lidx +1, alpha, np.mean(perf), np.percentile(perf, 25), np.median(perf), np.percentile(perf, 75), np.sum(perf < 0.25), np.sum(perf > 0.75)))

        if predictions and performance:
            results['predictions'].append(Ypred)
        elif predictions:
            # Only predictions
            XVD = utils.mult_diag(D, XvalV, left=False)
            Ypred = np.dot(XVD, UTY)
            results['predictions'].append(Ypred)
        if weights:
            # weights
            betas = np.dot(utils.mult_diag(D, V, left=False), UTY)
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
                  Kval=None, Yval=None,
                  ridges=[0.0], EPS=1e-10, verbose=False,
                  performance=False, predictions=False, weights=False):
    '''Solve the dual (kernel) L2 regression problem for each L2 parameter.
    '''
    results = ddic(list)

    if predictions:
        assert Kval is not None
    if performance:
        assert (Yval is not None) and (Kval is not None)
        if Yval.ndim == 1: Yval = Yval[...,None]

    L, Q = LA.eigh(Ktrain)
    gidx = L > EPS
    L = L[gidx]
    Q = Q[:, gidx]

    QTY = np.dot(Q.T, Ytrain)
    if predictions or performance:
        KvalQ = np.dot(Kval, Q)
    for rdx, alpha in enumerate(ridges):
        D = 1.0 / (L + alpha**2)

        if performance:
            KvalQD = utils.mult_diag(D, KvalQ, left=False)
            Ypred = np.dot(KvalQD, QTY)
            cc = utils.columnwise_correlation(Ypred, Yval, zscoreb=False)
            results['performance'].append(cc)

            if verbose:
                perf = np.nan_to_num(cc)
                print('''alpha %02i: %8.03f, mean=%0.04f, (25,50,75)pctl=(%0.04f,%0.04f,%0.04f), (0.2<r>0.8): (%03i,%03i)''' % (rdx +1, alpha, np.mean(perf), np.percentile(perf, 25), np.median(perf), np.percentile(perf, 75), np.sum(perf < 0.25), np.sum(perf > 0.75)))

        if predictions and performance:
            results['predictions'].append(Ypred)
        elif predictions:
            KvalQD = utils.mult_diag(D, KvalQ, left=False)
            Ypred = np.dot(KvalQD, QTY)
            results['predictions'].append(Ypred)

        if weights:
            QD = utils.mult_diag(D, Q, left=False)
            alphas = np.dot(QD, QTY)
            results['weights'].append(alphas)

    return clean_results_dict(dict(results))


def kernel_spatiotemporal_prior(Xtrain, temporal_prior, spatial_prior,
                                Xtest=None, ndelays=10):
    '''Compute the kernel matrix of a model with a spatio-temporal prior'''
    from .utils import delay_signal

    if Xtest is None:
        Xtest = Xtrain
    kernel = 0.0
    for idx in xrange(ndelays):
        Xi = delay_signal(Xtrain, [idx])
        for ddx in xrange(ndelays):
            Xd = delay_signal(Xtest, [ddx])
            kernel += np.dot(temporal_prior[ddx,idx]*np.dot(Xd, spatial_prior), Xi.T)
    return kernel


def test_kernel_kron():
    from .utils import delay_signal

    # generate data
    n,p,d = 20, 10, 5
    Xtrain = np.random.randn(n,p)
    Xtest = np.random.randn(n/2,p)
    # construct prior
    a, b = np.random.randn(p,p), np.random.randn(d,d)
    sigma_x = np.dot(a.T, a)
    sigma_t = np.dot(b.T, b)
    sigma = np.kron(sigma_t, sigma_x)

    Xtrn = delay_signal(Xtrain, range(d))
    Xtst = delay_signal(Xtest, range(d))
    XSXtrn = reduce(np.dot, [Xtrn, sigma, Xtrn.T])

    K = kernel_spatiotemporal_prior(Xtrain, sigma_t, sigma_x, ndelays=d)
    assert np.allclose(XSXtrn, K)
    assert np.allclose(np.corrcoef(XSXtrn.ravel(), K.ravel())[0,1], 1)

    XSXtst = reduce(np.dot, [Xtst, sigma, Xtrn.T])
    K = kernel_spatiotemporal_prior(Xtrain, sigma_t, sigma_x, Xtest=Xtest, ndelays=d)
    assert np.allclose(XSXtst, K)
    assert np.allclose(np.corrcoef(XSXtst.ravel(), K.ravel())[0,1], 1)


def kernel_cvridge(Ktrain, Ytrain,
                   Ktest=None, Ytest=None,
                   ridges=[0.0],
                   folds='cv', nfolds=5, blocklen=5, trainpct=0.8,
                   performance=False, predictions=False, weights=False,
                   verbose=True, EPS=1e-10,
                   ):
    ridges = np.asarray(ridges)
    import time
    start_time = time.time()

    n = Ktrain.shape[0]
    if not isinstance(folds, list):
        folds = utils.generate_trnval_folds(n, sampler=folds,
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
        trn, val = fold
        ntrn, nval = len(trn), len(val)
        if verbose:
            txt = (fdx+1,nfolds,ntrn,nval)
            print('train ridge fold  %i/%i: ntrain=%i, nval=%i'%txt)

        Ktrn = utils.fast_indexing(Ktrain, trn, trn)
        Kval = utils.fast_indexing(Ktrain, val, trn)

        res = solve_l2_dual(Ktrn, Ytrain[trn],
                            Kval, zscore(Ytrain[val]),
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
            kernel_weights=False,
            group_optimal=True,
            ):
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
        * (list) Can also be a list of (train, val) pairs: [(trn1, val1),...]
    nfolds (int):
        Number of learning folds
    blocklen (int):
        Chunk data into blocks of this size, and sample these blocks
    trainpct (float 0-1):
        Percentage of data to use in training if using a bootstrap sampler.
    withinset_test (bool):
        If no ``Xtest`` or ``Ytest`` is given and ``predictions`` and/or
        ``performance`` are requested, compute these values based on training set.
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
        Value used to threshold small eigenvalues
    group_optimal: bool
        Select the optimal `ridge` and `kernel` parameter across all
        responses. If False, it'll fit the optimal for each individual response.

    Returns
    -------
    fit (optional; dict):
        Cross-validation results per response for each fold, kernel and L2 parameters
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

    if kernel_name is None: raise ValueError('Say linear if linear')
    kernel_params = [None] if (kernel_name == 'linear') else kernel_params
    nkparams = len(kernel_params)

    kernel_params = np.asarray(kernel_params)
    ridges = np.asarray(ridges)

    Ytrain, Ytest = check_response_dimensionality(Ytrain, Ytest, allow_test_none=True)
    Xtrain, Xtest = check_response_dimensionality(Xtrain, Xtest, allow_test_none=True)

    # Check for generalized tikhonov
    if Li is not None:
        Xtrain = np.dot(Xtrain, Li)

    n, p = Xtrain.shape

    if not isinstance(folds, list):
        folds = utils.generate_trnval_folds(n, sampler=folds,
                                            nfolds=nfolds,
                                            testpct=1-trainpct,
                                            nchunks=blocklen)
        if kernel_name != 'linear':
            # if kernel is not linear we need to get a list to re-use folds
            folds = [(trn,val) for trn,val in folds]

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
            trn, val = fold
            ntrn, nval = len(trn), len(val)
            if verbose:
                txt = (fdx+1,nfolds,ntrn,nval)
                print('train ridge fold  %i/%i: ntrain=%i, nval=%i'%txt)

            if solve_dual is False:
                res = solve_l2_primal(Xtrain[trn], Ytrain[trn],
                                      Xtrain[val], zscore(Ytrain[val]),
                                      ridges, EPS=EPS,
                                      weights=False,
                                      predictions=False,
                                      performance=True,
                                      verbose=verbose)
            else:
                Ktrain = utils.fast_indexing(kernel,trn, trn)
                Kval = utils.fast_indexing(kernel,val, trn)
                res = solve_l2_dual(Ktrain, Ytrain[trn],
                                    Kval, zscore(Ytrain[val]),
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

    surface = np.nan_to_num(results.mean(0)).mean(-1)[...,None]
    kernmax, ridgemax = argmax_2d(surface) if group_optimal else argmax_2d(results.mean(0))

    if Ytest is not None:
        Ytest = zscore(Ytest)
    elif withinset_test:
        Ytest = zscore(Ytrain)

    if Xtest is not None:
        if Li is not None:
            Xtest = np.dot(Xtest, Li)

    if solve_dual and (Xtest is not None):
        ktest_object = lazy_kernel(Xtest, Xtrain, kernel_type=kernel_name)

    fullfit = {}
    kernel_optimals, ridge_optimals = kernel_params[kernmax], ridges[ridgemax]

    import itertools
    for kernopt, ridgeopt in itertools.product(kernel_params, ridges):
        if kernopt is None:
            voxels = np.where(ridge_optimals == ridgeopt, True, False)
        else:
            voxels = np.logical_and(np.where(kernel_optimals == kernopt, True, False),
                                    np.where(ridge_optimals == ridgeopt, True, False))

        if voxels.sum() == 0:
            continue

        if group_optimal:
            Ytrn, Ytst = Ytrain, Ytest
        else:
            Ytrn = Ytrain[:, voxels]
            Ytst = Ytest[:, voxels] if Ytest is not None else None


        if verbose:
            desc = 'held-out' if (Xtest is not None) else 'within'
            outro = 'Predicting {d} set:\ncvperf={cc},ridge={alph},kernel={kn},kernel_param={kp}'
            outro = outro.format(d=desc,cc=surface.max(),alph=ridgeopt,
                                 kn=kernel_name,kp=kernopt)
            print(outro)

        if solve_dual:
            # Set the parameter to the optimal
            ktrain_object.update(kernopt, verbose=verbose)
            if Xtest is not None:
                # project test data to kernel
                ktest_object.update(kernopt, verbose=verbose)
                ktest = ktest_object.kernel
            elif withinset_test:
                # predict within set if so desired
                ktest = ktrain_object.kernel
            else:
                ktest = None
            fit = solve_l2_dual(ktrain_object.kernel, Ytrn,
                                ktest, Ytst,
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
                # normalize the weights to have an STA equivalent norm
                fit['weights'] *= (ridgeopt**2 + Xtrain.shape[0])/float(Xtrain.shape[0])

        else:
            fit = solve_l2_primal(Xtrain, Ytrn,
                                  Xtest, Ytst,
                                  ridges=[ridgeopt],
                                  performance=performance,
                                  predictions=predictions,
                                  weights=weights,
                                  verbose=verbose,
                                  EPS=EPS,
                                  )
            if 'weights' in fit:
                fit['weights'] *= (ridgeopt**2 + Xtrain.shape[0])/float(Xtrain.shape[0])


        # once fit, we want to project back to
        if (Li is not None) and ('weights' in fit) and (not kernel_weights):
            # project back
            fit['weights'] = np.dot(Li, fit['weights'])

        if group_optimal:
            fullfit = fit
        else:
            if (len(fullfit) == 0):
                # construct the container for all voxels
                fullfit = {k:np.zeros(v.shape[:-1]+(Ytrain.shape[-1],)) for k,v in fit.iteritems()}

            for k,v in fit.iteritems():
                # store results for this voxel
                fullfit[k][...,voxels] = v

    if verbose: print('Duration %0.04f[mins]' % ((time.time()-start_time)/60.))
    fullfit = clean_results_dict(dict(fullfit))
    fullfit['cvresults'] = results
    return fullfit


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


def test_solve_l2_primal():
    ridges = [0.0, 10.0, 100.0, 1000.0]
    ridge_test = 1
    # get some data
    B, (Xtrn, Xval), (Ytrn, Yval) = utils.generate_data(n=100, p=20,
                                                        noise=0, valsize=20, dozscore=False)
    # get direct solution
    Bhat_direct = simple_ridge_primal(Xtrn, Ytrn, ridge=ridges[ridge_test]**2)
    fit = solve_l2_primal(Xtrn, Ytrn, Xval=Xval, Yval=zscore(Yval),
                          ridges=ridges, verbose=False, EPS=0, # NO EPS threshold
                          weights=True, predictions=False, performance=False)
    Bhat_indirect = fit['weights']
    assert np.allclose(Bhat_indirect[ridge_test], Bhat_direct)
    # check we can get OLS
    Bols = ols(Xtrn, Ytrn)
    Bhat_indirect_ols = fit['weights'][0]
    assert np.allclose(Bols, Bhat_indirect_ols)
    # test keyword arguments work as expected
    fit = solve_l2_primal(Xtrn, Ytrn, Xval=Xval, Yval=zscore(Yval),
                          ridges=ridges, verbose=False, EPS=0, # NO EPS threshold
                          weights=False, predictions=True, performance=True)
    assert ('predictions' in fit) and ('performance' in fit) and ('weights' not in fit)
    # check predictions
    Yhat_direct = np.dot(Xval, Bhat_direct)
    Yhat_indirect = fit['predictions']
    assert np.allclose(Yhat_indirect[ridge_test], Yhat_direct)
    # check performance
    cc_direct = utils.columnwise_correlation(Yhat_direct, Yval)
    cc_indirect = fit['performance']
    assert np.allclose(cc_direct, cc_indirect[ridge_test])


def test_solve_l2_dual():
    ridges = [0.0, 10.0, 100.0, 1000.0]
    ridge_test = 2
    # get some data
    B, (Xtrn, Xval), (Ytrn, Yval) = utils.generate_data(n=100, p=20,
                                                        noise=0, valsize=20, dozscore=False)
    # get direct solution
    Bhat_direct = simple_ridge_dual(Xtrn, Ytrn, ridge=ridges[ridge_test]**2)
    Ktrn = np.dot(Xtrn, Xtrn.T)
    Kval = np.dot(Xval, Xtrn.T)
    fit = solve_l2_dual(Ktrn, Ytrn, Kval=Kval, Yval=zscore(Yval),
                        ridges=ridges, verbose=False, EPS=0, # NO EPS threshold
                        weights=True, predictions=False, performance=False)
    # project to linear space
    Bhat_indirect = np.tensordot(Xtrn.T, fit['weights'], (1,1)).swapaxes(0,1)
    assert np.allclose(Bhat_indirect[ridge_test], Bhat_direct)
    # check we can get OLS
    Bols = ols(Xtrn, Ytrn)
    # project to linear space
    Bhat_indirect_ols = np.dot(Xtrn.T, fit['weights'][0])
    assert np.allclose(Bols, Bhat_indirect_ols)
    # test keyword arguments work as expected
    fit = solve_l2_dual(Ktrn, Ytrn, Kval=Kval, Yval=zscore(Yval),
                        ridges=ridges, verbose=False, EPS=0, # NO EPS threshold
                        weights=False, predictions=True, performance=True)
    assert ('predictions' in fit) and ('performance' in fit) and ('weights' not in fit)
    # check predictions
    Yhat_direct = np.dot(Xval, Bhat_direct)
    Yhat_indirect = fit['predictions']
    assert np.allclose(Yhat_indirect[ridge_test], Yhat_direct)
    # check performance
    cc_direct = utils.columnwise_correlation(Yhat_direct, Yval)
    cc_indirect = fit['performance']
    assert np.allclose(cc_direct, cc_indirect[ridge_test])
    # compare against primal representation
    fit_primal = solve_l2_primal(Xtrn, Ytrn, Xval=Xval, Yval=zscore(Yval),
                                 ridges=ridges, verbose=False, EPS=0, # NO EPS threshold
                                 weights=True, predictions=False, performance=False)
    Bhat_primal = fit_primal['weights']
    assert np.allclose(Bhat_primal, Bhat_indirect)

    # test non-linear kernel
    kernels_to_test = ['gaussian', 'poly', 'polyhomo', 'multiquad']
    kernel_params_to_test = [10., 3., 2., 20.]
    ridges = [0] # No regularization
    for kernel_name, kernel_param in zip(kernels_to_test, kernel_params_to_test):
        lzk = lazy_kernel(Xtrn, kernel_type=kernel_name)
        lzk.update(kernel_param)
        alphas = zscore(np.random.randn(Xtrn.shape[0], 20))
        Y = np.dot(lzk.kernel, alphas)
        # NB: multiquad kernel produces negative eigen-values! This means that
        # thresholding the eigen-values to be positive (EPS > 0) will lead to
        # inperfect weight recovery. For this reason, the test uses EPS=None.
        EPS = None if kernel_name == 'multiquad' else 0
        fit = solve_l2_dual(lzk.kernel, Y,
                            ridges=ridges, verbose=False, EPS=EPS,
                            weights=True, predictions=False, performance=False)
        assert np.allclose(alphas, fit['weights'].squeeze())


def test_cvridge():
    ridges = np.logspace(0,3,10)
    voxel = 20
    ridge = 5
    ps = [50, 100]
    ns = [100, 50]

    # test primal and dual
    for N, P in zip(ns, ps):
        # get fake data
        B, (Xt, Xv), (Yt, Yv) = utils.generate_data(n=N, p=P, valsize=30, v=100, noise=2.0)
        # Check all works for 1 voxel case
        fit = cvridge(Xt, Yt[:,voxel].squeeze(),
                      Xtest=Xv, Ytest=Yv[:, voxel].squeeze(),
                      ridges=ridges, kernel_name='linear',
                      kernel_params=None, folds='cv', nfolds=5, blocklen=5,
                      verbose=False, EPS=0, withinset_test=False,
                      performance=True, predictions=True, weights=True)
        cvres = fit['cvresults']
        optidx = np.argmax(cvres.squeeze().mean(0))
        optridge = ridges[optidx]
        B = simple_ridge_primal(Xt, Yt, ridge=optridge**2)
        assert np.allclose(fit['weights'].squeeze(), B[:, voxel])

        # check all works for 1 ridge case
        fit = cvridge(Xt, Yt,
                             Xtest=Xv, Ytest=Yv,
                             ridges=[ridges[ridge]], kernel_name='linear',
                             kernel_params=None, folds='cv', nfolds=5, blocklen=5,
                             verbose=False, EPS=0, withinset_test=False,
                             performance=True, predictions=True, weights=True)
        cvres = fit['cvresults']
        B = simple_ridge_primal(Xt, Yt, ridge=ridges[ridge]**2)
        assert np.allclose(fit['weights'].squeeze(), B)

        # one ridge, one voxel
        fit = cvridge(Xt, Yt[:,voxel].squeeze(),
                             Xtest=Xv, Ytest=Yv[:, voxel].squeeze(),
                             ridges=[ridges[ridge]], kernel_name='linear',
                             kernel_params=None, folds='cv', nfolds=5, blocklen=5,
                             verbose=False, EPS=0, withinset_test=False,
                             performance=True, predictions=True, weights=True)
        cvres = fit['cvresults']
        B = simple_ridge_primal(Xt, Yt, ridge=ridges[ridge]**2)
        assert np.allclose(fit['weights'].squeeze(), B[:, voxel])

        # check predictions work
        fit = cvridge(Xt, Yt,
                             Xtest=Xv, Ytest=Yv,
                             ridges=ridges, kernel_name='linear',
                             kernel_params=None, folds='cv', nfolds=5, blocklen=5,
                             verbose=False, EPS=0, withinset_test=False,
                             performance=True, predictions=True, weights=True)
        cvres = fit['cvresults']
        optidx = np.argmax(cvres.squeeze().mean(0).mean(-1))
        optridge = ridges[optidx]
        B = simple_ridge_primal(Xt, Yt, ridge=optridge**2)
        assert np.allclose(fit['weights'], B)

        # test cv results
        folds = [(np.arange(10,N), np.arange(10)),
                 (np.arange(20,N), np.arange(20)),
                 (np.arange(30,N), np.arange(30)),
                 ]
        fit = cvridge(Xt, Yt,
                      Xtest=Xv, Ytest=Yv,
                      ridges=ridges, kernel_name='linear',
                      kernel_params=None, folds=folds, nfolds=5, blocklen=5,
                      verbose=False, EPS=0, withinset_test=False,
                      performance=True, predictions=True, weights=True)
        cvres = fit['cvresults']
        for fdx in xrange(len(folds)):
            # compute the fold prediction performance
            B = simple_ridge_primal(Xt[folds[fdx][0]],
                                    Yt[folds[fdx][0]],
                                    ridge=ridges[ridge]**2)
            Yhat = np.dot(Xt[folds[fdx][1]], B)
            cc = utils.columnwise_correlation(Yhat, Yt[folds[fdx][1]])
            assert np.allclose(cc, cvres[fdx,0,ridge])

    # test non-linear kernel CV
    Ns = [100, 50]
    Ps = [50, 100]
    from scipy import linalg as LA
    for N, P in zip(Ns, Ps):
        B, (Xtrn, Xval), (Ytrn, Yval) = utils.generate_data(n=N, p=P,
                                                            noise=0, valsize=20,
                                                            dozscore=False)

        # test non-linear kernel
        kernels_to_test = ['gaussian', 'poly', 'polyhomo', 'multiquad']
        kernel_params = [10., 3., 2., 100.]
        ridges = [0.0]
        for kernel_name, kernel_param in zip(kernels_to_test, kernel_params):
            lzk = lazy_kernel(Xtrn, kernel_type=kernel_name)
            lzk.update(kernel_param)
            alphas = zscore(np.random.randn(Xtrn.shape[0], 20))
            Y = np.dot(lzk.kernel, alphas)
            # NB: multiquad kernel produces negative eigen-values! This means that
            # thresholding the eigen-values to be positive (EPS > 0) will lead to
            # inperfect weight recovery. For this reason, the test uses EPS=None.
            EPS = None if kernel_name == 'multiquad' else 0
            fit = cvridge(Xtrn, Y,
                                 ridges=ridges,
                                 kernel_name=kernel_name, kernel_params=kernel_params,
                                 folds='cv', nfolds=5, blocklen=5, trainpct=0.8,
                                 verbose=True, EPS=EPS,
                                 weights=True, predictions=False, performance=False)
            cvres = fit['cvresults']
            surface = np.nan_to_num(cvres.mean(0)).mean(-1)
            # find the best point in the 2D space
            max_point = np.where(surface.max() == surface)
            # make sure it's unique (conservative-ish biggest ridge/parameter)
            max_point = map(max, max_point)
            # The maximum point
            kernmax, ridgemax = max_point
            kernopt, ridgeopt = kernel_params[kernmax], ridges[ridgemax]
            # Solve explicitly
            lzk.update(kernopt)
            L, Q = LA.eigh(lzk.kernel)
            alpha_hat = np.dot(np.dot(Q, np.diag(1.0/L)), np.dot(Q.T, Y))
            assert np.allclose(alpha_hat, fit['weights'].squeeze())

            if N > P:
                # N < P cross-validation will not always work in recovering the true
                # kernel parameter because similar kernel parameters yield close to
                # optimal answers in the folds
                assert np.allclose(alphas, fit['weights'].squeeze())


def test_cvridge_voxelwise():
    ridges = np.logspace(0,3,10)
    voxel = 20
    ridge = 5
    ps = [50, 100]
    ns = [100, 50]

    # test primal and dual
    for N, P in zip(ns, ps):
        # get fake data
        B, (Xt, Xv), (Yt, Yv) = utils.generate_data(n=N, p=P, valsize=30, v=100, noise=2.0)
        # Check all works for 1 voxel case
        fit = cvridge(Xt, Yt[:,voxel].squeeze(),
                      Xtest=Xv, Ytest=Yv[:, voxel].squeeze(),
                      ridges=ridges, kernel_name='linear',
                      kernel_params=None, folds='cv', nfolds=5, blocklen=5,
                      verbose=False, EPS=0, withinset_test=False,
                      performance=True, predictions=True, weights=True,
                      group_optimal=False)
        cvres = fit['cvresults']
        optidx = np.argmax(cvres.squeeze().mean(0))
        optridge = ridges[optidx]
        B = simple_ridge_primal(Xt, Yt, ridge=optridge**2)
        assert np.allclose(fit['weights'].squeeze(), B[:, voxel])

        # check all works for 1 ridge case
        fit = cvridge(Xt, Yt,
                      Xtest=Xv, Ytest=Yv,
                      ridges=[ridges[ridge]], kernel_name='linear',
                      kernel_params=None, folds='cv', nfolds=5, blocklen=5,
                      verbose=False, EPS=0, withinset_test=False,
                      performance=True, predictions=True, weights=True,
                      group_optimal=False)
        cvres = fit['cvresults']
        B = simple_ridge_primal(Xt, Yt, ridge=ridges[ridge]**2)
        assert np.allclose(fit['weights'].squeeze(), B)

        # one ridge, one voxel
        fit = cvridge(Xt, Yt[:,voxel].squeeze(),
                      Xtest=Xv, Ytest=Yv[:, voxel].squeeze(),
                      ridges=[ridges[ridge]], kernel_name='linear',
                      kernel_params=None, folds='cv', nfolds=5, blocklen=5,
                      verbose=False, EPS=0, withinset_test=False,
                      performance=True, predictions=True, weights=True,
                      group_optimal=False)
        cvres = fit['cvresults']
        B = simple_ridge_primal(Xt, Yt, ridge=ridges[ridge]**2)
        assert np.allclose(fit['weights'].squeeze(), B[:, voxel])

        # check predictions work
        fit = cvridge(Xt, Yt,
                      Xtest=Xv, Ytest=Yv,
                      ridges=ridges, kernel_name='linear',
                      kernel_params=None, folds='cv', nfolds=5, blocklen=5,
                      verbose=False, EPS=0, withinset_test=False,
                      performance=True, predictions=True, weights=True,
                      group_optimal=False)
        cvres = fit['cvresults']

        optidxs = np.argmax(cvres.mean(0).squeeze(), 0)
        for odx, optidx in enumerate(optidxs):
            optridge = ridges[optidx]
            B = simple_ridge_primal(Xt, Yt[:, odx], ridge=optridge**2)
            assert np.allclose(fit['weights'][:, odx], B)

        # test cv results
        folds = [(np.arange(10,N), np.arange(10)),
                 (np.arange(20,N), np.arange(20)),
                 (np.arange(30,N), np.arange(30)),
                 ]
        fit = cvridge(Xt, Yt,
                      Xtest=Xv, Ytest=Yv,
                      ridges=ridges, kernel_name='linear',
                      kernel_params=None, folds=folds, nfolds=5, blocklen=5,
                      verbose=False, EPS=0, withinset_test=False,
                      performance=True, predictions=True, weights=True,
                      group_optimal=False)
        cvres = fit['cvresults']
        for fdx in xrange(len(folds)):
            # compute the fold prediction performance
            B = simple_ridge_primal(Xt[folds[fdx][0]],
                                    Yt[folds[fdx][0]],
                                    ridge=ridges[ridge]**2)
            Yhat = np.dot(Xt[folds[fdx][1]], B)
            cc = utils.columnwise_correlation(Yhat, Yt[folds[fdx][1]])
            assert np.allclose(cc, cvres[fdx,0,ridge])

    # test non-linear kernel CV
    Ns = [100, 50]
    Ps = [50, 100]
    from scipy import linalg as LA
    for N, P in zip(Ns, Ps):
        B, (Xtrn, Xval), (Ytrn, Yval) = utils.generate_data(n=N, p=P,
                                                            noise=0, valsize=20,
                                                            dozscore=False)

        # test non-linear kernel
        kernels_to_test = ['gaussian', 'poly', 'polyhomo', 'multiquad']
        kernel_params = [10., 3., 2., 100.]
        ridges = [0.0]
        for kernel_name, kernel_param in zip(kernels_to_test, kernel_params):
            lzk = lazy_kernel(Xtrn, kernel_type=kernel_name)
            lzk.update(kernel_param)
            alphas = zscore(np.random.randn(Xtrn.shape[0], 20))
            Y = np.dot(lzk.kernel, alphas)
            # NB: multiquad kernel produces negative eigen-values! This means that
            # thresholding the eigen-values to be positive (EPS > 0) will lead to
            # inperfect weight recovery. For this reason, the test uses EPS=None.
            EPS = None if kernel_name == 'multiquad' else 0
            fit = cvridge(Xtrn, Y,
                          ridges=ridges,
                          kernel_name=kernel_name, kernel_params=kernel_params,
                          folds='cv', nfolds=5, blocklen=5, trainpct=0.8,
                          verbose=True, EPS=EPS,
                          weights=True, predictions=False, performance=False,
                          group_optimal=False)
            cvresults = fit['cvresults']

            for voxidx in xrange(cvresults.shape[-1]):
                # only for one voxel
                cvres = cvresults[...,voxidx][...,None]
                surface = np.nan_to_num(cvres.mean(0)).mean(-1)
                # find the best point in the 2D space
                max_point = np.where(surface.max() == surface)
                # make sure it's unique (conservative-ish biggest ridge/parameter)
                max_point = map(max, max_point)
                # The maximum point
                kernmax, ridgemax = max_point
                kernopt, ridgeopt = kernel_params[kernmax], ridges[ridgemax]
                # Solve explicitly
                lzk.update(kernopt)
                L, Q = LA.eigh(lzk.kernel)
                alpha_hat = np.dot(np.dot(Q, np.diag(1.0/L)), np.dot(Q.T, Y[:, voxidx]))
                assert np.allclose(alpha_hat, fit['weights'][...,voxidx].squeeze())

                if N > P:
                    # N < P cross-validation will not always work in recovering the true
                    # kernel parameter because similar kernel parameters yield close to
                    # optimal answers in the folds
                    assert np.allclose(alphas[:, voxidx], fit['weights'][...,voxidx].squeeze())
    return


def simple_generalized_tikhonov(X, Y, L, ridge=10.0):
    '''Direct implementation of generalized tikhonov regression
    '''
    XTXLTL = np.dot(X.T, X) + ridge*np.dot(L.T, L)
    XTY = np.dot(X.T, Y)
    betas = np.dot(LA.inv(XTXLTL), XTY)
    return betas


def direct_generalized_tikhonov(X, Y, LTL, ridge=10.0):
    '''Direct implementation of generalized tikhonov regression
    '''
    XTXLTL = np.dot(X.T, X) + ridge*LTL
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
    alphas = np.dot(AATIi, Y)
    weights = np.dot(A.T, alphas)
    betas = np.dot(Li, weights)
    return betas


def test_generalized_tikhonov():
    Ns = [100, 50]
    Ps = [50, 100]
    for N, p in zip(Ns, Ps):
        B, (X, Xval), (Y, Yval) = utils.generate_data(n=N, p=p, valsize=30)
        Yval = zscore(Yval)
        L = np.random.randint(0, 100, (p,p))
        Li = LA.inv(L)
        ridge = 10.0
        direct = simple_generalized_tikhonov(X, Y, L, ridge=ridge**2)
        stdform = generalized_tikhonov(X, Y, Li, ridge=ridge**2)
        stdform_dual = _generalized_tikhonov_dual(X, Y, Li, ridge=ridge**2)
        assert np.allclose(direct, stdform)
        assert np.allclose(direct, stdform_dual)

        # compute predictions and performance
        Yhat = np.dot(Xval, direct)
        cc = utils.columnwise_correlation(Yhat, Yval)

        # use standard machinery
        Atrn = np.dot(X, Li)
        Aval = np.dot(Xval, Li)
        fit = solve_l2_primal(Atrn, Y, Aval, Yval=Yval,
                              ridges=[ridge], performance=True,
                              weights=True, predictions=True)
        W = np.dot(Li, fit['weights'].squeeze())
        assert np.allclose(W, direct)
        assert np.allclose(fit['predictions'], Yhat)
        assert np.allclose(fit['performance'], cc)

        # use standard machiner dual
        Atrn = np.dot(X, Li)
        Aval = np.dot(Xval, Li)
        Ktrn = np.dot(Atrn, Atrn.T)
        Kval = np.dot(Aval, Atrn.T)
        fit = solve_l2_dual(Ktrn, Y, Kval, Yval=Yval,
                            ridges=[ridge], performance=True,
                            weights=True, predictions=True)
        W = np.dot(Li, np.dot(Atrn.T, fit['weights'].squeeze()))
        assert np.allclose(W, direct)
        assert np.allclose(fit['predictions'], Yhat)
        assert np.allclose(fit['performance'], cc)

        # Check that it works
        fit = cvridge(X, Y, Xtest=Xval, Ytest=Yval,
                                 ridges=[ridge], Li=Li,
                                 verbose=False,
                                 weights=True,
                                 performance=True,
                                 predictions=True)
        cvresults = fit['cvresults']
        assert np.allclose(fit['weights'], direct)
        assert np.allclose(fit['performance'], cc)
        assert np.allclose(fit['predictions'], Yhat)


if __name__ == '__main__':
    pass
