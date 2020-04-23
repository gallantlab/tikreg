from tikreg.models import *             # TODO:
from tikreg.models import _ols, _generalized_tikhonov_dual

import tikreg.utils as tikutils

def test_kernel_kron():
    # generate data
    n,p,d = 20, 10, 5
    delays = range(d)
    Xtrain = np.random.randn(n,p)
    Xtest = np.random.randn(int(n/2),p)
    # construct prior
    a, b = np.random.randn(p,p), np.random.randn(d,d)
    sigma_x = np.dot(a.T, a)
    sigma_t = np.dot(b.T, b)
    sigma = np.kron(sigma_t, sigma_x)

    Xtrn = tikutils.delay_signal(Xtrain, delays)
    Xtst = tikutils.delay_signal(Xtest, delays)
    XSXtrn = np.linalg.multi_dot([Xtrn, sigma, Xtrn.T])

    K = kernel_spatiotemporal_prior(Xtrain, sigma_t, sigma_x, delays=delays)
    assert np.allclose(XSXtrn, K)
    assert np.allclose(np.corrcoef(XSXtrn.ravel(), K.ravel())[0,1], 1)

    XSXtst = np.linalg.multi_dot([Xtst, sigma, Xtrn.T])
    K = kernel_spatiotemporal_prior(Xtrain, sigma_t, sigma_x, Xtest=Xtest, delays=delays)
    assert np.allclose(XSXtst, K)
    assert np.allclose(np.corrcoef(XSXtst.ravel(), K.ravel())[0,1], 1)


def test_kernel_banded_temporal():
    A = np.random.randn(10,10)
    B = np.random.randn(20,10)
    ridge_scale = 3.0**2
    STS = np.eye(10)*ridge_scale
    T = np.random.randn(5,5)
    TTT = np.dot(T.T, T)

    # ktrain
    K = kernel_spatiotemporal_prior(A, TTT, STS,
                                    delays=range(5))
    kk = kernel_banded_temporal_prior(np.dot(A, A.T), TTT, ridge_scale,
                                      delays=range(5))
    assert np.allclose(K, kk)
    # ktest
    K = kernel_spatiotemporal_prior(A, TTT, STS, Xtest=B,
                                    delays=range(5))
    kk = kernel_banded_temporal_prior(np.dot(B, A.T), TTT, ridge_scale,
                                      delays=range(5))
    assert np.allclose(K, kk)


def test_ols():
    B, X, Y = tikutils.generate_data(noise=0, dozscore=False)
    Bh = ols(X, Y)
    assert np.allclose(Bh, B)
    Bh = _ols(X, Y)
    assert np.allclose(Bh, B)


def test_olspred():
    B, (Xtrn, Xtest), (Ytrn, Ytest) = tikutils.generate_data(noise=0, testsize=20, dozscore=False)
    Bh = ols(Xtrn, Ytrn)
    Ytest_direct = np.dot(Xtest, Bh)    # Explicit predictions
    Ytest_tricks = olspred(Xtrn, Ytrn, Xtest=Xtest) # implicit predictions
    assert np.allclose(Ytest_tricks, Ytest_direct)
    # implicit within-set predictions
    Ytrn_hat = olspred(Xtrn, Ytrn)
    assert np.allclose(Ytrn_hat, Ytrn)


def test_solve_l2_primal():
    ridges = [0.0, 10.0, 100.0, 1000.0]
    ridge_test = 1
    # get some data
    B, (Xtrn, Xtest), (Ytrn, Ytest) = tikutils.generate_data(n=100, p=20,
                                                        noise=0, testsize=20, dozscore=False)
    # get direct solution
    Bhat_direct = simple_ridge_primal(Xtrn, Ytrn, ridge=ridges[ridge_test]**2)
    fit = solve_l2_primal(Xtrn, Ytrn, Xtest=Xtest, Ytest=zscore(Ytest),
                          ridges=ridges, verbose=False, EPS=0, # NO EPS threshold
                          weights=True, predictions=False, performance=False)
    Bhat_indirect = fit['weights']
    assert np.allclose(Bhat_indirect[ridge_test], Bhat_direct)
    # check we can get OLS
    Bols = ols(Xtrn, Ytrn)
    Bhat_indirect_ols = fit['weights'][0]
    assert np.allclose(Bols, Bhat_indirect_ols)
    # test keyword arguments work as expected
    fit = solve_l2_primal(Xtrn, Ytrn, Xtest=Xtest, Ytest=zscore(Ytest),
                          ridges=ridges, verbose=False, EPS=0, # NO EPS threshold
                          weights=False, predictions=True, performance=True)
    assert ('predictions' in fit) and ('performance' in fit) and ('weights' not in fit)
    # check predictions
    Yhat_direct = np.dot(Xtest, Bhat_direct)
    Yhat_indirect = fit['predictions']
    assert np.allclose(Yhat_indirect[ridge_test], Yhat_direct)
    # check performance
    cc_direct = tikutils.columnwise_correlation(Yhat_direct, Ytest)
    cc_indirect = fit['performance']
    assert np.allclose(cc_direct, cc_indirect[ridge_test])


def test_solve_l2_dual():
    ridges = [0.0, 10.0, 100.0, 1000.0]
    ridge_test = 2
    # get some data
    B, (Xtrn, Xtest), (Ytrn, Ytest) = tikutils.generate_data(n=100, p=20,
                                                        noise=0, testsize=20, dozscore=False)
    # get direct solution
    Bhat_direct = simple_ridge_dual(Xtrn, Ytrn, ridge=ridges[ridge_test]**2)
    Ktrn = np.dot(Xtrn, Xtrn.T)
    Ktest = np.dot(Xtest, Xtrn.T)
    fit = solve_l2_dual(Ktrn, Ytrn, Ktest=Ktest, Ytest=zscore(Ytest),
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
    fit = solve_l2_dual(Ktrn, Ytrn, Ktest=Ktest, Ytest=zscore(Ytest),
                        ridges=ridges, verbose=False, EPS=0, # NO EPS threshold
                        weights=False, predictions=True, performance=True)
    assert ('predictions' in fit) and ('performance' in fit) and ('weights' not in fit)
    # check predictions
    Yhat_direct = np.dot(Xtest, Bhat_direct)
    Yhat_indirect = fit['predictions']
    assert np.allclose(Yhat_indirect[ridge_test], Yhat_direct)
    # check performance
    cc_direct = tikutils.columnwise_correlation(Yhat_direct, Ytest)
    cc_indirect = fit['performance']
    assert np.allclose(cc_direct, cc_indirect[ridge_test])
    # compare against primal representation
    fit_primal = solve_l2_primal(Xtrn, Ytrn, Xtest=Xtest, Ytest=zscore(Ytest),
                                 ridges=ridges, verbose=False, EPS=0, # NO EPS threshold
                                 weights=True, predictions=False, performance=False)
    Bhat_primal = fit_primal['weights']
    assert np.allclose(Bhat_primal, Bhat_indirect)

    # test non-linear kernel
    kernels_to_test = ['gaussian', 'ihpolykern', 'hpolykern', 'multiquad']
    kernel_params_to_test = [10., 3., 2., 20.]
    ridges = [0] # No regularization
    for kernel_name, kernel_param in zip(kernels_to_test, kernel_params_to_test):
        lzk = lazy_kernel(Xtrn, kernel_type=kernel_name)
        lzk.update(kernel_param)
        rlambdas = zscore(np.random.randn(Xtrn.shape[0], 20))
        Y = np.dot(lzk.kernel, rlambdas)
        # NB: multiquad kernel produces negative eigen-values! This means that
        # thresholding the eigen-values to be positive (EPS > 0) will lead to
        # inperfect weight recovery. For this reason, the test uses EPS=None.
        EPS = None if kernel_name == 'multiquad' else 0
        fit = solve_l2_dual(lzk.kernel, Y,
                            ridges=ridges, verbose=False, EPS=EPS,
                            weights=True, predictions=False, performance=False)
        assert np.allclose(rlambdas, fit['weights'].squeeze())


def test_cvridge():
    ridges = np.logspace(1,3,10)
    voxel = 20
    ridge = 5
    ps = [50, 100]
    ns = [100, 50]

    # test primal and dual
    for N, P in zip(ns, ps):
        # get fake data
        B, (Xt, Xv), (Yt, Yv) = tikutils.generate_data(n=N, p=P, testsize=30, v=100, noise=2.0)
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
        for fdx in range(len(folds)):
            # compute the fold prediction performance
            B = simple_ridge_primal(Xt[folds[fdx][0]],
                                    Yt[folds[fdx][0]],
                                    ridge=ridges[ridge]**2)
            Yhat = np.dot(Xt[folds[fdx][1]], B)
            cc = tikutils.columnwise_correlation(Yhat, Yt[folds[fdx][1]])
            assert np.allclose(cc, cvres[fdx,0,ridge])

    # test non-linear kernel CV
    Ns = [100, 50]
    Ps = [50, 100]
    from scipy import linalg as LA

    np.random.seed(8)
    for N, P in zip(Ns, Ps):
        B, (Xtrn, Xtest), (Ytrn, Ytest) = tikutils.generate_data(n=N, p=P,
                                                            noise=0, testsize=20,
                                                            dozscore=False)

        # test non-linear kernel
        kernels_to_test = ['gaussian', 'ihpolykern', 'hpolykern', 'multiquad']
        kernel_params = [10., 3., 2., 100.]
        ridges = [0.0]
        for kernel_name, kernel_param in zip(kernels_to_test, kernel_params):
            lzk = lazy_kernel(Xtrn, kernel_type=kernel_name)
            lzk.update(kernel_param)
            rlambdas = zscore(np.random.randn(Xtrn.shape[0], 20))
            Y = np.dot(lzk.kernel, rlambdas)
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
            rlambda_hat = np.dot(np.dot(Q, np.diag(1.0/L)), np.dot(Q.T, Y))
            assert np.allclose(rlambda_hat, fit['weights'].squeeze())

            if N > P:
                # N < P cross-testidation will not always work in recovering the true
                # kernel parameter because similar kernel parameters yield close to
                # optimal answers in the folds
                # NB: gaussian kernel doesn't always pass this test because
                #     the optimal kernel parameter is not always found.
                #     the np.seed fixes this.
                assert np.allclose(rlambdas, fit['weights'].squeeze())


def test_generalized_tikhonov():
    Ns = [100, 50]
    Ps = [50, 100]
    for N, p in zip(Ns, Ps):
        B, (X, Xtest), (Y, Ytest) = tikutils.generate_data(n=N, p=p, testsize=30)
        Ytest = zscore(Ytest)
        L = np.random.randint(0, 100, (p,p))
        Li = LA.inv(L)
        ridge = 10.0
        direct = simple_generalized_tikhonov(X, Y, L, ridge=ridge**2)
        stdform = generalized_tikhonov(X, Y, Li, ridge=ridge**2)
        stdform_dual = _generalized_tikhonov_dual(X, Y, Li, ridge=ridge**2)
        assert np.allclose(direct, stdform)
        assert np.allclose(direct, stdform_dual)

        # compute predictions and performance
        Yhat = np.dot(Xtest, direct)
        cc = tikutils.columnwise_correlation(Yhat, Ytest)

        # use standard machinery
        Atrn = np.dot(X, Li)
        Atest = np.dot(Xtest, Li)
        fit = solve_l2_primal(Atrn, Y, Atest, Ytest=Ytest,
                              ridges=[ridge], performance=True,
                              weights=True, predictions=True)
        W = np.dot(Li, fit['weights'].squeeze())
        assert np.allclose(W, direct)
        assert np.allclose(fit['predictions'], Yhat)
        assert np.allclose(fit['performance'], cc)

        # use standard machiner dual
        Atrn = np.dot(X, Li)
        Atest = np.dot(Xtest, Li)
        Ktrn = np.dot(Atrn, Atrn.T)
        Ktest = np.dot(Atest, Atrn.T)
        fit = solve_l2_dual(Ktrn, Y, Ktest, Ytest=Ytest,
                            ridges=[ridge], performance=True,
                            weights=True, predictions=True)
        W = np.dot(Li, np.dot(Atrn.T, fit['weights'].squeeze()))
        assert np.allclose(W, direct)
        assert np.allclose(fit['predictions'], Yhat)
        assert np.allclose(fit['performance'], cc)

        # Check that it works
        fit = cvridge(X, Y, Xtest=Xtest, Ytest=Ytest,
                                 ridges=[ridge], Li=Li,
                                 verbose=False,
                                 weights=True,
                                 performance=True,
                                 predictions=True)
        cvresults = fit['cvresults']
        assert np.allclose(fit['weights'], direct)
        assert np.allclose(fit['performance'], cc)
        assert np.allclose(fit['predictions'], Yhat)
