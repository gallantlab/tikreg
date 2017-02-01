import numpy as np
np.set_printoptions(precision=4, suppress=True)
np.random.seed(1337)

import itertools

import tikypy as tk
from tikypy import (models,
                    spatial_priors as sps,
                    temporal_priors as tps,
                    utils as tikutils,
                    )


def get_abc_data(banded=True, p=50, n=100):
    from scipy.stats import zscore

    if banded:
        weights = np.asarray([1.0, 100.0, 10000.])
    else:
        weights = np.ones(3)

    weights /= np.linalg.norm(weights)

    Aw, (A, Atest), (Yat, Yav) = tikutils.generate_data(n=n, p=p, v=20, testsize=50,
                                                        feature_sparsity=0.5, noise=0.)
    Bw, (B, Btest), (Ybt, Ybv) = tikutils.generate_data(n=n, p=p, v=20, testsize=50,
                                                        feature_sparsity=0.5, noise=0.)
    Cw, (C, Ctest), (Yct, Ycv) = tikutils.generate_data(n=n, p=p, v=20, testsize=50,
                                                        feature_sparsity=0.5, noise=0.)


    responses_train = zscore(Yat*weights[0] + Ybt*weights[1] + Yct*weights[2])
    responses_test = zscore(Yav*weights[0] + Ybv*weights[1] + Ycv*weights[2])

    for rdx in xrange(responses_train.shape[-1]):
        # different noise levels
        noise = np.log(rdx + 1)
        responses_train[:, rdx] += np.random.randn(responses_train.shape[0])*noise
        responses_test[:, rdx] += np.random.randn(responses_test.shape[0])*noise

    responses_train = tikutils.hrf_convolution(responses_train)
    responses_test = tikutils.hrf_convolution(responses_test)

    features_train = [A.astype(np.float32),
                      B.astype(np.float32),
                      C.astype(np.float32)]
    features_test = [Atest.astype(np.float32),
                     Btest.astype(np.float32),
                     Ctest.astype(np.float32)]

    return (features_train, features_test,
            responses_train, responses_test)


def test_fullfit(n=100, p=50):
    ridges = np.logspace(-3,3,10)
    nridges = len(ridges)
    ndelays = 5
    delays = range(ndelays)

    oo = get_abc_data(banded=True, n=n, p=p)
    features_train, features_test, responses_train, responses_test = oo
    features_sizes = [fs.shape[1] for fs in features_train]

    hyparams = np.logspace(0,3,5)
    spatial_priors = [sps.SphericalPrior(features_sizes[0], hyperparameters=[1.]),
                      sps.SphericalPrior(features_sizes[1], hyperparameters=hyparams),
                      sps.SphericalPrior(features_sizes[2], hyperparameters=hyparams),
                      ]

    reload(models)
    temporal_prior = tps.SphericalPrior(delays)
    folds = tikutils.generate_trnval_folds(responses_train.shape[0],
                                           sampler='bcv',
                                           nfolds=(1,5),
                                           )
    folds = list(folds)

    reload(models)
    res  = models.estimate_stem_wmvnp(features_train,
                                      responses_train,
                                      features_test,
                                      responses_test,
                                      ridges=ridges,
                                      normalize_kernel=True,
                                      temporal_prior=temporal_prior,
                                      feature_priors=spatial_priors,
                                      weights=True,
                                      performance=True,
                                      predictions=True,
                                      mean_cv_only=False,
                                      folds=(1,5),
                                      method='SVD',
                                      verbosity=1,
                                      cvresults=None,
                                      )

    for rdx in range(responses_train.shape[-1]):
        optima = res['optima'][rdx]
        temporal_opt, spatial_opt, ridge_scale = optima[0], optima[1:-1], optima[-1]

        Ktrain = 0.
        Ktest = 0.
        this_temporal_prior = temporal_prior.get_prior(hhparam=temporal_opt)
        for fdx, (fs_train, fs_test, fs_prior, fs_hyper) in enumerate(zip(features_train,
                                                                          features_test,
                                                                          spatial_priors,
                                                                          spatial_opt)):
            Ktrain += models.kernel_spatiotemporal_prior(fs_train,
                                                         this_temporal_prior,
                                                         fs_prior.get_prior(fs_hyper),
                                                         delays=temporal_prior.delays)

            if fs_test is not None:
                Ktest += models.kernel_spatiotemporal_prior(fs_train,
                                                            this_temporal_prior,
                                                            fs_prior.get_prior(fs_hyper),
                                                            delays=temporal_prior.delays,
                                                            Xtest=fs_test)

        if np.allclose(Ktest, 0.0):
            Ktest = None

        # solve for this response
        response_solution = models.solve_l2_dual(Ktrain, responses_train[:, [rdx]],
                                                 Ktest=Ktest,
                                                 Ytest=responses_test[:, [rdx]],
                                                 ridges=[ridge_scale],
                                                 performance=True,
                                                 predictions=True,
                                                 weights=True,
                                                 verbose=1,
                                                 method='SVD')


        for k,v in response_solution.items():
            # compare each vector output
            assert np.allclose(res[k][:, rdx].squeeze(), response_solution[k].squeeze())






def test_ols():
    # test we can get OLS solution
    delays = [0]
    ndelays = len(delays)

    # make some features and signal for which we know
    # the optimal ridge parameter is zero
    Af = np.random.randn(150, 10)
    Bf = np.random.randn(150, 20)
    Cf = np.random.randn(150, 30)

    A, Atest = Af[:100], Af[100:]
    B, Btest = Bf[:100], Bf[100:]
    C, Ctest = Cf[:100], Cf[100:]

    nvox = 20
    Aw = np.random.randn(Af.shape[-1], nvox)
    Bw = np.random.randn(Bf.shape[-1], nvox)
    Cw = np.random.randn(Cf.shape[-1], nvox)

    Yf = np.dot(Af, Aw) + np.dot(Bf, Bw) + np.dot(Cf, Cw)
    responses_train, responses_test = Yf[:100], Yf[100:]

    features_train = [A,B,C]
    features_test = [Atest, Btest, Ctest]
    features_sizes = [fs.shape[1] for fs in features_train]

    # solve for OLS using L2 machinery
    direct_fit = models.solve_l2_primal(tikutils.delay_signal(np.hstack(features_train), delays),
                                        responses_train,
                                        tikutils.delay_signal(np.hstack(features_test), delays),
                                        responses_test,
                                        verbose=True,
                                        ridges=[0.],
                                        weights=True,
                                        performance=True,
                                        predictions=True)

    # create feature priors
    spatial_priors = [sps.SphericalPrior(features_sizes[0], hyperparameters=[1]),
                      sps.SphericalPrior(features_sizes[1], hyperparameters=[1]),
                      sps.SphericalPrior(features_sizes[2], hyperparameters=[1]),
                      ]


    # test all priors
    tpriors = [tps.SmoothnessPrior(delays),
               tps.SmoothnessPrior(delays, wishart=True),
               tps.SmoothnessPrior(delays, wishart=False),
               tps.SmoothnessPrior(delays, wishart=np.eye(len(delays))),
               tps.GaussianKernelPrior(delays, sigma=2.0),
               tps.HRFPrior([1] if delays == [0] else delays), # b/c delay at 0 has no covariance
               tps.SphericalPrior(delays),
               ]


    for temporal_prior in tpriors:
        print(temporal_prior)

        all_temporal_hypers = [temporal_prior.get_hyperparameters()]
        all_spatial_hypers = [[1.]]*len(spatial_priors)

        # get all combinations of hyperparameters
        all_hyperparams = itertools.product(*(all_temporal_hypers + all_spatial_hypers))

        Ktrain = 0.
        Ktest = 0.

        for spatiotemporal_hyperparams in all_hyperparams:
            temporal_hyperparam = spatiotemporal_hyperparams[0]
            spatial_hyperparams = spatiotemporal_hyperparams[1:]

            this_temporal_prior = temporal_prior.get_prior(alpha=1.0,
                                                           hhparam=temporal_hyperparam)

            for fdx, (fs_train, fs_test, fs_prior, fs_hyper) in enumerate(zip(features_train,
                                                                              features_test,
                                                                              spatial_priors,
                                                                              spatial_hyperparams)):

                kernel_train = models.kernel_spatiotemporal_prior(fs_train,
                                                                  this_temporal_prior,
                                                                  fs_prior.get_prior(fs_hyper),
                                                                  delays=delays)
                Ktrain += kernel_train

                kernel_test = models.kernel_spatiotemporal_prior(fs_train,
                                                                 this_temporal_prior,
                                                                 fs_prior.get_prior(fs_hyper),
                                                                 Xtest=fs_test,
                                                                 delays=delays)
                Ktest += kernel_test

            fit = models.solve_l2_dual(Ktrain, responses_train,
                                       Ktest, responses_test,
                                       ridges=[0., 1e-03, 1., 10.0, 100.],
                                       verbose=True,
                                       weights=True,
                                       performance=True,
                                       predictions=True)


            # make sure we can predict perfectly
            assert np.allclose(fit['performance'][0], 1.)

            # get the feature weights from the kernel weights
            weights = np.tensordot(tikutils.delay_signal(np.hstack(features_train), delays).T,
                                   fit['weights'], (1,1)).swapaxes(0,1)
            if not np.allclose(this_temporal_prior, 1):
                # scale weights to account for temporal hyper-prior scale
                weights *= this_temporal_prior
            assert np.allclose(weights[0], direct_fit['weights'])
            assert np.allclose(fit['predictions'][0], direct_fit['predictions'].squeeze())


def test_ridge_solution(normalize_kernel=True, method='SVD'):
    # make sure we can recover the ridge solution
    ridges = np.round(np.logspace(-3,3,5), 4)
    nridges = len(ridges)

    delays = np.unique(np.random.randint(0,10,10))
    ndelays = len(delays)

    features_train, features_test, responses_train, responses_test = get_abc_data()
    features_sizes = [fs.shape[1] for fs in features_train]

    spatial_priors = [sps.SphericalPrior(features_sizes[0]),
                      sps.SphericalPrior(features_sizes[1]),
                      sps.SphericalPrior(features_sizes[2]),
                      ]

    reload(models)
    tpriors = [tps.SphericalPrior(delays)]
    temporal_prior = tpriors[0]
    folds = tikutils.generate_trnval_folds(responses_train.shape[0],
                                           sampler='bcv',
                                           nfolds=(1,5),
                                           )
    folds = list(folds)

    res = models.crossval_stem_wmvnp(features_train,
                                     responses_train,
                                     temporal_prior=temporal_prior,
                                     feature_priors=spatial_priors,
                                     folds=folds,
                                     ridges=ridges,
                                     verbosity=2,
                                     method=method,
                                     normalize_kernel=normalize_kernel,
                                     )

    sprior_ridge = res['spatial'][0]
    newridges = res['ridges']

    # direct fit
    X = np.hstack([tikutils.delay_signal(t.astype(np.float64), delays)*(sprior_ridge[i]**-1)\
                   for i,t in enumerate(features_train)])

    fit = models.cvridge(X,
                         responses_train,
                         folds=folds,
                         ridges=newridges,
                         verbose=True
                         )
    print(newridges)
    print(res['spatial'].squeeze())
    print(res['ridges'].squeeze())
    assert np.allclose(fit['cvresults'].squeeze(), res['cvresults'].squeeze())


def test_ridge_solution_raw():
    # make sure we recover the ridge solution
    # when we don't normalize hyperparameters
    test_ridge_solution(normalize_kernel=False)


def test_ridge_solution_chol():
    # test the ridge solution recovery using cholesky decomposition
    test_ridge_solution(normalize_kernel=True, method='Chol')


def test_ridge_solution_raw_chol():
    # test the ridge solution recovery using cholesky decomposition
    # and no kernel normalizations
    test_ridge_solution(normalize_kernel=False, method='Chol')


def test_cv_api(show_figures=False, ntest=50):
    # if show_figures=True, this function will create
    # images of the temporal priors, and the feature prior hyperparameters in 3D

    ridges = [0., 1e-03, 1., 10.0, 100.]
    nridges = len(ridges)
    ndelays = 10
    delays = range(ndelays)

    features_train, features_test, responses_train, responses_test = get_abc_data()
    features_sizes = [fs.shape[1] for fs in features_train]

    spatial_priors = [sps.SphericalPrior(features_sizes[0]),
                      sps.SphericalPrior(features_sizes[1], hyperparameters=np.logspace(-3,3,7)),
                      sps.SphericalPrior(features_sizes[2], hyperparameters=np.logspace(-3,3,7)),
                      ]

    # do not scale first. this removes duplicates
    spatial_priors[0].set_hyperparameters(1.0)

    # non-diagonal hyper-prior
    W = np.random.randn(ndelays, ndelays)
    W = np.dot(W.T, W)

    tpriors = [tps.SphericalPrior(delays),
               tps.SmoothnessPrior(delays, hhparams=np.logspace(-3,1,8)),
               tps.SmoothnessPrior(delays, wishart=True),
               tps.SmoothnessPrior(delays, wishart=False),
               tps.SmoothnessPrior(delays, wishart=W, hhparams=np.logspace(-3,3,5)),
               tps.GaussianKernelPrior(delays, hhparams=np.linspace(1,ndelays/2,ndelays)),
               tps.HRFPrior([1] if delays == [0] else delays),
               ]

    nfolds = (1,5)                      # 1 times 5-fold cross-validation
    folds = tikutils.generate_trnval_folds(responses_train.shape[0], sampler='bcv', nfolds=nfolds)
    nfolds = np.prod(nfolds)

    for ntp, temporal_prior in enumerate(tpriors):
        print(temporal_prior)

        all_temporal_hypers = [temporal_prior.get_hhparams()]
        all_spatial_hypers = [t.get_hyperparameters() for t in spatial_priors]

        # get all combinations of hyperparameters
        all_hyperparams = list(itertools.product(*(all_temporal_hypers + all_spatial_hypers)))
        nspatial_hyperparams = np.prod([len(t) for t in all_spatial_hypers])
        ntemporal_hyperparams = np.prod([len(t) for t in all_temporal_hypers])

        mean_cv_only = False
        results = np.zeros((nfolds,
                            ntemporal_hyperparams,
                            nspatial_hyperparams,
                            nridges,
                            1 if mean_cv_only else responses_train.shape[-1]),
                           dtype=[('fold', np.float32),
                                  ('tp', np.float32),
                                  ('sp', np.float32),
                                  ('ridges', np.float32),
                                  ('responses', np.float32),
                                  ])

        for hyperidx, spatiotemporal_hyperparams in enumerate(all_hyperparams):
            temporal_hyperparam = spatiotemporal_hyperparams[0]
            spatial_hyperparams = spatiotemporal_hyperparams[1:]
            spatial_hyperparams /= np.linalg.norm(spatial_hyperparams)

            # get indices
            shyperidx = np.mod(hyperidx, nspatial_hyperparams)
            thyperidx = int(hyperidx // nspatial_hyperparams)
            print (thyperidx, temporal_hyperparam), (shyperidx, spatial_hyperparams)

            this_temporal_prior = temporal_prior.get_prior(alpha=1.0, hhparam=temporal_hyperparam)

            if show_figures:
                from matplotlib import pyplot as plt

                if (hyperidx == 0) and (ntp == 0):
                    # show points in 3D
                    from tikypy import priors
                    cartesian_points = [t[1:]/np.linalg.norm(t[1:]) for t in all_hyperparams]
                    angles = priors.cartesian2polar(np.asarray(cartesian_points))
                    priors.show_spherical_angles(angles[:,0], angles[:,1])

                if hyperidx == 0:
                    # show priors with different hyper-priors
                    oldthyper = 0
                    plt.matshow(this_temporal_prior, cmap='inferno')
                else:
                    if thyperidx > oldthyper:
                        oldthyper = thyperidx
                        plt.matshow(this_temporal_prior, cmap='inferno')

            # only run a few
            if hyperidx > ntest:
                continue

            Ktrain = 0.
            Kval = 0.

            for fdx, (fs_train, fs_test, fs_prior, fs_hyper) in enumerate(zip(features_train,
                                                                              features_test,
                                                                              spatial_priors,
                                                                              spatial_hyperparams)):

                kernel_train = models.kernel_spatiotemporal_prior(fs_train,
                                                                  this_temporal_prior,
                                                                  fs_prior.get_prior(fs_hyper),
                                                                  delays=delays)
                Ktrain += kernel_train

            kernel_normalizer = tikutils.determinant_normalizer(Ktrain)
            Ktrain /= kernel_normalizer

            # cross-validation
            for ifold, (trnidx, validx) in enumerate(folds):
                ktrn = tikutils.fast_indexing(Ktrain, trnidx, trnidx)
                kval = tikutils.fast_indexing(Ktrain, validx, trnidx)

                fit = models.solve_l2_dual(ktrn, responses_train[trnidx],
                                           kval, responses_train[validx],
                                           ridges=ridges,
                                           verbose=False,
                                           performance=True)
                if mean_cv_only:
                    cvfold = np.nan_to_num(fit['performance']).mean(-1)[...,None]
                else:
                    cvfold = fit['performance']
                results[ifold, thyperidx, shyperidx] = cvfold


def test_stmvn_prior(method='SVD'):
    ridges = np.logspace(0,1,5)
    nridges = len(ridges)
    ndelays = 10
    delays = range(ndelays)

    features_train, features_test, responses_train, responses_test = get_abc_data()
    features_sizes = [fs.shape[1] for fs in features_train]

    spatial_priors = [sps.SphericalPrior(features_sizes[0]),
                      sps.SphericalPrior(features_sizes[1], hyperparameters=np.logspace(-3,3,7)),
                      sps.SphericalPrior(features_sizes[2], hyperparameters=np.logspace(-3,3,7)),
                      ]

    # do not scale first. this removes duplicates
    spatial_priors[0].set_hyperparameters(1.0)

    # non-diagonal hyper-prior
    W = np.random.randn(ndelays, ndelays)
    W = np.dot(W.T, W)

    tpriors = [tps.SphericalPrior(delays),
               tps.GaussianKernelPrior(delays, hhparams=np.linspace(1,ndelays/2,ndelays)),
               tps.SmoothnessPrior(delays, hhparams=np.logspace(-3,1,2)),
               tps.SmoothnessPrior(delays, wishart=W, hhparams=np.logspace(-3,3,5)),
               tps.SmoothnessPrior(delays, wishart=True),
               tps.SmoothnessPrior(delays, wishart=False),
               tps.HRFPrior([1] if delays == [0] else delays),
               ]

    from tikypy import models
    reload(models)
    res = models.crossval_stem_wmvnp(features_train,
                                     responses_train,
                                     # delays=delays,
                                     temporal_prior=tpriors[2],
                                     feature_priors=spatial_priors,
                                     folds=(1,5),
                                     ridges=ridges,
                                     verbosity=1,
                                     method=method,
                                     )

    # find optima
    cvmean = res['cvresults'].mean(0)
    population_optimal = False
    if population_optimal is True:
        cvmean = np.nan_to_num(cvmean).mean(-1)[...,None]

    for idx in range(cvmean.shape[-1]):
        tpopt, spopt, ropt = models.find_optimum_mvn(cvmean[...,idx],
                                                     res['temporal'],
                                                     res['spatial'],
                                                     res['ridges'],
                                                     )
        print "temporal=%0.03f," % float(tpopt),
        print "spatial=(%0.03f,%0.03f, %0.03f)"%tuple(spopt)
        print "ridge=%0.03f"%float(ropt)

    return res


def test_mkl_scaling():
    delays = np.arange(5)
    ndelays = len(delays)

    oo = get_abc_data()
    oo = [[dataset.astype(np.float64) for dataset in fakedat] for fakedat in oo]
    features_train, features_test, responses_train, responses_test = oo
    features_sizes = [fs.shape[1] for fs in features_train]

    spatial_priors = [sps.SphericalPrior(features_sizes[0]),
                      sps.SphericalPrior(features_sizes[1]),
                      sps.SphericalPrior(features_sizes[2]),
                      ]

    temporal_prior = tps.SphericalPrior(delays)
    sprior_ridge = np.ones(3)/np.linalg.norm(np.ones(3))

    K = 0
    for fi,fs in enumerate(features_train):
        K += models.kernel_spatiotemporal_prior(fs,
                                                temporal_prior.get_prior(),
                                                spatial_priors[0].get_prior(sprior_ridge[fi]),
                                                delays=temporal_prior.delays)

        if fi == 0:
            # test the first feature space
            scale = sprior_ridge[0]**-2
            kk = np.dot(tikutils.delay_signal(features_train[0], delays),
                        tikutils.delay_signal(features_train[0], delays).T)*scale
            assert np.allclose(kk, K)

    X = np.hstack([tikutils.delay_signal(t.astype(np.float64), delays)*sprior_ridge[i]**-1 \
                   for i,t in enumerate(features_train)])
    Kn = np.dot(X, X.T)

    assert np.allclose(K, Kn)


def test_hyperopt_functionality():
    import hyperopt
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


    delays = np.arange(5)#np.unique(np.random.randint(0,10,10))
    ndelays = len(delays)

    features_train, features_test, responses_train, responses_test = get_abc_data()
    features_sizes = [fs.shape[1] for fs in features_train]

    feature_priors = [sps.SphericalPrior(features_sizes[0]),
                      sps.SphericalPrior(features_sizes[1]),
                      sps.SphericalPrior(features_sizes[2]),
                      ]

    reload(models)
    tpriors = [tps.SphericalPrior(delays)]
    # tpriors = [tps.SmoothnessPrior(delays, hhparams=np.linspace(0,10,5))]
    temporal_prior = tpriors[0]

    folds = tikutils.generate_trnval_folds(responses_train.shape[0],
                                           sampler='bcv',
                                           nfolds=(1,5),
                                           )
    folds = list(folds)


    # count = 0
    # def increase_count_by_one():
    #     global count    # Needed to modify global copy of globvar
    #     count = count + 1


    def objective(params):
        # increase_count_by_one()
        feature_hyparams = params[:-1]
        scale_hyparams = params[-1]

        temporal_prior.set_hhparameters(1.)

        for fi, feature_prior in enumerate(feature_priors[1:]):
            feature_prior.set_hyperparameters(feature_hyparams[fi])

        # does not affect
        feature_priors[0].set_hyperparameters(1.)
        res = models.crossval_stem_wmvnp(features_train,
                                         responses_train,
                                         ridges=np.asarray([scale_hyparams]),
                                         normalize_kernel=False,
                                         temporal_prior=temporal_prior,
                                         feature_priors=feature_priors,
                                         performance=True,
                                         folds=(2,5),
                                         method='SVD',
                                         verbosity=2,
                                         )
        cvres = res['cvresults'].mean(0).mean(-1).mean()
        print 'features:', feature_hyparams
        print 'ridges:', scale_hyparams
        print res['spatial'], res['temporal'], res['ridges']
        print cvres
        return (1 - cvres)**2#np.max(cvres, 0)


    space = (hp.loguniform('rB', 0, 7),
             hp.loguniform('rC', 0, 7),
             hp.loguniform('ridge', -7, 7),
             )


    ntrials = 100
    trials = Trials()

    best_params = fmin(objective,
                       space=space,
                       algo=tpe.suggest,
                       max_evals=ntrials,
                       trials=trials)


    print best_params
