import numpy as np
import itertools

import tikypy as tk
from tikypy import (models,
                    spatial_priors as sps,
                    temporal_priors as tps,
                    utils as tikutils,
                    )


def get_abc_data():
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

    return (features_train, features_test,
            responses_train, responses_test)


def test_mkl_ols():
    ndelays=1
    delays = range(ndelays)

    features_train, features_test, responses_train, responses_test = get_abc_data()
    features_sizes = [fs.shape[1] for fs in features_train]

    print('')
    direct_fit = models.solve_l2_primal(tikutils.delay_signal(np.hstack(features_train), delays),
                                        responses_train,
                                        tikutils.delay_signal(np.hstack(features_test), delays),
                                        responses_test,
                                        verbose=True,
                                        weights=True, performance=True)

    spatial_priors = [sps.SphericalPrior(features_sizes[0]),
                      sps.SphericalPrior(features_sizes[1]),
                      sps.SphericalPrior(features_sizes[2]),
                      ]


    tpriors = [tps.SmoothnessPrior(delays),
               tps.SmoothnessPrior(delays, wishart=True),
               tps.SmoothnessPrior(delays, wishart=False),
               tps.SmoothnessPrior(delays, wishart=np.eye(len(delays))),
               tps.GaussianKernelPrior(delays, sigma=2.0),
               tps.HRFPrior([1] if delays == [0] else delays),
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
                                       weights=True, performance=True)


            assert np.allclose(fit['performance'][0], 1.)

            weights = np.tensordot(tikutils.delay_signal(np.hstack(features_train), delays).T,
                                   fit['weights'], (1,1)).swapaxes(0,1)
            if not np.allclose(this_temporal_prior, 1):
                # scale weights
                weights *= this_temporal_prior
            assert np.allclose(weights[0], direct_fit['weights'])


def test_cv_api(show_figures=False, ntest=50):
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

    nfolds = (1,5)                      # x times 5-fold cross-validation
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
    ridges = np.logspace(0,3,5)
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
               tps.SmoothnessPrior(delays, hhparams=np.logspace(-3,1,5)),
               tps.SmoothnessPrior(delays, wishart=True),
               tps.SmoothnessPrior(delays, wishart=False),
               tps.SmoothnessPrior(delays, wishart=W, hhparams=np.logspace(-3,3,5)),
               tps.GaussianKernelPrior(delays, hhparams=np.linspace(1,ndelays/2,ndelays)),
               tps.HRFPrior([1] if delays == [0] else delays),
               ]


    from tikypy import models
    reload(models)
    res = models.spatiotemporal_mvn_prior_regression(features_train,
                                                     responses_train,
                                                     delays=delays,
                                                     temporal_prior=tpriors[1],
                                                     feature_priors=spatial_priors,
                                                     nfolds=(1,5),
                                                     ridges=ridges,
                                                     verbosity=2,
                                                     method=method,
                                                     )
