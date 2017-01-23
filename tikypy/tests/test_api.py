import numpy as np
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
               tps.GaussianKernelPrior(delays),
               tps.HRFPrior([1] if delays == [0] else delays),
               tps.SphericalPrior(delays),
               ]


    for temporal_prior in tpriors:
        print(temporal_prior)
        Ktrain = 0.
        Ktest = 0.

        for fs_train, fs_test, feature_prior in zip(features_train,
                                                    features_test,
                                                    spatial_priors):

            if hasattr(temporal_prior, 'update_prior'):
                temporal_prior.update_prior()
            alpha = 1.0
            kernel_train = models.kernel_spatiotemporal_prior(fs_train,
                                                              temporal_prior.get_prior(alpha),
                                                              feature_prior.get_prior(alpha),
                                                              delays=delays)
            Ktrain += kernel_train

            kernel_test = models.kernel_spatiotemporal_prior(fs_train,
                                                             temporal_prior.get_prior(alpha),
                                                             feature_prior.get_prior(alpha),
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
        if not np.allclose(temporal_prior.get_prior(alpha), 1):
            # scale weights
            weights *= temporal_prior.get_prior(alpha)
        assert np.allclose(weights[0], direct_fit['weights'])
