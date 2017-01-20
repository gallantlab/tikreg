import numpy as np
import tikypy as tk
from tikypy import models, spatial_priors as sps, temporal_priors as tps

def test_band_api():
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
    Ytrain, Ytest = Yf[:100], Yf[100:]


    features_train = [A,B,C]
    features_test = [Atest, Btest, Ctest]

    spatial_priors = [sps.SphericalPrior(),
                      sps.SphericalPrior(),
                      sps.SphericalPrior(),
                      ]


    delays = range(10)
    wishart_prior = tps.SphericalPrior(delays)
    temporal_prior = tps.SmoothnessPrior(delays).set_wishart(wishart_prior)
    temporal_prior = tps.GaussianKernelPrior(delays)
    temporal_prior = tps.HRFPrior(delays)
    temporal_prior = tps.SphericalPrior(delays)

    Ktrain = 0.
    Ktest = 0.


    for fs_train, fs_test, feature_prior in zip(features_train,
                                                features_test,
                                                spatial_priors):
        feature_prior.update_prior(fs_train.shape[-1])
        if hasattr(temporal_prior, 'update_prior'):
            temporal_prior.update_prior()

        kernel_train = models.kernel_spatiotemporal_prior(fs_train,
                                                          temporal_prior.asarray,
                                                          feature_prior.asarray)
        Ktrain += kernel_train

        kernel_test = models.kernel_spatiotemporal_prior(fs_train,
                                                         temporal_prior.asarray,
                                                         feature_prior.asarray,
                                                         Xtest=fs_test)
        Ktest += kernel_test

    fit = models.solve_l2_dual(Ktrain, Ytrain,
                               Ktest, Ytest,
                               ridges=[0., 1., 10.0, 100.],
                               verbose=True,
                               weights=True, performance=True)
