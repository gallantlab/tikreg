import numpy as np
import itertools

from tikypy import BasePrior
import tikypy.utils as tikutils
from tikypy import kernels as tkernel

##############################
# functions
##############################

def get_delays_from_prior(raw_prior, delays):
    if delays is None:
        prior = raw_prior
        delays = np.arange(raw_prior.shape[0])
    else:
        assert min(delays) >= 0 and max(delays) < raw_prior.shape[0]
        delays = np.asarray(delays)
        prior = tikutils.fast_indexing(raw_prior, delays, delays)
    return prior, delays


class TemporalPrior(BasePrior):
    '''Basic temporal prior
    '''
    def __init__(self, prior, delays=None, **kwargs):
        '''
        '''
        prior, delays = get_delays_from_prior(prior, delays)
        self.delays = delays
        self.ndelays = len(delays)
        super(TemporalPrior, self).__init__(prior, **kwargs)


class CustomPrior(TemporalPrior):
    '''Specify a custom prior
    '''
    def __init__(self, *args, **kwargs):
        '''
        '''
        super(CustomPrior, self).__init__(*args, **kwargs)


class PriorFromPenalty(TemporalPrior):
    '''
    '''
    def __init__(self, penalty, delays=None, **kwargs):
        '''
        '''
        prior = np.zeros_like(penalty)
        if delays is None:
            delays = np.arange(penalty.shape[0])

        super(PriorFromPenalty, self).__init__(prior, delays=delays, **kwargs)

        # overwrite penalty after init
        self.penalty = penalty.copy()
        self.wishart = np.eye(penalty.shape[0])
        self.wishart_lambda = 0.0

    def prior2penalty(self, regularizer=0.0, dodetnorm=False):
        if regularizer > 0.0:
            # make sure we have a valid prior
            assert not np.allclose(self.prior, 0)
            penalty = super(PriorFromPenalty, self).prior2penalty(regularizer=regularizer,
                                                                  dodetnorm=dodetnorm)
        elif dodetnorm:
            # re-scale
            penalty = self.penalty / tikutils.determinant_normalizer(self.penalty)
        else:
            # exact
            penalty = self.penalty
        return penalty

    def set_wishart(self, wishart_array):
        assert np.allclose(wishart_array.shape, self.penalty.shape)
        self.wishart = wishart_array

    def update_prior(self, wishart_lambda=0.0, dodetnorm=False):
        '''
        '''
        self.wishart_lambda = wishart_lambda

        # compute prior
        prior = np.linalg.inv(self.penalty + self.wishart_lambda*self.wishart)
        # select requested delays from prior
        prior, delays = get_delays_from_prior(prior, self.delays)
        # update object prior
        self.prior = prior

        if dodetnorm:
            # normalize
            self.normalize_prior()


class SmoothnessPrior(PriorFromPenalty):
    '''
    '''
    def __init__(self, delays=range(5), order=2, **kwargs):
        '''
        '''
        maxdelay = max(delays)+1
        penalty = tikutils.difference_operator(order, maxdelay)
        CTC = np.dot(penalty, penalty.T)
        super(SmoothnessPrior, self).__init__(CTC, delays=delays, **kwargs)



class SphericalPrior(TemporalPrior):
    '''Equivalent to ridge.
    '''
    def __init__(self, delays=range(5), **kwargs):
        '''
        '''
        raw_prior = np.eye(max(delays)+1)
        super(SphericalPrior, self).__init__(raw_prior, delays=delays, **kwargs)


class HRFPrior(TemporalPrior):
    '''Haemodynamic response function prior
    '''
    def __init__(self, delays=None, dt=2.0, duration=20, **kwargs):
        '''
        '''

        H = tikutils.hrf_default_basis(dt=dt, duration=duration)
        raw_prior = np.dot(H, H.T).astype(np.float64)
        super(HRFPrior, self).__init__(raw_prior, delays=delays, **kwargs)

    def prior2penalty(self, regularizer=1e-08):
        # default HRF prior is low rank, so need to regularize to invert
        return super(HRFPrior, self).prior2penalty(regularizer=regularizer)




class WishartPrior(object):
    def __init__(self, prior, wishart_array):
        '''
        '''
        self.prior = prior
        self.wishart = wishart_array

        self.wishart_lambda = 0.0
        self.detnorm = 1.0
        self.penalty = self.prior2penalty(self.prior)

    def update_wishart(self, wishart_lambda=0.0):
        self.wishart_lambda = wishart_lambda

    def update_prior(self):
        prior = np.linalg.inv(self.penalty + self.wishart_lambda*self.wishart)
        self.detnorm = tikutils.determinant_normalizer(prior)
        prior /= self.detnorm
        self.prior = prior




class GaussianKernelPrior(TemporalPrior):
    '''Smoothness prior
    '''
    def __init__(self, delays=range(5), sigma=1.0,  **kwargs):
        '''
        '''

        full_delays = np.arange(max(delays)+1)
        self.kernel_object = tkernel.lazy_kernel(full_delays[...,None],
                                                 kernel_type='gaussian')

        self.kernel_object.update(sigma)
        prior = self.kernel_object.kernel
        super(GaussianKernelPrior, self).__init__(prior, delays=delays, **kwargs)

    def update_prior(self, sigma=1.0, dodetnorm=False):
        '''
        '''
        if sigma == self.kernel_object.kernel_parameter:
            # it's already set, do nothing
            return

        # compute prior
        self.kernel_object.update(sigma)
        prior = self.kernel_object.kernel
        # select requested delays from prior
        prior, delays = get_delays_from_prior(prior, self.delays)
        # update object prior
        self.prior = prior

        if dodetnorm:
            # normalize
            self.normalize_prior()
