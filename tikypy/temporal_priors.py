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
    def __init__(self, penalty, delays=None, wishart=True, **kwargs):
        '''
        '''
        prior = np.zeros_like(penalty)
        if delays is None:
            delays = np.arange(penalty.shape[0])

        super(PriorFromPenalty, self).__init__(prior, delays=delays, **kwargs)

        # overwrite penalty after init
        self.penalty = penalty.copy()

        # set prior on prior
        if isinstance(wishart, np.ndarray):
            assert np.allclose(wishart.shape, self.penalty.shape)
            self.wishart = wishart
        elif wishart is True:
            self.wishart = np.eye(self.penalty.shape[0])
        elif wishart is False:
            self.wishart = np.zeros_like(self.penalty)
        else:
            raise ValueError('invalid prior for prior')

        # compute prior
        self.prior = self.get_prior(alpha=1.0)



    def set_wishart(self, wishart):
        if isinstance(wishart, BasePrior):
            wishart = wishart.asarray
        assert np.allclose(wishart.shape, self.penalty.shape)
        self.wishart = wishart


    def get_prior(self, alpha=1.0, wishart_alpha=0.0, dodetnorm=False):
        '''
        '''
        self.wishart_alpha = wishart_alpha

        # compute prior
        prior = np.linalg.inv(self.penalty + self.wishart_alpha*self.wishart)
        # select requested delays from prior
        prior, delays = get_delays_from_prior(prior, self.delays)
        # update object prior
        self.prior = prior

        if dodetnorm:
            # normalize
            self.normalize_prior()

        return alpha*self.prior

class SmoothnessPrior(PriorFromPenalty):
    '''
    '''
    def __init__(self, delays=range(5), order=2, wishart=True, **kwargs):
        '''
        '''
        maxdelay = max(delays)+1
        penalty = tikutils.difference_operator(order, maxdelay)
        CTC = np.dot(penalty, penalty.T)
        super(SmoothnessPrior, self).__init__(CTC, delays=delays, wishart=wishart, **kwargs)



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
        if delays is not None:
            if len(delays) == 1 and delays[0] == 0:
                raise ValueError('Using delay zero by itself is not allowed')
        H = tikutils.hrf_default_basis(dt=dt, duration=duration)
        raw_prior = np.dot(H, H.T).astype(np.float64)
        super(HRFPrior, self).__init__(raw_prior, delays=delays, **kwargs)


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

    def get_prior(self, alpha=1.0, sigma=1.0, dodetnorm=False):
        '''
        '''
        if sigma == self.kernel_object.kernel_parameter:
            # it's already set, do nothing
            pass
        else:
            # update gaussian width
            self.kernel_object.update(sigma)

        prior = self.kernel_object.kernel
        # select requested delays from prior
        prior, delays = get_delays_from_prior(prior, self.delays)
        # update object prior
        self.prior = prior

        if dodetnorm:
            # normalize
            self.normalize_prior()

        return alpha*self.prior
