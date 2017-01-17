import numpy as np
import itertools

import utils as tikutils


##############################
# Classes
##############################

class BasePrior(object):
    '''Base class for priors
    '''
    def __init__(self, prior, dodetnorm=False):
        '''
        '''
        assert prior.ndim == 2

        if dodetnorm:
            self.detnorm = tikutils.determinant_normalizer(prior)
        else:
            self.detnorm = 1.0
        self.prior = prior / self.detnorm
        self.penalty = 0.0
        self.dodetnorm = dodetnorm

    @property
    def asarray(self):
        return self.prior

    def prior2penalty(self, regularizer=0.0):
        penalty = np.linalg.inv(self.prior + regularizer*np.eye(self.prior.shape[0]))
        if self.dodetnorm:
            penalty /= tikutils.determinant_normalizer(penalty)
        self.penalty = penalty
        return penalty

    def normalize_prior(self):
        self.detnorm = tikutils.determinant_normalizer(self.prior)
        self.prior /= self.detnorm


def get_delays_from_prior(raw_prior, delays):
    if delays is None:
        prior = raw_prior
        delays = np.arange(raw_prior.shape[0])
    else:
        assert (min(delays) >= 0) and (max(delays) < raw_prior.shape[0])
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


class SphericalPrior(TemporalPrior):
    '''Equivalent to ridge.
    '''
    def __init__(self, delays=range(5), **kwargs):
        '''
        '''
        raw_prior = np.eye(len(np.linspace(0, max(delays))))
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
        # default is low rank, so need to regularize to invert
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






class PriorFromPenalty(object):
    '''
    '''
    def __init__(self, penalty, *args, **kwargs):
        '''
        '''
        self.nn = penalty.shape[0]
        self.wishart_lambda = 0.0
        self.wishart = 0.0
        self.prior = 1.0
        self.penalty = penalty
        self.detnorm = 1.0

    def set_wishart(self, wishart_array):
        self.wishart = wishart_array

    def set_prior(self, wishart_lambda=0.0):
        '''
        '''
        self.wishart_lambda = wishart_lambda
        prior = np.linalg.inv(self.penalty + self.wishart_lambda*self.wishart)
        self.detnorm = tikutils.determinant_normalizer(prior)
        prior /= self.detnorm
        self.prior = prior

    def get_prior(self, *args, **kwargs):
        '''
        '''
        self.set_prior(*args, **kwargs)
        return self.prior







class GaussianKernelPrior(TemporalPrior):
    '''Smoothness prior
    '''
    def __init__(self, *args, **kwargs):
        '''
        '''
        super(GaussianKernelPrior, self).__init__(*args, **kwargs)
