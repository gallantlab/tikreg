import numpy as np
import itertools

import tikypy.utils as tikutils


##############################
# functions
##############################

def get_delays_from_prior(raw_prior, delays):
    if delays is None:
        prior = raw_prior
        delays = np.arange(raw_prior.shape[0])
    else:
        assert (min(delays) >= 0) and (max(delays) < raw_prior.shape[0])
        delays = np.asarray(delays)
        prior = tikutils.fast_indexing(raw_prior, delays, delays)
    return prior, delays



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

        self.prior = prior.copy()
        self.detnorm = 1.0
        self.penalty = None
        self.penalty_detnorm = 1.0
        self.dodetnorm = dodetnorm

        if self.dodetnorm:
            self.normalize_prior()

    @property
    def asarray(self):
        return self.prior

    def prior2penalty(self, regularizer=0.0, dodetnorm=False):
        penalty = np.linalg.inv(self.prior + regularizer*np.eye(self.prior.shape[0]))
        if self.penalty is None:
            # first time
            self.penalty = penalty
        detnorm = tikutils.determinant_normalizer(penalty) if dodetnorm else 1.0
        return penalty / detnorm

    def normalize_prior(self):
        self.detnorm = tikutils.determinant_normalizer(self.prior)
        self.prior /= self.detnorm

    def normalize_penalty(self):
        self.penalty_detnorm = tikutils.determinant_normalizer(self.penalty)
        self.penalty /= self.penalty_detnorm


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
        # if delays is None:
        #     delays = np.arange(penalty.shape[0])
        # assert (min(delays) >= 0) and (max(delays) < penalty.shape[0])

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
        # default HRF prior is low rank, so need to regularize to invert
        return super(HRFPrior, self).prior2penalty(regularizer=regularizer)



class SmoothnessPrior(PriorFromPenalty):
    '''
    '''
    def __init__(self, delays=range(5), order=2, **kwargs):
        '''
        '''
        penalty = tikutils.difference_operator(order, len(delays))
        CTC = np.dot(penalty, penalty.T)
        super(SmoothnessPrior, self).__init__(CTC, delays=delays, **kwargs)


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
    def __init__(self, *args, **kwargs):
        '''
        '''
        super(GaussianKernelPrior, self).__init__(*args, **kwargs)
