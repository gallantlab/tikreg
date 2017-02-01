import numpy as np
import tikypy.utils as tikutils


##############################
# Classes
##############################

class BasePrior(object):
    '''Base class for priors
    '''
    def __init__(self, prior, dodetnorm=False, hyperparameters=[1.0]):
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
        self.hyperparameters = np.atleast_1d(hyperparameters)

    @property
    def asarray(self):
        return self.prior

    def prior2penalty(self, dodetnorm=False):
        penalty = np.linalg.inv(self.prior) # must be invertible
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

    def get_prior(self, alpha=1.0, dodetnorm=False, **kwargs):
        if dodetnorm:
            prior = self.asarray / tikutils.determinant_normalizer(self.asarray)
        else:
            prior = self.asarray
        return alpha**-2 * prior

    def set_hyperparameters(self, hyperparameters):
        if np.isscalar(hyperparameters):
            hyperparameters = np.atleast_1d(hyperparameters)
        self.hyperparameters = hyperparameters

    def get_hyperparameters(self):
        return self.hyperparameters
