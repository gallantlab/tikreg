import numpy as np

from stevem import BasePrior
import stevem.utils as tikutils


class CustomPrior(BasePrior):
    '''Specify a custom prior
    '''
    def __init__(self, *args, **kwargs):
        '''
        '''
        super(CustomPrior, self).__init__(*args, **kwargs)


class SphericalPrior(BasePrior):
    '''Equivalent to ridge.
    '''
    def __init__(self, feature_space, **kwargs):
        '''
        '''
        if isinstance(feature_space, np.ndarray):
            nfeatures = feature_space.shape[-1]
        elif np.isscalar(feature_space):
            nfeatures = feature_space
        elif isinstance(feature_space, tuple):
            # last dimension is features
            nfeatures = feature_space[1]
        else:
            raise ValueError('%s is not allowed'%type(feature_space))

        raw_prior = np.eye(nfeatures)
        super(SphericalPrior, self).__init__(raw_prior, **kwargs)

    def get_prior(self, alpha=1.0):
        return alpha**-2.0 * self.asarray


class PriorFromPenalty(BasePrior):
    '''
    '''
    def __init__(self, penalty, wishart=True, **kwargs):
        '''
        '''
        prior = np.zeros_like(penalty)
        super(PriorFromPenalty, self).__init__(prior, **kwargs)

        # overwrite penalty after init
        self.penalty = penalty.copy()
        self.wishart = np.eye(penalty.shape[0])

    def prior2penalty(self, regularizer=0.0, dodetnorm=True):
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

    def set_wishart(self, wishart_prior):
        '''
        '''
        if isinstance(wishart_prior, BasePrior):
            wishart_prior = wishart_prior.asarray
        assert np.allclose(wishart_prior.shape, self.penalty.shape)
        self.wishart = wishart_prior

    def get_prior(self, alpha=1.0, wishart_lambda=0.0, dodetnorm=False):
        '''
        '''
        # compute prior
        prior = np.linalg.inv(self.penalty + wishart_lambda*self.wishart)

        if dodetnorm:
            prior /= tikutils.determinant_normalizer(prior)

        return alpha**-2.0 * prior
