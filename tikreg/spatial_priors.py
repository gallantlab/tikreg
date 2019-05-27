'''Handle feature MVN priors.
'''
import numpy as np

from tikreg import BasePrior
import tikreg.utils as tikutils


class CustomPrior(BasePrior):
    '''Specify a custom feature prior
    '''
    def __init__(self, *args, **kwargs):
        '''
        Parameters
        ----------
        covmat : 2D np.ndarray (p, p)
            Covariance matrix of MVN.

        Examples
        --------
        >>> mat = np.random.randn(5, 5)
        >>> cov = np.dot(mat.T, mat)
        >>> custom_prior = CustomPrior(cov)
        >>> print(custom_prior.asarray.shape)
        (5, 5)
        >>> prior_lambda_one = custom_prior.get_prior(1.0)
        >>> print(np.round(prior_lambda_one, 2)) # doctest: +SKIP
        [[ 3.27 -2.02  2.18  0.05 -5.37]
         [-2.02  6.59  0.47 -1.2   3.95]
         [ 2.18  0.47  3.75  0.35 -2.52]
         [ 0.05 -1.2   0.35  5.9   1.31]
         [-5.37  3.95 -2.52  1.31 10.86]]
        >>> prior_lambda_two = custom_prior.get_prior(2.0)
        >>> print(np.round(prior_lambda_two, 2)) # doctest: +SKIP
        [[ 0.82 -0.51  0.54  0.01 -1.34]
         [-0.51  1.65  0.12 -0.3   0.99]
         [ 0.54  0.12  0.94  0.09 -0.63]
         [ 0.01 -0.3   0.09  1.48  0.33]
         [-1.34  0.99 -0.63  0.33  2.72]]
        '''
        super(CustomPrior, self).__init__(*args, **kwargs)


class SphericalPrior(BasePrior):
    '''Equivalent to ridge regression.
    '''
    def __init__(self, feature_space, **kwargs):
        r'''Create a spherical MVN prior for a feature space.

        Parameters
        ----------
        feature_space : 2D np.ndarray (n, p) or int
            Matrix containing features (one column per feature).
            Or a scalar corresponding to
            the number of features (i.e. p).

        Examples
        --------
        >>> spherical_prior = SphericalPrior(5)
        >>> print(spherical_prior.asarray.shape)
        (5, 5)
        >>> print(np.round(spherical_prior.asarray, 2))
        [[1. 0. 0. 0. 0.]
         [0. 1. 0. 0. 0.]
         [0. 0. 1. 0. 0.]
         [0. 0. 0. 1. 0.]
         [0. 0. 0. 0. 1.]]
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
        '''Apply regularization to prior

        Parameters
        ----------
        alpha : scalar
            Regularization parameter (lambda).

        Returns
        -------
        ridge_matrix : 2D np.ndarray (p, p)
            Scaled ridge matrix (i.e. :math:`\lambda^{-2} I_p`)

        Examples
        --------
        >>> spherical_prior = SphericalPrior(5)
        >>> prior_covar = spherical_prior.get_prior(2.0)
        >>> print(np.round(prior_covar, 2))
        [[0.25 0.   0.   0.   0.  ]
         [0.   0.25 0.   0.   0.  ]
         [0.   0.   0.25 0.   0.  ]
         [0.   0.   0.   0.25 0.  ]
         [0.   0.   0.   0.   0.25]]
        '''
        return alpha**-2.0 * self.asarray


class PriorFromPenalty(BasePrior):
    '''Build a prior from a Tikhnov feature penalty covariance.
    '''
    def __init__(self, penalty, wishart=True, **kwargs):
        '''
        Parameters
        ----------
        penalty : 2D np.ndarray (p, p)
            Penalty covariance. If the Tikhhov penalty is C,
            then the penalty covariance is: np.dot(C.T, C)
        wishart : bool, optional
            Regularize the penalty covariance before taking
            the inverse to compute the prior.
        hhparams : list_like, optional
            Parameters used to regularize the computation of the prior
            from the penalty.
        dodetnorm : bool, optional
            Set the determinat of the initial prior to 1.
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
        r'''Convert the penalty to a prior.

        Parameters
        ----------
        alpha : scalar
            The regularization scale on the prior (i.e. lambda)
        wishart_lambda : scalar
            The regularization parameter for the hyper-prior.
            Defaults to `0` for no regularization.
        dodetnorm : bool, optional
            Set the determinant of the MVN prior to 1.

        Returns
        -------
        regularized_prior : 2D np.ndarray (p, p)
            Regularized prior from penalty.
            By default, no regularization is applied.

        Notes
        -----

        .. math::

            \Sigma = (P + \gamma I)^{-1}

        where P is the penalty covariance, :math:`\gamma` is the
        hyper-prior parameter (`hhparam`).
        '''
        # compute prior
        prior = np.linalg.inv(self.penalty + wishart_lambda*self.wishart)

        if dodetnorm:
            prior /= tikutils.determinant_normalizer(prior)

        return alpha**-2.0 * prior
