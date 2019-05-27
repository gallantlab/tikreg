'''Handle temporal MVN priors.
'''
import numpy as np
import itertools

from tikreg import BasePrior
import tikreg.utils as tikutils
from tikreg import kernels as tkernel

##############################
# functions
##############################

def get_delays_from_prior(raw_prior, delays):
    '''
    Parameters
    ----------
    raw_prior : 2D np.ndarray (k, k)
        The raw array defined continuously from
        the first delay to the last: `[min(delays),..., max(delays)]`,
        and `max(delays) - min(delays) = k`.
    delays : None or list_like (d)
        When delays is None, the delays of the `raw_prior`
        are assumed to be `[0, 1, ..., k-1]`.
        Otherwise, the delays are specified and are relative to the `raw_prior`.

    Return
    -------
    prior : 2D np.ndarray (

    '''
    if delays is None:
        prior = raw_prior
        delays = np.arange(raw_prior.shape[0])
    else:
        assert min(delays) >= 0 and max(delays) < raw_prior.shape[0]
        delays = np.asarray(delays)
        prior = tikutils.fast_indexing(raw_prior, delays, delays)
    return prior, delays


class TemporalPrior(BasePrior):
    '''Basic temporal MVN prior.
    '''
    def __init__(self, prior, delays=None, hhparams=[0.0], **kwargs):
        '''
        Parameters
        ----------
        prior : 2D np.ndarray (d d)
            Covariance matrix of MVN
        delays : list_like, optional
            Number of delays to use. Only positive delays supported.
            Defaults to size of prior `np.arange(d)`.
        hhparams : list_like, optional
            MVN hyper-prior parameters to evaluate.
            Only used if the prior is constructed from a hyper-prior.
            Defaults to `[0]`.
        '''
        prior, delays = get_delays_from_prior(prior, delays)
        self.delays = delays
        self.ndelays = len(delays)
        self.hhparams = np.atleast_1d(hhparams)

        super(TemporalPrior, self).__init__(prior, **kwargs)

    def set_hhparameters(self, hhparams):
        '''Set the hyper-prior parameters
        '''
        if np.isscalar(hhparams):
            hhparams = np.atleast_1d(hhparams)
        self.hhparams = hhparams

    def get_hhparams(self):
        '''Return the active hyper-prior parameters.
        '''
        return self.hhparams



class CustomPrior(TemporalPrior):
    '''Specify a custom temporal MVN prior.
    '''
    def __init__(self, *args, **kwargs):
        '''
        Parameters
        ----------
        covmat : 2D np.ndarray (d d)
            Covariance matrix of MVN
        delays : list_like, optional
            Number of delays to use. Only positive delays supported.
            Defaults to size of prior `np.arange(d)`.
        hhparams : list_like, optional
            MVN hyper-prior parameters to evaluate.
            Only used if the prior is constructed from a hyper-prior.
            Defaults to `[0]`.

        Examples
        --------
        >>> mat = np.random.randn(5, 5)
        >>> cov = np.dot(mat.T, mat)
        >>> custom_prior = CustomPrior(cov)
        >>> prior_covar = custom_prior.get_prior()
        >>> print(prior_covar.shape)
        (5, 5)
        >>> print(np.round(prior_covar, 2)) # doctest: +SKIP
        [[ 4.98  0.   -0.66 -2.72 -3.89]
         [ 0.   12.28 -5.91  0.49 -4.39]
         [-0.66 -5.91  9.88 -2.15  7.56]
         [-2.72  0.49 -2.15  5.49  3.13]
         [-3.89 -4.39  7.56  3.13 12.95]]
        '''
        super(CustomPrior, self).__init__(*args, **kwargs)


class PriorFromPenalty(TemporalPrior):
    '''Build a prior from a Tikhnov temporal penalty covariance.
    '''
    def __init__(self, penalty, delays=None, wishart=True, **kwargs):
        '''
        Parameters
        ----------
        penalty : 2D np.ndarray (d, d)
            Temporal penalty covariance. If the Tikhhov temporal penalty is C,
            then the temporal penalty covariance is: np.dot(C.T, C)
        delays : list_like, optional
            Number of delays to use. Only positive delays supported.
            Defaults to `np.arange(d)`
        wishart : bool, optional
            Regularize the temporal penalty covariance before taking
            the inverse to compute the prior.
        hhparams : list_like, optional
            Parameters used to regularize the computation of the prior
            from the penalty.
        dodetnorm : bool, optional
            Set the determinat of the initial prior to 1.
        '''
        prior = np.zeros_like(penalty)
        if delays is None:
            delays = np.arange(penalty.shape[0])

        dodetnorm = False
        if 'dodetnorm' in kwargs:
            dodetnorm = kwargs.pop('dodetnorm')

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

        self.dodetnorm = dodetnorm
        # compute prior
        prior = self.get_prior(alpha=1.0)
        if self.dodetnorm:
            prior /= tikutils.determinant_normalizer(prior)
        self.prior = prior


    def set_wishart(self, wishart):
        '''Set the covariance of the hyper-prior regularization.

        wishart : 2D np.ndarray (d,d)
            Defaults to an identity.
        '''
        # TODO: Remove this option.
        if isinstance(wishart, BasePrior):
            wishart = wishart.asarray
        assert np.allclose(wishart.shape, self.penalty.shape)
        self.wishart = wishart

    def get_prior(self, alpha=1.0, hhparam=0.0, dodetnorm=False):
        r'''Convert the penalty to a prior.

        Parameters
        ----------
        alpha : scalar
            The regularization scale on the prior (i.e. lambda)
        hhparam : scalar
            The regularization parameter for the hyper-prior.
            Defaults to `0` for no regularization.
        dodetnorm : bool, optional
            Set the determinant of the MVN prior to 1.

        Returns
        -------
        regularized_prior : 2D np.ndarray (d, d)
            Regularized prior from penalty.
            By default, no regularization is applied.

        Notes
        -----

        .. math::

            \Sigma = (P + \gamma I)^{-1}

        where P is the temporal penalty covariance, :math:`\gamma` is the
        hyper-prior parameter (`hhparam`).
        '''
        # compute prior
        prior = np.linalg.inv(self.penalty + hhparam*self.wishart)

        # select requested delays from prior
        prior, delays = get_delays_from_prior(prior, self.delays)

        if dodetnorm:
            prior /= tikutils.determinant_normalizer(prior)

        return alpha**-2.0 * prior

class SmoothnessPrior(PriorFromPenalty):
    '''Smoothness temporal MVN prior.
    '''
    def __init__(self, delays=range(5), order=2, wishart=True, **kwargs):
        '''
        Parameters
        ----------
        delays : list_like, optional
            Number of delays to use. Only positive delays supported.
            Defaults to: `[0,1,2,3,4]`.
        order : even integer, optional
            The order of the difference operator that defines the smoothness prior.
            Defaults to `2` for a second-order difference operator.
        wishart : bool, optional
            Regularize the temporal penalty covariance before taking
            the inverse to compute the prior.
        hhparams : list_like, optional
            Parameters used to regularize the computation of the prior
            from the penalty.
        dodetnorm : bool, optional
            Set the determinat of the initial prior to 1.

        Examples
        --------
        >>> smoothness_prior = SmoothnessPrior(delays=np.arange(5))
        >>> print(smoothness_prior.penalty)
        [[ 5. -4.  1.  0.  0.]
         [-4.  6. -4.  1.  0.]
         [ 1. -4.  6. -4.  1.]
         [ 0.  1. -4.  6. -4.]
         [ 0.  0.  1. -4.  5.]]
        >>> prior_covar = smoothness_prior.get_prior()
        >>> print(prior_covar.shape)
        (5, 5)
        >>> print(np.round(prior_covar, 2))
        [[1.53 2.22 2.25 1.78 0.97]
         [2.22 3.78 4.   3.22 1.78]
         [2.25 4.   4.75 4.   2.25]
         [1.78 3.22 4.   3.78 2.22]
         [0.97 1.78 2.25 2.22 1.53]]
        '''
        maxdelay = max(delays)+1
        penalty = tikutils.difference_operator(order, maxdelay)
        CTC = np.dot(penalty, penalty.T)
        super(SmoothnessPrior, self).__init__(CTC, delays=delays, wishart=wishart, **kwargs)



class SphericalPrior(TemporalPrior):
    '''Equivalent to ridge regression.
    '''
    def __init__(self, delays=range(5), **kwargs):
        '''
        Parameters
        ----------
        delays : list_like, optional
            Number of delays to use. Only positive delays supported.
            Defaults to: `[0,1,2,3,4]`.

        Examples
        --------
        >>> spherical_prior = SphericalPrior(delays=np.arange(5))
        >>> prior_covar = spherical_prior.get_prior()
        >>> print(prior_covar.shape)
        (5, 5)
        >>> print(np.round(prior_covar, 2))
        [[1. 0. 0. 0. 0.]
         [0. 1. 0. 0. 0.]
         [0. 0. 1. 0. 0.]
         [0. 0. 0. 1. 0.]
         [0. 0. 0. 0. 1.]]
        '''
        raw_prior = np.eye(max(delays)+1)
        super(SphericalPrior, self).__init__(raw_prior, delays=delays, **kwargs)


class HRFPrior(TemporalPrior):
    '''Haemodynamic response function prior
    '''
    def __init__(self, delays=None, dt=2.0, duration=20, **kwargs):
        '''Generates a discrete sampling of the HRF.

        The time samples are generated as:
        `time_samples = np.arange(0, duration, dt)`

        And the corresponding delays:
        `delays = np.arange(len(time_samples))`

        Parameters
        ----------
        dt : float
            Sampling rate of the BOLD signal (TR).
        duration : int
            Duration in seconds over which to sample the HRF.
        delays : list_like, optional
            Defaults to all delays spanning the requested duration.

        Examples
        --------
        >>> hrf_prior = HRFPrior(delays=range(5), dt=2.0, duration=20)
        >>> print(hrf_prior.delays)
        [0 1 2 3 4]
        >>> prior_covar = hrf_prior.get_prior()
        >>> print(prior_covar.shape)
        (5, 5)
        >>> print(np.round(prior_covar, 2))
        [[0.   0.   0.   0.   0.  ]
         [0.   0.44 0.55 0.22 0.18]
         [0.   0.55 1.28 0.97 0.54]
         [0.   0.22 0.97 1.   0.56]
         [0.   0.18 0.54 0.56 0.36]]
        '''
        if delays is not None:
            if len(delays) == 1 and delays[0] == 0:
                raise ValueError('Using delay zero by itself is not allowed')
            assert min(delays)*dt >= 0
            assert max(delays)*dt <= duration - dt
        H = tikutils.hrf_default_basis(dt=dt, duration=duration)
        raw_prior = np.dot(H, H.T).astype(np.float64)
        if 'hhparams' in kwargs:
            raise ValueError('HRFPrior does not accept a hyper-prior')
        super(HRFPrior, self).__init__(raw_prior, delays=delays, **kwargs)


class GaussianKernelPrior(TemporalPrior):
    '''Gaussian kernel temporal prior (a.k.a RBF kernel).
    '''
    def __init__(self, delays=None, sigma=1.0,  **kwargs):
        '''Construct a temporal MVN prior from a Gaussian (RBF) kernel

        Parameters
        ----------
        delays : list_like, optional
            Number of delays to use. Only positive delays supported.
            Defaults to: `[0,1,2,3,4]`.
        sigma : scalar, optional
            The width of the Gaussian.
            Defaults to `1`.
        hhparams : list_like, optional
            Parameters used to regularize the computation of the prior
            from the penalty.
            Defaults to sigma value used on instantiation: `[sigma]`.
        dodetnorm : bool, optional
            Set the determinat of the initial prior to 1.

        References
        ----------
        For a great description of Gaussian Processes and their
        relationship to RBF Kernels:
        https://distill.pub/2019/visual-exploration-gaussian-processes/

        Examples
        --------
        >>> gaussian_prior = GaussianKernelPrior(delays=np.arange(5))
        >>> prior_covar = gaussian_prior.get_prior()
        >>> print(prior_covar.shape)
        (5, 5)
        >>> print(np.round(prior_covar, 2))
        [[1.   0.61 0.14 0.01 0.  ]
         [0.61 1.   0.61 0.14 0.01]
         [0.14 0.61 1.   0.61 0.14]
         [0.01 0.14 0.61 1.   0.61]
         [0.   0.01 0.14 0.61 1.  ]]
        '''
        full_delays = np.arange(max(delays)+1)
        self.kernel_object = tkernel.lazy_kernel(full_delays[...,None],
                                                 kernel_type='gaussian')

        self.kernel_object.update(sigma)
        prior = self.kernel_object.kernel

        if 'hhparams' not in kwargs:
            kwargs['hhparams'] = [sigma]
        super(GaussianKernelPrior, self).__init__(prior, delays=delays, **kwargs)


    def get_prior(self, alpha=1.0, hhparam=1.0, dodetnorm=False):
        '''Gaussian/RBF kernel as a temporal prior covariance matrix.

        Parameters
        ----------
        alpha : scalar, optional
            The regularization scale (i.e. lambda).
        hhparam : scalar, optional
            The width of the Gaussian
            Defaults to `1`.
        dodetnorm : bool, optional
            Set the determinat of the prior covariance matrix to 1.
        '''
        self.kernel_object.update(hhparam)

        prior = self.kernel_object.kernel
        # select requested delays from prior
        prior, delays = get_delays_from_prior(prior, self.delays)

        if dodetnorm:
            prior /= tikutils.determinant_normalizer(prior)

        return alpha**-2.0 * prior
