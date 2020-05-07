'''
'''
import numpy as np
from scipy.stats import zscore

from scipy.linalg import toeplitz

try:
    from scipy.misc import comb
except ImportError:
    # scipy >= 1.2 for PY3
    from scipy.special import comb

try:
    reduce
except NameError:
    from functools import reduce


def isdiag(mat):
    '''Determine whether matrix is diagonal.

    Parameters
    ----------
    mat : 2D np.ndarray (n, n)

    Returns
    -------
    ans : bool
        True if matrix is diagonal
    '''
    if mat.ndim != 2:
        return False

    if mat.shape[0] != mat.shape[1]:
        return False

    if not np.allclose(mat[-1,:-1], 0):
        # last row has non-zero elements
        return False

    # zeros in lower triangular
    idxs = np.tril_indices_from(mat, -1)
    if not np.allclose(mat[idxs], 0):
        return False

    # zeros in upper triangular
    idxs = np.triu_indices_from(mat, 1)
    if not np.allclose(mat[idxs], 0):
        return False

    return True


def SVD(X, **kwargs):
    '''Robust SVD decomposition.

    First uses scipy.linalg.svd by default.
    If the SVD does not converge, it will
    use a slower more robust SVD algorithm (DGESVD).

    See `scipy.linalg.svd` for full documentation.

    Parameters
    ----------
    X : 2D np.ndarray (n, m)
        Matrix to decompose

    full_matrices : bool, optional
        Faster performance when True.
        Defaults to False (numpy/scipy convention).

    Returns
    -------
    U, S, VT : tuple of np.ndarrays
        SVD decomposition of the matrix
    '''
    import scipy.linalg as LA

    if 'full_matrices' not in kwargs:
        kwargs['full_matrices'] = False

    try:
        O = LA.svd(X, **kwargs)
    except LA.LinAlgError as e:
        from warnings import warn
        warn('%s... trying slow SVD' % e)
        from scipy.linalg import lapack
        u,s,vt,info = lapack.dgesvd(X, **kwargs)
        O = u,s,vt
    return O


def difference_operator(order, nobs):
    '''Get a finite difference operator matrix of size `nobs`.

    Parameters
    ----------
    order : int, odd
        The order of the discrete difference (e.g. 2nd order)
    nobs : int
        The size of the output matrix

    Returns
    -------
    mat : 2D np.ndarray (nobs, nobs)
        Discrete difference operator
    '''
    if nobs == 1:
        return np.asarray([[1]])

    depth = order + 1
    # pascal triangle row
    kernel = np.asarray([comb(depth-1, idx) for idx in range(depth)])
    sign = (-1)**np.arange(len(kernel))
    kernel *= sign
    vec = np.zeros(nobs)
    if order % 2 == 0:
        lkern = int(len(kernel)/2)
        vec[:len(kernel)-lkern] = kernel[lkern:]
        convmat = toeplitz(vec, np.zeros(nobs))
        convmat += np.tril(convmat, -1).T
    elif order == 1:
        vec[:len(kernel)] = kernel
        convmat = toeplitz(vec)
    else:
        raise NotImplementedError
    return convmat


def hrf_default_basis(dt=2.0, duration=32):
    '''Hemodynamic response function basis set.

    Wrapper to `hrf_estimation` package.

    Parameters
    ----------
    dt : float, optional
        Temporal sampling rate in seconds
        Defaults to 2.0 (i.e TR=2.0[secs])
    duration : int, optional
        Period over which to sample the HRF.
        Defaults to 32 [seconds].

    Returns
    --------
    hrf_basis : 2D np.ndarray (duration/dt, 3)
        HRF basis set sampled over the specified
        time period at the sampling rate requested.
    '''
    try:
        import hrf_estimation as he
    except ImportError:
        txt = '''You need to install `hrf_estimation` package: "pip install hrf_estimation"'''
        raise ImportError(txt)

    time = np.arange(0, duration, dt)
    h1 = he.hrf.spm_hrf_compat(time)
    h2 = he.hrf.dspmt(time)
    h3 = he.hrf.ddspmt(time)

    hrf_basis = np.c_[h1, h2, h3]
    return hrf_basis


def fast_indexing(a, rows, cols=None):
    '''Extract row and column entries from a 2D np.ndarray.

    Much faster and memory efficient than fancy indexing for
    rows and cols from a matrix. Slightly faster for row indexing.

    Parameters
    ----------
    a : 2D np.ndarray (n, m)
        A matrix
    rows : 1D np.ndarray (k)
        Row indices
    cols : 1D np.ndarray (l)
        Column indices

    Returns
    -------
    b : 2D np.ndarray (k, l)
        Subset of the matrix `a`
    '''
    # Cannot remember origin of this trick.
    if cols is None:
        cols = np.arange(a.shape[-1])
    idx = rows.reshape(-1,1)*a.shape[1] + cols
    return a.take(idx)


def determinant_normalizer(mat, thresh=1e-08):
    '''Compute scalar to normalize covariance matrix determinant to 1.

    Uses the pseudo-determinant for numerical robustness.
    This is implemented by ignoring the smallest eigenvalues.

    Parameters
    ----------
    mat : 2D np.ndarray (n, n)
        Covariance matrix
    thresh : float_like, optional
        Threshold for the smallest eigenvalues to use.
        Only eigenvalues larger than this are used to
        compute the pseudo-determinant.

    Returns
    -------
    scale : float_like
        Scalar such that dividing `mat` by this scalar
        sets the matrix determinant to 1

    Notes
    -----
    The determinant can be thought of as the variance of
    a covariance matrix in high-dimensions. Analogous to
    standardizing data to have unit variance (e.g. z-scoring),
    one can standardize a matrix to have a determinant of 1.
    Setting the determinant to 1 allows the covariance structure
    to vary while keeping the 'variance' constant.

    This function is useful when sampling covariance matrices
    from Wishart distributions or when the generating covariance
    matrix function has hyper-parameters of its own.

    Examples
    --------
    >>> mat = np.random.randn(20,20)
    >>> cov = np.dot(mat.T, mat)
    >>> assert np.allclose(np.linalg.det(cov), 1.0) is False
    >>> scale = determinant_normalizer(cov)
    >>> assert np.allclose(np.linalg.det(cov/scale), 1.0)
    '''
    evals = np.linalg.eigvalsh(mat)
    gdx = evals > thresh
    det = np.prod(evals[gdx])
    scale = det**(1./gdx.sum())
    if np.isinf(scale) or np.isnan(scale) or scale==0:
        scale = 1.0
    return scale

def delay2slice(delay):
    '''Get a slicer for an array at a desired delay (e.g. a[delay2slice(3)]).

    Parameters
    ----------
    delay : int
        Delay number (in units of samples)

    Returns
    -------
    slicer : slice_object
        Object used to get data from an array
        at the specified delay.

    See also
    --------
    delay_signal : Explicitly delay data by creating a copy.

    Examples
    --------
    >>> # e.g. 5 timepoints and 3 features
    >>> mat = np.arange(5*3).reshape(5,3)
    >>> print(mat)
    [[ 0  1  2]
     [ 3  4  5]
     [ 6  7  8]
     [ 9 10 11]
     [12 13 14]]
    >>> slicer = delay2slice(2) # delay by 2 samples
    >>> # starts at `delay` and has (n - delay) samples
    >>> print(mat[slicer])
    [[0 1 2]
     [3 4 5]
     [6 7 8]]
    >>> # starts at 0th sample and shifts data explicitly
    >>> delayed_mat = delay_signal(mat, [2])
    >>> print(delayed_mat)
    [[0 0 0]
     [0 0 0]
     [0 1 2]
     [3 4 5]
     [6 7 8]]
    '''
    if delay > 0:
        ii = slice(None, -delay)
    elif delay == 0:
        ii = slice(None, None)
    elif delay < 0:
        raise ValueError('No negative delays')
    return ii


def generate_data(n=100, p=10, v=2,
                  noise=1.0,
                  testsize=None,
                  dozscore=False,
                  feature_sparsity=0.0):
    '''Get some B,X,Y data generated from gaussian (0,1).

    Parameters
    ----------
    n, p, v : int
        Number of samples (n), features (p) and responses (v).
    noise : float
        Noise level
    testsize : int, optional
        Samples in the test set. Defaults None (i.e. no test set).
    dozscore : bool
        Standardize the features are responses to zero-mean and unit-norm.
    feature_sparsity : float between 0-1
        Number of irrelevant features as percentage of total.

    Returns
    -------
    B : 2D np.ndarray (p, v)
        True feature weights
    (Xtrain, Xtest): two-tuple of 2D np.ndarrays ((n,p), (testsize, p))
        Feature matrix for training and test set.
    (Ytrain, Ytest): two-tuple of 2D np.ndarrays ((n,v), (testsize, v))
        Response matrix for training and test set.
        Ytest is optional and contains no noise.

    Examples
    --------
    >>> B, (Xtrain, Xtest), (Ytrain, Ytest) = generate_data(n=100, p=10, v=2,testsize=20)
    >>> B.shape, (Xtrain.shape, Xtest.shape), (Ytrain.shape, Ytest.shape)
    ((10, 2), ((100, 10), (20, 10)), ((100, 2), (20, 2)))
    >>> B, X, Y = generate_data(n=100, p=10, v=2,noise=1.0,testsize=0) # No test data
    >>> B.shape, X.shape, Y.shape
    ((10, 2), (100, 10), (100, 2))
    '''
    if testsize is None:
        testsize = 0

    X = np.random.randn(n, p)
    B = np.random.randn(p, v)
    if dozscore: X = zscore(X)
    nzeros = int(feature_sparsity*p)
    if nzeros > 0:
        B[-nzeros:,:] = 0

    Y = np.dot(X, B)
    if dozscore:
        Y = zscore(Y)

    E = np.random.randn(*Y.shape)*noise
    Y += E
    if dozscore:
        Y = zscore(Y)

    if testsize:
        Xval = np.random.randn(testsize, p)
        if dozscore:
            Xval = zscore(Xval)
        Yval = np.dot(Xval, B)
        if dozscore: Yval = zscore(Yval)
        return B, (X,Xval), (Y, Yval)

    return B, X, Y


def mult_diag(d, mat, left=True):
    """Efficient multiply a full matrix by a diagonal matrix.

    This function should always be faster than dot.

    Parameters
    ----------
    d : 1D np.ndarray (n)
        Contains the diagonal elements.
    mat : 2D np.ndarray (n,n)
        Contains the matrix

    Returns
    --------
    res (n, n)
        Result of multiplying the matrices

    Notes
    ------
    This code by:
    Pietro Berkes berkes@gatsby.ucl.ac...
    Mon Mar 26 11:55:47 CDT 2007
    http://mail.scipy.org/pipermail/numpy-discussion/2007-March/026807.html

    Examples
    --------
    >>> mat = np.random.randn(20,20)
    >>> d = np.random.randn(20)
    >>> assert np.allclose(mult_diag(d, mat, left=True), np.dot(np.diag(d), mat))
    >>> assert np.allclose(mult_diag(d, mat, left=False), np.dot(mat, np.diag(d)))
    """
    if left:
        return (d*mat.T).T
    else:
        return d*mat


def noise_ceiling_correction(repeats, yhat, dozscore=True):
    """Noise ceiling corrected correlation coefficient.

    Correlation coefficient estimate that better reflects
    the error in the predictions that is due to the inaccuracy
    of the model, rather than the noise intrinsic in the responses.
    This is achieved by removing the non-stationary part of the
    noise in the measured signals across multiple repetitions of the
    same stimulus/task/conditions.

    Parameters
    ----------
    repeats : np.ndarray (nreps, ntpts, nunits)
        Timecourses for each repeat.
        Each repeat is `ntpts` long.
    yhat : np.ndarray (ntpts, nunits)
        Predicted timecourse for each unit measured (e.g. voxels, neurons, etc).
    dozscore : bool
        This implementation only works correctly if the
        `repeats` and `yhat` timecourses are z-scored.
        If these are already z-scored, set to False.

    Returns
    -------
    r_ncc : np.ndarray (nunits)
        The noise ceiling corrected correlation coefficient
        for each of the units. One may square this result
        (while keeping the sign) to obtain the amount of
        explainable variance explained.

    Notes
    -----
    Repeats are used to compute the amount of explainable variance
    in each one of the units  (e.g. voxels, neurons, etc.). This is
    equivalent to estimating the adjusted :math:`R^2` of an OLS model
    that predicts each individual repeat timecourse with
    the mean timecourse computed across repetitions.
    This process is performed individually for each unit.

    The mean timecouse of each unit is computed. The product between
    the predicted responses and the mean timecourse is then computed.
    This value is then divided by the amount of explainable variance.

    :math:`r_{ncc}` is misbehaved if :math:`R^2` is very low.

    References
    ----------
    Schoppe, et al. (2016), Hsu, et al. (2004), David, et al. (2005).

    Examples
    --------
    First, simulate some repeated data for 50 units (e.g. voxels, neurons).

    >>> nreps, ntpts, nunits, noise = 10, 100, 50, 2.0
    >>> _, _, Y = generate_data(n=ntpts, testsize=0, v=nunits, noise=0.0, dozscore=True)
    >>> repeats = np.asarray([Y for i in range(nreps)])
    >>> # Add i.i.d. gaussian noise to each copy of the data
    >>> repeats += np.random.randn(nreps, ntpts, nunits)*noise
    >>> print(repeats.shape)
    (10, 100, 50)

    Compute the noise ceiling corrected correlation coefficient using random predictions.
    Because the predictions are unrelated to the data, we expected a value of 0.

    >>> mean_nccr = noise_ceiling_correction(repeats, np.random.randn(ntpts, nunits)).mean()
    >>> mean_nccR2 = (mean_nccr**2)*np.sign(mean_nccr)
    >>> # As ntpts -> inf, this converges to 0 because predictions are random
    >>> assert np.allclose(round(mean_nccR2, 2), 0)

    Next, we produce more accurate predictions.

    >>> Yhat = Y + np.random.randn(ntpts, nunits)*0.5 # little noise
    >>> # raw correlation reflects both noise in signals (2.0) and predictions (0.5)
    >>> raw_perf = columnwise_correlation(repeats.mean(0), Yhat).mean()
    >>> print(raw_perf)         # doctest: +SKIP
    0.7534268012539665

    The raw correlation coefficient computed reflects the amount of noise in the signals (2.0)
    and the amount of noise in our model (0.5). The noise ceiling corrected correlation coefficient
    can be used to obtain an estimate that more closely reflects the error due to predictions alone.

    >>> accurate_mean_nccr = noise_ceiling_correction(repeats, Yhat).mean()
    >>> print(accurate_mean_nccr) # doctest: +SKIP
    0.8954483753448146
    >>> # As nreps, ntpts -> inf, the nccr is determined by the prediction error alone (0.5)
    >>> # Analytically, the error in the predictions is 0.5 so the expected correlation is ~0.89
    >>> print(round(analytic_expected_correlation(0.5), 6))
    0.894427
    >>> assert np.allclose(accurate_mean_nccr, analytic_expected_correlation(0.5), rtol=1e-01) # Small simulation
    """
    ntpts = repeats.shape[1]
    reps = zscore(repeats, 1) if dozscore else repeats
    yhat = zscore(yhat, 0) if dozscore else yhat

    R2 = explainable_variance(reps, ncorrection=True, dozscore=False)
    ymean = reps.mean(0)
    ycov = (ymean*yhat).sum(0)/(ntpts - 1) # sample covariance
    return ycov/np.sqrt(R2)


def explainable_variance(repeats, ncorrection=True, dozscore=True):
    '''Compute the explainable variance in the recorded signals.

    Parameters
    ----------
    repeats : np.ndarray (nreps, ntpts, nsignals)
        The timecourses for each stimulus repetition.
        Each of repeat is `ntpts` long.
    ncorrection : bool, optional
        Bias correction for number of repeats.
        Equivalent to computing the adjusted R^2.
    dozscore : bool, optional
        This implementation only works with z-scored repeats. If
        the each repetition is already z-scored, set to False.

    Returns
    -------
    EV : np.ndarray (nsignals)
        The explainable variance computed across repeats.
        Equivalently, the adjusted :math:`R^2` value.

    References
    ----------
    Schoppe, et al. (2016), Hsu, et al. (2004), David, et al. (2005).

    Notes
    -----
    Explainable variance can be interpreted as the :math:`R^2` of a model
    that predicts each repetition with the mean across repetitions.

    Examples
    --------

    First, simulate some repeated data for 50 units (e.g. voxels, neurons).

    >>> nreps, ntpts, nunits, noise = 10, 100, 50, 2.0
    >>> _, _, Y = generate_data(n=ntpts, testsize=0, v=nunits, noise=0.0)
    >>> repeats = np.asarray([Y for i in range(nreps)])
    >>> # Add i.i.d. gaussian noise to each copy of the data
    >>> repeats += np.random.randn(nreps, ntpts, nunits)*noise
    >>> print(repeats.shape)
    (10, 100, 50)

    The repeats can be used to compute the explainable variance
    for each simulated unit.

    >>> EV = explainable_variance(repeats)
    >>> EV.shape
    (50,)
    >>> print(EV.mean()) # doctest: +SKIP
    0.20099454817453574
    >>> analytic_R2 = analytic_expected_correlation(2.0)**2
    >>> print(round(analytic_R2, 6))
    0.2
    '''
    repeats = zscore(repeats, 1) if dozscore else repeats
    residual = repeats - repeats.mean(0)
    residualvar = np.mean(residual.var(1), 0)
    ev = 1 - residualvar

    if ncorrection:
        ev = ev - ((1 - ev) / np.float((repeats.shape[0] - 1)))
    return ev


def absmax(arr):
    '''Find the absolute maximum of an array.

    This is somewhat more efficient than e.g. np.nanmax(np.abs(arr))

    Parameter
    ----------
    arr : np.ndarray

    Returns
    -------
    maxval : scalar
        The absolute maximum value in the array

    Examples
    --------
    >>> arr = np.random.randn(20,20)
    >>> maxval = absmax(arr)
    >>> direct_maxval = np.nanmax(np.abs(arr))
    >>> assert np.allclose(maxval, direct_maxval)
    '''
    return max(abs(np.nanmin(arr)), abs(np.nanmax(arr)))


def delay_signal(mat, delays=[0, 1, 2, 3], fill=0):
    r'''Create a temporally shifted version of the data

    Parameters
    ----------
    mat : 2D np.ndarray (n, p)
        The first dimension is time.
    delays : list_like (d,)
        Amount by which to shift the signals in time.
    fill : scalar, optional
        Value to fill the empty values with.

    Returns
    -------
    delayed_mat : 2D np.ndarray (n, p*d)
        The data delayed at the requested lags.
        The resulting array is larger than the original
        whenever more than one delay is requested.

    Notes
    -----
    The data is delayed such that each `p` columns correspond
    to one delay. The order of the delays is preserved.

    .. math::

        X = \left[X_{\delta \left(d_1 \right)}, X_{\delta \left(d_2 \right)}, \ldots, X_{\delta \left(d_D \right)}\right]

    where :math:`X_{\delta \left(j \right)}` corresponds to the original data delayed by `j` samples.

    Examples
    --------
    >>> mat = np.arange(5*3).reshape(5,3)
    >>> print(mat)
    [[ 0  1  2]
     [ 3  4  5]
     [ 6  7  8]
     [ 9 10 11]
     [12 13 14]]
    >>> delayed_mat = delay_signal(mat, [0,1,2])
    >>> print(delayed_mat.shape)
    (5, 9)
    >>> print(delayed_mat) #
    [[ 0  1  2  0  0  0  0  0  0]
     [ 3  4  5  0  1  2  0  0  0]
     [ 6  7  8  3  4  5  0  1  2]
     [ 9 10 11  6  7  8  3  4  5]
     [12 13 14  9 10 11  6  7  8]]
    >>> delayed_mat_neg = delay_signal(mat, [-1,0,1]) # negative delays
    >>> print(delayed_mat_neg)
    [[ 3  4  5  0  1  2  0  0  0]
     [ 6  7  8  3  4  5  0  1  2]
     [ 9 10 11  6  7  8  3  4  5]
     [12 13 14  9 10 11  6  7  8]
     [ 0  0  0 12 13 14  9 10 11]]
    '''
    if mat.ndim == 1:
        mat = mat[...,None]
    n, p = mat.shape
    out = np.ones((n, p*len(delays)), dtype=mat.dtype)*fill

    for ddx, num in enumerate(delays):
        beg, end = ddx*p, (ddx+1)*p
        if num == 0:
            out[:, beg:end] = mat
        elif num > 0:
            out[num:, beg:end] = mat[:-num]
        elif num < 0:
            out[:num, beg:end] = mat[abs(num):]
    return out


def columnwise_rsquared(ypred, y, **kwargs):
    '''Compute the R2

    Predictions and actual responses are matrices whose columns
    correspond to the units sampled (e.g. voxels, neurons, etc).

    Parameters
    ----------
    ypred : 2D np.ndarray (n, v)
        Matrix of predicted responses. The first dimension is samples.
        The second dimension is corresponds to the measured signals.
    y : 2D np.ndarray (n, v)
        Matrix of actual responses.
    kwargs : optional
        These are ignored.

    Returns
    -------
    R2 : 1D np.ndarray (v,)
        The coefficient of determination (R2) for each
        of the `v` responses measured

    References
    ----------
    https://en.wikipedia.org/wiki/Coefficient_of_determination#Extensions
    '''
    assert ypred.shape == y.shape # dimensions must match
    return 1 - (y - ypred).var(axis=0)/y.var(axis=0)


def columnwise_correlation(ypred, y, zscorea=True, zscoreb=True, axis=0):
    r'''Compute the correlation coefficients

    Predictions and actual responses are matrices whose columns
    correspond to the units sampled (e.g. voxels, neurons, etc).

    Parameters
    ----------
    ypred : 2D np.ndarray (n, v)
        Matrix of predicted responses. The first dimension is samples.
        The second dimension is corresponds to the measured signals.
    y : 2D np.ndarray (n, v)
        Matrix of actual responses.
    zscorea, zscoreb : bool, optional
        Defaults to True.
        This implementation works by first z-scoring
        the actual and predicted responses. If they are
        already z-scored, then the computation is made
        faster by setting these values to False.
    axis : int, optional
        Dimension corresponding to samples over which to correlate.
        Defaults to 0.

    Returns
    -------
    corr : 1D np.ndarray (v,)
        The correlation coefficient (R2) for each
        of the `v` responses.

    Examples
    --------
    >>> x = np.random.randn(100,2)
    >>> y = np.random.randn(100,2)
    >>> cc = columnwise_correlation(x, y)
    >>> cc.shape
    (2,)
    >>> c1 = np.corrcoef(x[:,0], y[:,0])[0,1]
    >>> c2 = np.corrcoef(x[:,1], y[:,1])[0,1]
    >>> assert np.allclose(cc, np.r_[c1, c2])

    Notes
    -----
    Recall that the correlation cofficient is defined as

    .. math::
       \rho_{x, y} = \frac{cov(X,Y)}{var(x)var(y)}

    Since it is scale invariant, we can zscore and get the same

    .. math::
       \rho_{x, y} = \rho_{zscore(x), zscore(y)} = \frac{cov(X,Y)}{1*1} =
       \frac{1}{N}\frac{\sum_i^n \left(x_i - 0 \right) \left(y_i - 0 \right)}{1*1} =
       \frac{1}{N}\sum_i^n \left(x_i * y_i \right)
    '''
    if zscorea:
        y = zscore(y, axis=axis)
    if zscoreb:
        ypred = zscore(ypred, axis=axis)
    corr = (y * ypred).mean(axis=axis)
    return corr


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    import numbers
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def generate_trnval_folds(N, sampler='cv', nchunks=5, nfolds=5, testpct=0.2,
                          random_state=None):
    """Split dataset into training and validation folds

    Parameters
    ----------
    N : int
        The number of samples in the full training set
    nchunks : int
        Divide the dataset into chunks of size `nchunks` before sampling.
    nfolds : int, tuple
        Number of folds to return
    sampler : {'cv', 'bcv', 'mbb', nbb'}
        * cv:  standard k-fold cross-validation
                Samples are first split into chunks of size `nchunks`.
                The folds are then constructed by splitting these chunks
                into training sets of approximately (1 - 1/`nfolds`)% in size.
                For nfolds=5, the training set is ~80% in size.
        * bcv : repeated k-fold cross-validation.
                K-fold cross-validation repeated Q times.
                `nfolds` is given as a tuple of (nreps, nfolds).
                E.g., for twice repeated 5-fold cross-validation: `nfolds=(1, 5)`.
                This sampler first splits the dataset into chunks of size `nchunks`.
        * nbb : bootstrap
                Classic naive bootstrap sampler that respects `nchunks`.
                For each `fold`, a total of N*(1.0 - testpct)
                observations are sampled with replacement for the training set.
                The rest of the unsampled observations is used for the validation set.
        * mbb : moving block bootstrap
                Blocked bootstrap sampler that respects `nchunks`.
                For each `fold`, a total of approximately N*(1.0 - testpct)
                observations are sampled with replacement for the training set.
                The rest of the unsampled observations is used for the validation set.
                Bootstrap samples are generated in blocks of size `nchunks` and .
                the start of the blocks is random. The same training fold may
                contain the blocks: [[0,1,2,3,4], [2,3,4,5,6], ...].
    testpct : float (in 0-1 range), optional
        Only used when using bootstrap samplers (i.e. `mbb` and `nbb`)
    random_state : int, np.random.RandomState, or None
        Random generator state, used for reproducibility.

    Yields
    ------
    ifold : tuple of 1D arrays  (trainidx, validx)
        Training indices for each fold are the first element of the tuple.
        Validation indices for each fold are the second element of the tuple.

    Notes
    -----
    By default, this function is optimized for autocorrelated signals.
    If sampled signals are not autocorrelated, set `nchunks=1`

    Examples
    --------
    >>> folds = generate_trnval_folds(100, sampler='cv')
    >>> fold_sizes = [(len(trnidx),len(validx)) for trnidx, validx in folds]
    >>> print(fold_sizes)
    [(80, 20), (80, 20), (80, 20), (80, 20), (80, 20)]
    >>> folds = generate_trnval_folds(100, sampler='bcv', nfolds=(2,5))
    >>> print(len(list(folds)))
    10
    >>> folds = generate_trnval_folds(127, sampler='cv', nchunks=10, nfolds=5)
    >>> fold_sizes = [(len(trnidx),len(validx)) for trnidx, validx in folds]
    >>> print(fold_sizes)       # doctest: +SKIP
    [(97, 30), (97, 30), (107, 20), (107, 20), (107, 20)]
    >>> folds = generate_trnval_folds(100, sampler='nbb')
    >>> fold_sizes = [(len(np.unique(trnidx)),len(validx)) for trnidx, validx in folds]
    >>> print(fold_sizes)       # doctest: +SKIP
    [(50, 50), (60, 40), (55, 45), (60, 40), (55, 45)]
    >>> folds = generate_trnval_folds(100, sampler='mbb')
    >>> fold_sizes = [(len(np.unique(trnidx)),len(validx)) for trnidx, validx in folds]
    >>> print(fold_sizes)       # doctest: +SKIP
    [(57, 43), (60, 40), (60, 40), (65, 35), (51, 49)]
    """
    # TODO This function is a POS. Needs rewrite.
    oN = N
    ntrain = int(N - N*(testpct)) # for bootstrap only
    samples = np.arange(N)
    step = 1 if sampler == 'mbb' else nchunks
    samples = [samples[idx:idx+nchunks] for idx in range(0,N-nchunks+1, step)]
    N = len(samples)
    samples = [list(tt) for tt in samples]

    rng = check_random_state(random_state)

    append = lambda z: reduce(lambda x, y: x+y, z)
    allidx = np.arange(oN)
    if sampler == 'cv':
        rng.shuffle(samples)
        sets = np.array_split(np.arange(len(samples)), nfolds)
        for i,v in enumerate(sets):
            val = np.asarray(append([samples[t] for t in v]))
            train = allidx[~np.in1d(allidx, val)]
            yield train, val
    elif sampler == 'bcv':
        # Repeat the cross-validation N times
        assert isinstance(nfolds, tuple)
        reps, nfolds = nfolds
        for rdx in range(reps):
            rng.shuffle(samples)
            sets = np.array_split(np.arange(len(samples)), nfolds)
            for i,v in enumerate(sets):
                val = np.asarray(append([samples[t] for t in v]))
                train = allidx[~np.in1d(allidx, val)]
                yield train, val

    elif sampler == 'nbb' or sampler == 'mbb':
        fun = lambda x: [x[t] for t in rng.randint(0, N, int(ntrain/nchunks))]
        for bdx in range(nfolds):
            train = np.asarray(append(fun(samples)))
            val = allidx[~np.in1d(allidx, train)]
            yield train, val



def hrf_convolution(input_responses, HRF=None, do_convolution=True, dt=None):
    '''Convolve a series of impulses in a matrix with a given HRF

    Parameters
    ----------
    input_responses (n by p)
         A matrix containing ``p`` impulse time courses of length ``n``.
    HRF (m, or None)
         The HRF to convolve the impulses with. If ``None`` we will
         use the canonical HRF.
    dt (scalar)
         The sampling rate. This is only used to get the hemodynamic response
         function to convolve with if ``HRF`` is None.
    do_convolution (bool, or list of bools (p,))
         This tells us which responses to convolve and which to leave alone
         Defaults to ``True``, which convolves all responses

    Returns
    --------
    bold (n by p)
         The convolved impulses

    Examples
    ----------
    >>> impulses = np.zeros((100, 2))
    >>> impulses[5, 0] = 1
    >>> impulses[25, 1] = 1
    >>> bold = hrf_convolution(impulses, dt=1.0) # Default peak at 6s
    '''
    if input_responses.ndim == 1:
        input_responses = input_responses[...,None]

    bold = np.zeros_like(input_responses).astype(np.float)
    nresp = input_responses.shape[-1]

    if HRF is None:
        HRF = hrf_default_basis(dt=dt)[:, 0]

    if do_convolution is True:
        do_convolution = [True]*nresp
    for sidx in range(nresp):
        signal = input_responses[:, sidx]
        if do_convolution[sidx]:
            conv = np.convolve(signal, HRF, 'full')[:len(signal)]
        else:
            conv = 0.0
        bold[:, sidx] = conv
    return bold



def hyperopt_make_trial_data(tid, vals, loss):
    """
    Generate a valid dictionary as a trial for hyperopt.Trials.

    Parameters
    ----------
    tid : int
        trial id
    vals : dict
        mapping between parameter name and list of values, e.g.
        `{'feature_space_one': [1.0], 'feature_space_two': [100.0]}`
    loss : float
        loss for the current trial

    Returns
    -------
    trial : dict
        valid dict object for hyperopt.Trials
    """

    misc = {
        'tid': tid,
        'vals': vals,
        'cmd': ('domain_attachment', 'FMinIter_Domain'),
        'idxs': {k: [tid] for k in vals.keys()}
    }
    d = {
        'misc': misc,
        'tid': tid,
        'result': {'loss': loss, 'status': 'ok'},
        'state': 2,
        'spec': None,
        'owner': None,
        'book_time': None,
        'refresh_time': None,
        'exp_key': None
    }
    return d


def hyperopt_make_trials(values, losses, parameter_names=None):
    """

    Parameters
    ----------
    values : list of lists or 2D np.ndarray (n_trials, n_params)
        each element (or row) corresponds to a set of parameters previously
        tested
    losses : list of floats (n_params,)
        losses for previous trials
    parameter_names : list of str or None
        associated parameter names (must correspond to `spaces` passed to
        hyperopt). If None, defaults to ['X0', 'X1', ..., 'X`n_params`']

    Returns
    -------
    trials : hyperopt.Trials
        hyperopt Trials object containing reconstructed trials
    """
    import hyperopt as hpo
    # uniform the inputs
    nparams = len(values[0])
    if parameter_names is None:
        parameter_names = ['X{}'.format(i) for i in range(nparams)]
    vals = [{pn: [v] for pn, v in zip(parameter_names, val)} for val in values]
    trials = []
    for i, (v, l) in enumerate(zip(vals, losses)):
        trials.append(hyperopt_make_trial_data(i, v, l))
    hpo_trials = hpo.Trials()
    hpo_trials.insert_trial_docs(trials)
    hpo_trials.refresh()
    return hpo_trials


def analytic_expected_correlation(noise_level):
    r'''Expected correlation coefficient of simulated i.i.d. normal data.

    Compute the expectation on the correlation coefficient given
    the amount of noise in the data. Assumes signal and noise are
    i.i.d. normal with a fixed noise_level.

    Parameters
    ----------
    noise_level : float_like
        This corresponds to the sigma parameter of a MVN distribution.

    Returns
    -------
    expected_correlation : float
        The correlation coefficient that can be expected at the
        limit of infinite data given the amount of noise.

    Notes
    -----
    Assumes both signal and noise are generated from a MVN distribution
    with zero-mean and the noise variance is determined by :math:`\sigma` (`noise_level`).

    .. math::

        y = N(0, I)

        \epsilon = N(0, \sigma^2 I)

        y_{sampled} = y + \epsilon


    If our model is perfect (i.e. :math:`\hat{y} = y`), then the maximum correlation coefficient
    we can achieve is determined by :math:`\sigma`. Concretely:

    .. math::

        {\text{lim}_{n \to \infty}}:  R^2(\hat{y}, y_{sampled}) = \left(\frac{1}{1 + \sigma^2}\right)

        {\text{lim}_{n \to \infty}}: \rho(\hat{y}, y_{sampled}) = \sqrt{\left(\frac{1}{1 + \sigma^2}\right)}

    where :math:`\rho` is the correlation coefficient.

    Examples
    --------
    >>> nstim, noise_level = 100000, 1.0
    >>> ydata = np.random.randn(nstim)
    >>> noise = np.random.randn(nstim)*noise_level
    >>> ypred = zscore(ydata) + noise
    >>> empirical_correlation = columnwise_correlation(ydata, ypred)
    >>> print(empirical_correlation) # doctest: +SKIP
    0.7073279476259667
    >>> # As nstim -> inf, this converges to the analytic solution
    >>> expected_correlation = analytic_expected_correlation(noise_level)
    >>> print(round(expected_correlation, 6))
    0.707107
    >>> assert np.allclose(expected_correlation, empirical_correlation, atol=1e-02)
    '''
    return np.sqrt((1.0 + noise_level**2.0)**-1.0)


def cross_correlation(A, B, zscorea=True, zscoreb=True):
    '''Compute correlation for each column of A against
    every column of B (e.g. B is predictions).

    Parameters
    ----------
    A : 2D np.ndarray (n, p)
    B : 2D np.ndarray (n, q)

    Returns
    -------
    cross_corr : 2D np.ndarray (p, q)
    '''
    n = A.shape[0]

    # If needed
    if zscorea: A = zscore(A)
    if zscoreb: B = zscore(B)
    corr = np.dot(A.T, B)/float(n)
    return corr
