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

    See
    ---
    `scipy.linalg.svd` for full documentation
    '''
    import scipy.linalg as LA

    if 'full_matrices' not in kwargs:
        kwargs['full_matrices'] = False

    try:
        O = LA.svd(X, **kwargs)
    except LA.LinAlgError as e:
        from warnings import warn
        warn('%s... trying slow SVD' % e)
        from svd_dgesvd import svd_dgesvd as slow_svd
        O = slow_svd(X, **kwargs)
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
    '''Value to devide a matrix such that its determinant is 1.

    Compute the pseudo-determinant of the matrix
    '''
    evals = np.linalg.eigvalsh(mat)
    gdx = evals > thresh
    det = np.prod(evals[gdx])
    scale = det**(1./gdx.sum())
    if np.isinf(scale) or np.isnan(scale) or scale==0:
        scale = 1.0
    return scale

def delay2slice(delay):
    if delay > 0:
        ii = slice(None, -delay)
    elif delay == 0:
        ii = slice(None, None)
    elif delay < 0:
        raise ValueError('No negative delays')
    return ii


def generate_data(n=100, p=10, v=2,
                  noise=1.0,
                  testsize=0,
                  dozscore=False,
                  feature_sparsity=0.0):
    '''Get some B,X,Y data generated from gaussian (0,1).

    Parameters
    ----------
    n (int): Number of samples
    p (int): Number of features
    v (int): Number of responses
    noise (float): Noise level
    testsize (int): samples in validation set
    dozscore (bool): z-score features and responses?
    feature_sparsity (float in 0-1):
        number of irrelevant features as
        percentage of total

    Returns
    -------
    B (p-by-v np.ndarray):
        True feature weights
    (Xtrain, Xval):
        Feature matrix for training and validation
        Xval is optional and contains no noise
    (Ytrain, Yval):
        Response matrix for training and validation
        Yval is optional and contains no noise

    Examples
    --------
    >>> B, X, Y = generate_data(n=100, p=10, v=2,noise=1.0,testsize=0)
    >>> B.shape, X.shape, Y.shape
    ((10, 2), (100, 10), (100, 2))
    >>> B, (Xtrn, Xval), (Ytrn, Yval) = generate_data(n=100, p=10, v=2,testsize=20)
    >>> B.shape, Xval.shape, Yval.shape
    ((10, 2), (20, 10), (20, 2))
    '''
    X = np.random.randn(n, p)
    B = np.random.randn(p, v)
    if dozscore: X = zscore(X)
    nzeros = int(feature_sparsity*p)
    if nzeros > 0:
        B[-nzeros:,:] = 0

    Y = np.dot(X, B)
    Y += np.random.randn(*Y.shape)*noise
    if dozscore: Y = zscore(Y)

    if testsize:
        Xval = np.random.randn(testsize, p)
        if dozscore:
            Xval = zscore(Xval)
        Yval = np.dot(Xval, B)
        if dozscore: Yval = zscore(Yval)
        return B, (X,Xval), (Y, Yval)

    return B, X,Y


def mult_diag(d, mtx, left=True):
    """
    Multiply a full matrix by a diagonal matrix.
    This function should always be faster than dot.

    Parameters
    ----------
    d [1D (N,) array]
         contains the diagonal elements
    mtx [2D (N,N) array]
         contains the matrix

    Returns
    --------
    res (N, N)
         Result of multiplying the matrices

    Notes
    ------
    This code by:
    Pietro Berkes berkes@gatsby.ucl.ac...
    Mon Mar 26 11:55:47 CDT 2007
    http://mail.scipy.org/pipermail/numpy-discussion/2007-March/026807.html

    mult_diag(d, mts, left=True) == dot(diag(d), mtx)
    mult_diag(d, mts, left=False) == dot(mtx, diag(d))


    """
    if left:
        return (d*mtx.T).T
    else:
        return d*mtx


def noise_ceiling_correction(repeats, yhat, dozscore=True):
    """Noise ceiling corrected correlation coefficient.

    Parameters
    ----------
    repeats: np.ndarray, (nreps, ntpts, nsignals)
        The stimulus timecourses for each repeat.
        Each repeat is `ntpts` long.
    yhat: np.ndarray, (ntpts, nsignals)
        The predicted timecourse for each signal.
    dozscore: bool
        This algorithm only works correctly if the
        `repeats` and `yhat` timecourses are z-scored.
        If these are already z-scored, set to False.

    Returns
    -------
    r_ns: np.ndarray, (nsignals)
        The noise ceiling corrected correlation coefficient
        for each of the signals. One may square this result
        (while keeping the sign) to obtain the
        'explainable variance explained.'

    Notes
    -----
    $r_{ns}$ is misbehaved if $R^2$ is very low.

    References
    ----------
    Schoppe, et al. (2016), Hsu, et al. (2004), David, et al. (2005).
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

    This can be interpreted as the R^2 of a model that predicts
    each repetition with the mean across repetitions.

    Parameters
    ----------
    repeats: np.ndarray, (nreps, ntpts, nsignals)
        The timecourses for each stimulus repetition.
        Each of repeat is `ntpts` long.
    ncorrection: bool
        Bias correction for number of repeats.
        Equivalent to computing the adjusted R^2.
    dozscore: bool
        This algorithm only works with z-scored repeats. If
        the each repetition is already z-scored, set to False.

    Returns
    -------
    EV: np.ndarray (nsignals)
        The explainable variance computed across repeats.
        Equivalently, the (adjusted) R^2 value.
    '''
    repeats = zscore(repeats, 1) if dozscore else repeats
    residual = repeats - repeats.mean(0)
    residualvar = np.mean(residual.var(1), 0)
    ev = 1 - residualvar

    if ncorrection:
        ev = ev - ((1 - ev) / np.float((repeats.shape[0] - 1)))
    return ev


def absmax(arr):
    return max(abs(np.nanmin(arr)), abs(np.nanmax(arr)))


def delay_signal(data, delays=[0, 1, 2, 3], fill=0):
    '''
    >>> x = np.arange(6).reshape(2,3).T + 1
    >>> x
    array([[1, 4],
           [2, 5],
           [3, 6]])
    >>> delay_signal(x, [-1,2,1,0], fill=0)
    array([[2, 5, 0, 0, 0, 0, 1, 4],
           [3, 6, 0, 0, 1, 4, 2, 5],
           [0, 0, 1, 4, 2, 5, 3, 6]])
    >>> delay_signal(x, [-1,2,1,0], fill=np.nan)
    array([[  2.,   5.,  nan,  nan,  nan,  nan,   1.,   4.],
           [  3.,   6.,  nan,  nan,   1.,   4.,   2.,   5.],
           [ nan,  nan,   1.,   4.,   2.,   5.,   3.,   6.]])
    '''
    if data.ndim == 1:
        data = data[...,None]
    n, p = data.shape
    out = np.ones((n, p*len(delays)), dtype=data.dtype)*fill

    for ddx, num in enumerate(delays):
        beg, end = ddx*p, (ddx+1)*p
        if num == 0:
            out[:, beg:end] = data
        elif num > 0:
            out[num:, beg:end] = data[:-num]
        elif num < 0:
            out[:num, beg:end] = data[abs(num):]
    return out


def whiten_penalty(X, penalty=0.0):
    '''Whiten features (p)
    X (n, p) (e.g. time by features)
    '''
    cov = np.cov(X.T)
    u, s, ut = np.linalg.svd(cov, full_matrices=False)
    covnegsqrt = np.dot(mult_diag((s+penalty)**(-1/2.0), u, left=False), ut)
    return np.dot(covnegsqrt, X.T).T


def columnwise_rsquared(ypred, y, **kwargs):
    '''

    Notes
    -----
    https://en.wikipedia.org/wiki/Coefficient_of_determination#Extensions
    '''
    return 1 - np.var(y - ypred, axis=0)/np.var(y, axis=0)


def columnwise_correlation(ypred, y, zscorea=True, zscoreb=True, axis=0):
    r'''Compute correlations efficiently

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




def generate_trnval_folds(N, sampler='cv', testpct=0.2, nchunks=5, nfolds=5):
    '''
    N : int
        The number of samples in the training set
    samplers : str
        * cv: `nfolds` for cross-validation
        * bcv: `nfolds` is a tuple of (nrepeats, nfolds)
               for repeated cross-validation.
    '''
    oN = N
    ntrain = int(N - N*(testpct))
    samples = np.arange(N)
    step = 1 if sampler == 'mbb' else nchunks
    samples = [samples[idx:idx+nchunks] for idx in range(0,N-nchunks+1, step)]
    N = len(samples)
    samples = [list(tt) for tt in samples]

    append = lambda z: reduce(lambda x, y: x+y, z)
    allidx = np.arange(oN)
    if sampler == 'cv':
        np.random.shuffle(samples)
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
            np.random.shuffle(samples)
            sets = np.array_split(np.arange(len(samples)), nfolds)
            for i,v in enumerate(sets):
                val = np.asarray(append([samples[t] for t in v]))
                train = allidx[~np.in1d(allidx, val)]
                yield train, val

    elif sampler == 'nbb' or sampler == 'mbb':
        fun = lambda x: [x[t] for t in np.random.randint(0, N, int(ntrain/nchunks))]
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
         use the canonical given by :func:`hrf`
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
        `{'speechsem': [1.0], 'visualsem': [100.0]}`
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


def test_make_trials():
    """smoke test"""
    values = [[1.0, 3.0, 4.0],
              [44.0, 33.0, 2.0]]
    losses = [0.3, -0.2]

    hpo_trials = hyperopt_make_trials(values, losses)
    parameter_names = ['X{}'.format(i) for i in range(3)]
    vals = [{pn: [v] for pn, v in zip(parameter_names, val)} for val in values]

    assert len(hpo_trials.trials) == len(values)
    for trl, val, loss in zip(hpo_trials.trials, vals, losses):
        assert trl['result']['loss'] == loss
        assert trl['misc']['vals'] == val
    return hpo_trials
