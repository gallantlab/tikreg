'''
Kernels!
'''
import numpy as np

class SwitchError(Exception):
    pass


class lazy_kernel(object):
    """Generate kernels from cache

    Some kernels operate on the inner-product space, others operate
    on vector norms. This class caches these matrices and lets the
    user construct the kernel with a given parameter from the cache.
    This is fast because the inner-products don't need to be recomputed every
    time a new parameter is passed. This is useful when cross-validating kernel
    parameters (e.g. for a gaussian kernel) or when switching between different
    kernels types (e.g. from gaussian to multi-quadratic).
    """
    def __init__(self, xdata, ydata=None,
                 kernel_type=None, dtype=np.float64):
        '''
        Parameters
        ----------
        xdata : 2D np.ndarray (nx, p)
            The feature matrix X
        ydata : 2D np.ndarray (ny, p)
            If None, ``ydata = xdata``.
        kernel_type (str)
            * 'linear': Linear kernel (default)
            * 'ihpolykern' : Inhomogeneous polynomial kernel
            * 'hpolykern': Homogeneous polynomial kernel
            * 'gaussian': Gaussian kernel
            * 'multiquad': Multiquadratic kernel
        '''
        if kernel_type is None:
            kernel_type = 'linear'
        self.kernel_types = ['gaussian', 'multiquad', 'linear', 'ihpolykern', 'hpolykern']
        assert kernel_type in self.kernel_types
        self.kernel_type = kernel_type
        self.kernel_parameter = None
        if kernel_type in ['linear', 'ihpolykern', 'hpolykern']:
            # This simply gets the linear kernel
            self.cache = linear_kernel(xdata, ydata).astype(dtype)
            if kernel_type == 'linear':
                self.kernel = self.cache
        else:
            # The other kernels operate on the sq norm difference
            self.cache = vector_norm_sq(xdata, ydata).astype(dtype)

    def __repr__(self):
        return '%s.kernel(%s).kernel_parameter(%s)' % (__package__, self.kernel_type, str(self.kernel_parameter))

    def update(self, kernel_parameter, kernel_type=None, verbose=False):
        '''Update the kernel with the parameter given.

        Parameters
        ----------
        kernel_parameter (float-like)
            The parameter for the particular kernel
        kernel_type (str)
            The kernel to use. If none is given, assume
            we are using the kernel with which the class was instantiated.
            One can switch kernels in these directions:

            * multiquad <-> gaussian

            * linear <-> ihpolykern <-> hpolykern

        Returns
        -------
        lazy_kernel.kernel : np.ndarray (nx, ny)
            Sets the ``kernel`` property with the computed kernel.
        '''
        if kernel_type is None:
            kernel_type = self.kernel_type

        # Linear and polynomial kernels are a special case
        innerprod_kernels = ['linear', 'ihpolykern', 'hpolykern']
        if self.kernel_type in innerprod_kernels:
            if not (kernel_type in innerprod_kernels):
                msg = 'Cannot update "%s" to "%s". '%(self.kernel_type, kernel_type)
                raise SwitchError(msg)
        else:
            if (kernel_type in innerprod_kernels):
                msg = 'Cannot update "%s" to "%s". '%(self.kernel_type, kernel_type)
                raise SwitchError(msg)

        if self.kernel_type == 'linear' and kernel_type == 'linear':
            if verbose: print('kernel is already set: %s'%self)
            return

        # Check if we already have this update, in which case just return verbosely
        if (kernel_type == self.kernel_type) and (self.kernel_parameter == kernel_parameter):
            if verbose: print('kernel is already set: %s'%self)
            return

        self.kernel_parameter = kernel_parameter
        self.kernel_type = kernel_type
        if kernel_type == 'gaussian':
            self.kernel = np.exp(-1*self.cache/(2*self.kernel_parameter**2))
        elif kernel_type == 'multiquad':
            self.kernel = np.sqrt(self.cache + self.kernel_parameter**2)
        elif kernel_type == 'ihpolykern':
            self.kernel = (self.cache + 1.0)**self.kernel_parameter
        elif kernel_type == 'hpolykern':
            self.kernel = (self.cache)**self.kernel_parameter
        elif kernel_type == 'linear':
            self.kernel = self.cache
        else:
            raise ValueError('Kernel "%s" is not available. Choose from: %s' % (kernel_type, ','.join(self.kernel_types)))
        if verbose:
            print(self)


##############################
# Kernel constructor functions
##############################


def linear_kernel(xdata, ydata=None):
    '''Compute a linear kernel

    Parameters
    ----------
    xdata : 2D np.ndarray (nx, p)
    ydata : 2D np.ndarray (ny, p), or None
        If None, ``ydata = xdata``.

    Returns
    -------
    linear_kernel : 2D np.ndarray (nx, ny)
    '''
    if ydata is None:
        ydata = xdata
    return np.dot(xdata, ydata.T)


def homogeneous_polykern(data,ydata=None,powa=2):
    '''Compute the homogeneous polynomial kernel.

    The polynomial expansion does not include interactions.

    Parameters
    ----------
    xdata : 2D np.ndarray (nx, p)
    ydata : 2D np.ndarray (ny, p), or None
        If None, ``ydata = xdata``.
    powa : scalar
        Degree of polynomial expansion.
        Defaults to 2.

    Returns
    -------
    hpoly_kernel : 2D np.ndarray (nx, ny)
    '''
    return linear_kernel(data,ydata)**powa


def inhomogeneous_polykern(data,ydata=None,powa=2):
    '''Compute the homogeneous polynomial kernel.

    The polynomial expansion includes interaction terms.

    Parameters
    ----------
    xdata : 2D np.ndarray (nx, p)
    ydata : 2D np.ndarray (ny, p), or None
        If None, ``ydata = xdata``.
    powa : scalar
        Degree of polynomial expansion.
        Defaults to 2.

    Returns
    -------
    ihpoly_kernel : 2D np.ndarray (nx, ny)
    '''
    return (linear_kernel(data,ydata) + 1)**powa


def multiquad_kernel(xdata, ydata=None, c=1.0):
    '''Compute the multi-quadratic kernel.

    Parameters
    ----------
    xdata : 2D np.ndarray (nx, p)
    ydata : 2D np.ndarray (ny, p), or None
        If None, ``ydata = xdata``.
    c : scalar
        Defaults to 1.

    Returns
    -------
    multiquad_kernel : 2D np.ndarray (nx, ny)
    '''
    norm = vector_norm_sq(xdata, ydata)
    # Based on M. Oliver's MATLAB implementation.
    return np.sqrt(norm + c**2)


def gaussian_kernel(xdata, ydata=None, sigma=1.0):
    '''
    Compute the gaussian kernel along the first dimension
    This uses the :func:`vector_norm_sq` to compute
    the norms across the vectors in the matrices
    and then re-scales it

    Parameters
    ----------
    xdata : 2D np.ndarray (nx, p)
    ydata : 2D np.ndarray (ny by p) or None
         If None, ``ydata = xdata``.
    sigma (float):
         The width of the gaussian.

    Returns
    -------
    gaussian_kern : 2D np.ndarray (nx, ny)
         The gaussian kernel
    '''
    sigma = float(sigma)
    norm = -1*vector_norm_sq(xdata, ydata=ydata)
    return np.exp((norm/(2*sigma**2)))


def vector_norm_sq(xdata, ydata=None):
    '''Compute the squared vector norm.

    This assumes the vectors are in the first dimension.

    Parameters
    ----------
    xdata : 2D np.ndarray (nx by p)
    ydata : 2D np.ndarray (ny by p) or None
         If None, ydata = xdata

    Returns
    -------
    Q : 2D np.ndarray (nx, ny)
         The vectorm norms

    Examples
    --------
    >>> xdata = np.arange(3*10).reshape(3,10)
    >>> print(vector_norm_sq(xdata))
    [[   0 1000 4000]
     [1000    0 1000]
     [4000 1000    0]]
    '''
    simple = False
    if ydata is None:
        ydata = xdata
        simple = True

    XY = np.dot(xdata, ydata.T)
    sqX = np.sum(xdata**2, axis=1)[...,None]
    sqY = sqX if simple else np.sum(ydata**2, axis=1)[...,None]
    Q = (sqX - XY) + (sqY - XY.T).T
    return Q


def volterra_temporal(X, delays=[0,1,2], degree=2.0):
    '''Volterra series.

    Implemented as a wrapper around inhomogenous poly
    '''
    n, p = X.shape
    K = 0.0
    for pdx in xrange(p):
        delayed_feature = delay_signal(X[:,[pdx]], delays)
        K += np.dot(delayed_feature, delayed_feature.T)
    return K
