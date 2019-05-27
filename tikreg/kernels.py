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

    Parameters
    ----------
    xdata (nx-by-p)
        The feature matrix X
    ydata (ny-by-p)
        The feature matrix Y. If none is given,
        the kernel space in ``xdata`` is constructed.
    kernel_type (str)
        * 'linear': Linear kernel (default)
        * 'poly' : Inhomogeneous polynomial kernel
        * 'polyhomo': Homogeneous polynomial kernel
        * 'gaussian': Gaussian kernel
        * 'multiquad': Multiquadratic kernel
    """
    def __init__(self, xdata, ydata=None,
                 kernel_type=None, dtype=np.float64):
        if kernel_type is None:
            kernel_type = 'linear'
        self.kernel_types = ['gaussian', 'multiquad', 'linear', 'poly', 'polyhomo']
        assert kernel_type in self.kernel_types
        self.kernel_type = kernel_type
        self.kernel_parameter = None
        if kernel_type in ['linear', 'poly', 'polyhomo']:
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
            * linear <-> poly <-> polyhomo

        Returns
        -------
        lazy_kernel.kernel (nx by ny)
            Sets the ``kernel`` property with the computed kernel.
        '''
        if kernel_type is None:
            kernel_type = self.kernel_type

        # Linear and polynomial kernels are a special case
        innerprod_kernels = ['linear', 'poly', 'polyhomo']
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
            print('kernel is already set: %s'%self)
            return

        self.kernel_parameter = kernel_parameter
        self.kernel_type = kernel_type
        if kernel_type == 'gaussian':
            self.kernel = np.exp(-1*self.cache/(2*self.kernel_parameter**2))
        elif kernel_type == 'multiquad':
            self.kernel = np.sqrt(self.cache + self.kernel_parameter**2)
        elif kernel_type == 'poly':
            self.kernel = (self.cache + 1.0)**self.kernel_parameter
        elif kernel_type == 'polyhomo':
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
    '''
    xdata is n-by-p
    ydata is m-by-p or none
    '''
    if ydata is None:
        ydata = xdata
    return np.dot(xdata, ydata.T)


def polyhomokern(data,ydata=None,powa=2):
    '''
    Compute the polynomial kernel to the
    `powa` degree
    '''
    return linear_kernel(data,ydata)**powa


def polyinhomo(data,ydata=None,powa=2):
    '''
    Compute the inhomogeneous polynomial kernel to the
    `powa` degree
    '''
    return (linear_kernel(data,ydata) + 1)**powa


def multiquad_kernel(xdata, ydata=None, c=1.0):
    '''Multiquadratic kernel.
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
    xdata (nx by p)
    ydata (ny by p)
         If None, ydata = xdata, so we compute the
         gaussian kernel for the dataset ``xdata`` along the first dimension
    sigma (float):
         The width of the gaussian

    Returns
    -------
    gaukern (nx by ny) np.ndarray
         The gaussian kernel
    '''
    sigma = float(sigma)
    norm = -1*vector_norm_sq(xdata, ydata=ydata)
    return np.exp((norm/(2*sigma**2)))


def vector_norm_sq(xdata, ydata=None):
    '''
    Compute the element-wise vector norm
    across two matrices. This assumes the
    vectors are contained in the first dimension.

    Parameters
    ----------
    xdata  (nx by p)
    ydata  (ny by p)
         If None, ydata = xdata, so we compute the,
         norm of each vector in the dataset ``xdata``

    Returns
    -------
    Q (nx by ny)
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
    '''Basically a wrapper around poly

    Each column of X is used to
    '''
    n, p = X.shape
    K = 0.0
    for pdx in xrange(p):
        delayed_feature = delay_signal(X[:,[pdx]], delays)
        K += np.dot(delayed_feature, delayed_feature.T)
    return K
