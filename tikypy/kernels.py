from string import join as sjoin
import numpy as np

from tikypy.utils import mult_diag


class lazy_kernel(object):
    '''
    Initialize as linear for linear
    cannot go from linear to others or from others to linear
    '''
    def __init__(self, xdata=None, ydata=None,
                 kernel_type=None, dtype=np.float64):
        if kernel_type is None:
            kernel_type = 'linear'
        self.kernel_types = ['gaussian', 'multiquad', 'linear', 'poly', 'polyhomo']
        self.kernel_type = kernel_type
        self.parameter = None
        if kernel_type in ['linear', 'poly', 'polyhomo']:
            # This simply gets the linear kernel
            self.cache = linear_kernel(xdata, ydata).astype(dtype)
            if kernel_type == 'linear':
                self.kernel = self.cache
        else:
            # The other kernels operate on the norm
            self.cache = vector_norm(xdata, ydata).astype(dtype)

    def __repr__(self):
        return '%s.kernel(%s).parameter(%s)' % (__package__, self.kernel_type, str(self.parameter))

    def update(self, parameter, kernel_type=None, verbose=False):
        if kernel_type is None:
            kernel_type = self.kernel_type
        # Linear and polynomial kernels are a special case

        innerprod_kernels = ['linear', 'poly', 'polyhomo']
        if self.kernel_type in innerprod_kernels:
            if not (kernel_type in innerprod_kernels):
                msg = 'Cannot update "%s" to "%s". '%(self.kernel_type, kernel_type)
                raise ValueError(msg)
        else:
            if (kernel_type in innerprod_kernels):
                msg = 'Cannot update "%s" to "%s". '%(self.kernel_type, kernel_type)
                raise ValueError(msg)


        if self.kernel_type == 'linear' and kernel_type == 'linear':
            if verbose:
                print('kernel is already set:%s'%self)
            return

        # Check if we already have this update, in which case just return verbosely
        if (kernel_type == self.kernel_type) and (self.parameter == parameter):
            if verbose: print 'kernel is already set:', self
            return

        self.parameter = parameter
        self.kernel_type = kernel_type
        if kernel_type == 'gaussian':
            self.kernel = np.exp(-1*self.cache/(2*self.parameter**2))
        elif kernel_type == 'multiquad':
            self.kernel = np.sqrt(self.cache + self.parameter**2)
        elif kernel_type == 'poly':
            self.kernel = (self.cache + 1.0)**self.parameter
        elif kernel_type == 'polyhomo':
            self.kernel = (self.cache)**self.parameter
        elif kernel_type == 'linear':
            self.kernel = self.cache
        else:
            raise ValueError('Kernel "%s" is not available. Choose from: %s' % (kernel_type,  sjoin(self.kernel_types, ', ')))
        if verbose: print self


def multiquad_kernel(xdata, ydata = None, c =1.0 ):
    '''
    Multiquadratic kernel.
    Code taken from M.O.
    Matlab implementation: /auto/k1/moliver/code/krls.m
    '''
    norm = vector_norm(xdata, ydata)
    return np.sqrt(norm + c**2)

class lazy_multiquad_kernel:
    def __init__(self, xdata, ydata = None):
        self.sqnorm = vector_norm(xdata, ydata)
    def set_constant(self, c = 1.0):
        self.kernel = np.sqrt(self.sqnorm + c**2)

# Kernel constructor functions
############################################################
def linear_kernel(xdata, ydata=None):
    '''
    xdata is n-by-p
    ydata is m-by-p or none
    '''
    if ydata is None:
        ydata = xdata
    return np.dot(xdata, ydata.T)


def linkern(data, ydata=None):
    '''
    Produce a linear kernel of the input, `data`.

    Parameters
    ----------------
    `data` (2D numpy array)
           A (p by n) matrix, where `p` is the number of
           dimensions and `n` is the number of observations

    Returns
    -----------
    `kDat` (2D numpy array)
           A (n by n) matrix of the (linear) inner product space
           of `data` (i.e. np.dot(data.T, data)).

    Examples
    ---------
    >>> a = np.zeros((10,2))
    >>> a[:,0] = np.arange(10)
    >>> a[:,1] = np.arange(10,20)
    >>> a
    array([[  0.,  10.],
           [  1.,  11.],
           [  2.,  12.],
           [  3.,  13.],
           [  4.,  14.],
           [  5.,  15.],
           [  6.,  16.],
           [  7.,  17.],
           [  8.,  18.],
           [  9.,  19.]])
    >>> linkern(a)
    array([[  285.,   735.],
           [  735.,  2185.]])
    '''
    if ydata is None:
        res = np.dot(data.T, data)
    else:
        res = np.dot(data.T, ydata)
    return res


def polykern(data,ydata=None,powa=2):
    '''
    Compute the polynomial kernel to the
    `powa` degree

    '''

    return linkern(data,ydata)**powa


def fast_gaukern(xdata, ydata=None, sigma=1.0):
    '''
    A faster more readable implementation fo the gaussian kernel.
    It computes it along the first dimension, so the 2nd dimension
    elements must be the same size.

    Parameters
    ------------------
    `xdata`  np.ndarray:
         An (nx by p) dataset
    `ydata`  np.ndarray:
         An (ny by p) np.ndarray or None:
         If None, ydata = xdata, so we compute the
         gaussian kernel for the dataset `xdata` along the first dimension
    `sigma` float :
         The width of the gaussian

    Returns
    ------------
    `gaukern` (nx by ny) np.ndarray
         The gaussian kernel

    Examples
    ----------
    >>> np.random.seed(33)
    >>> xdata = np.random.randint(1,5, size=(4,10)).astype(np.float)
    >>> ydata = np.random.randint(1,5, size=(3,10)).astype(np.float)
    >>> np.round(fast_gaukern(xdata, sigma=3.0), 4)
    array([[ 1.    ,  0.3114,  0.6065,  0.2787],
           [ 0.3114,  1.    ,  0.3292,  0.1512],
           [ 0.6065,  0.3292,  1.    ,  0.1353],
           [ 0.2787,  0.1512,  0.1353,  1.    ]])
    >>> xdata.shape
    (4, 10)
    >>> np.round(fast_gaukern(xdata, sigma=3.0), 4).shape
    (4, 4)
    >>> np.round(fast_gaukern(xdata, ydata=ydata, sigma=3.0), 4)
    array([[ 0.3114,  0.4857,  0.2787],
           [ 0.169 ,  0.2636,  0.3292],
           [ 0.2946,  0.4111,  0.2359],
           [ 0.0868,  0.2359,  0.0622]])
    >>> np.round(fast_gaukern(xdata, ydata=ydata, sigma=3.0), 4).shape
    (4, 3)
    >>> xdata.shape[0], ydata.shape[0]
    (4, 3)
    '''

    if ydata is None:
        ydata = xdata

    XY = np.dot(xdata, ydata.T)
    sqX = np.sum(xdata**2, axis=1)
    sqY = np.sum(ydata**2, axis=1)
    dumbX = np.ones((xdata.shape[0], 1))
    dumbY = np.ones((ydata.shape[0], 1))
    xsq = np.dot(sqX[...,None], dumbY.T)
    ysq = np.dot(dumbX, sqY[...,None].T)
    Q = (2*XY - xsq - ysq) / float((2*sigma**2))
    return np.exp(Q)

def single_gaukern(data, sigma=1.0):
    '''
    TODO: Make faster by using symmetry!

    Compute the gaussian kernel for a single dataset
    along the second dimension dimension
    This is a bit faster than the generalized version
    since we do not need to do some computations

    Parameters
    -------------------
    data (n by p)

    Returns
    -------
    gaukern (p by p)
         The gaussian kernel

    Examples
    ----------
    >>> np.random.seed(33)
    >>> data = np.random.randint(1,5, size=(20,10)).astype(np.float)
    >>> gaukern = single_gaukern(data, sigma=3.0)
    >>> gaukern.shape
    (10, 10)
    >>> data.shape
    (20, 10)
    '''
    kern = np.dot(data.T, data)
    sqX = np.sum(data**2, axis=0)
    dumbX = np.ones((data.shape[1], 1))
    xsq = mult_diag(sqX, dumbX.T)

    tmp = 2*kern - xsq - xsq.T
    scale = float(2*sigma**2)
    gaukern = np.exp(tmp/scale)
    return gaukern

def single_time_gaukern(data, sigma=1.0):
    '''

    Compute the gaussian kernel for a single dataset
    along the first dimension dimension
    This is a bit faster than the generalized version
    since we do not need to do some computations

    Parameters
    -------------------
    data (n by p)

    Returns
    -------
    gaukern (n by n)
         The gaussian kernel

    Examples
    ----------
    >>> np.random.seed(33)
    >>> data = np.random.randint(1,5, size=(10,20)).astype(np.float)
    >>> n, p = data.shape
    >>> gaukern = single_time_gaukern(data, sigma=3.0)
    >>> gaukern.shape == (n, n)
    True
    '''

    kern = np.dot(data, data.T)
    sqX = np.sum(data**2, axis=1)
    dumbX = np.ones((data.shape[0], 1))
    xsq = mult_diag(sqX, dumbX.T)

    tmp = 2*kern - xsq - xsq.T
    scale = float(2*sigma**2)
    gaukern = np.exp(tmp/scale)
    return gaukern


def single_vector_norm(data):
    '''
    TODO: Make faster by using symmetry!

    Compute the vector norms for a single dataset
    along the second dimension dimension
    This is a bit faster than the generalized version
    since we do not need to do some computations

    Parameters
    -----------
    `data` (n by p) np.ndarray
         The dataset

    Returns
    --------
    vecnorms (p by p) np.ndarray
         The matrix of vector norms

    Examples
    ---------
    >>> np.random.seed(33)
    >>> xdata = np.random.randint(1,5, size=(10,4)).astype(np.float)
    >>> res = single_vector_norm(xdata)
    >>> res, res.shape
    (array([[  0.,  24.,  32.,  33.],
           [ 24.,   0.,  40.,  25.],
           [ 32.,  40.,   0.,  35.],
           [ 33.,  25.,  35.,   0.]]), (4, 4))


    '''
    kern = np.dot(data.T, data)
    sqX = np.sum(data**2, axis=0)
    dumbX = np.ones((data.shape[1], 1))
    xsq = mult_diag(sqX, dumbX.T)
    vecnorm = xsq + xsq.T - 2*kern
    return vecnorm

def multiquad_kernel(xdata, ydata=None, c =1.0):
    '''
    Multiquadratic kernel.
    Code taken from M.O.
    Matlab implementation: /auto/k1/moliver/code/krls.m
    '''
    norm = vector_norm(xdata, ydata)
    return np.sqrt(norm + c**2)

class lazy_multiquad_kernel:
    def __init__(self, xdata, ydata = None):
        self.sqnorm = vector_norm(xdata, ydata)
    def set_constant(self, c = 1.0):
        self.kernel = np.sqrt(self.sqnorm + c**2)


def gaussian_kernel(xdata, ydata=None, sigma=1.0):
    '''
    Compute the gaussian kernel along the first dimension
    This uses the :func:`vector_norm` to compute
    the norms across the vectors in the matrices
    and then re-scales it

    Parameters
    ------------------
    `xdata`  np.ndarray:
         An (nx by p) dataset
    `ydata`  np.ndarray:
         An (ny by p) np.ndarray or None:
         If None, ydata = xdata, so we compute the
         gaussian kernel for the dataset `xdata` along the first dimension
    `sigma` float :
         The width of the gaussian

    Returns
    ------------
    `gaukern` (nx by ny) np.ndarray
         The gaussian kernel

    Examples
    ----------
    >>> np.random.seed(33)
    >>> xdata = np.random.randint(1,5, size=(10,4)).astype(np.float)
    >>> ydata = np.random.randint(1,5, size=(20,4)).astype(np.float)
    >>> norm = vector_norm(xdata)
    >>> resgau = np.exp(((-1*norm)/(2*3.0**2)))
    >>> np.allclose(fast_gaukern(xdata, sigma=3.0), resgau)
    True
    >>> norm = vector_norm(xdata, ydata=ydata)
    >>> resgau = np.exp(((-1*norm)/(2*5.0**2)))
    >>> np.allclose(fast_gaukern(xdata, ydata=ydata, sigma=5.0), resgau)
    True
    '''
    sigma = float(sigma)
    norm = -1*vector_norm(xdata, ydata=ydata)
    return np.exp((norm/(2*sigma**2)))


def vector_norm(xdata, ydata=None, jk=False):
    '''
    Compute the element-wise vector norm
    across two matrices. This assumes the
    vectors are contained in the first dimension.

    Parameters
    ------------------
    `xdata`  np.ndarray:
         An (nx by p) dataset
    `ydata`  np.ndarray:
         An (ny by p) np.ndarray or None
         If None, ydata = xdata, so we compute the,
         norm of each vector in the dataset `xdata`
    `jk` bool:
         If "j/k", then simply return the linear kernel

    Returns
    ------------
    `Q` (nx by ny) np.ndarray
         The gaussian kernel

    Examples
    ----------
    >>> np.random.seed(33)
    >>> xdata = np.random.randint(1,5, size=(3,10)).astype(np.float)
    >>> ydata = np.random.randint(1,5, size=(2,10)).astype(np.float)
    >>> np.round(vector_norm(xdata), 4)
    array([[  0.,  21.,   9.],
           [ 21.,   0.,  20.],
           [  9.,  20.,   0.]])
    >>> np.round(vector_norm(xdata, ydata=ydata), 4)
    array([[ 23.,  21.],
           [ 34.,  32.],
           [ 36.,  22.]])
    '''
    if ydata is None:
        ydata = xdata

    XY = np.dot(xdata, ydata.T)
    if jk: return XY
    sqX = np.sum(xdata**2, axis=1)
    sqY = np.sum(ydata**2, axis=1)
    dumbX = np.ones((xdata.shape[0], 1))
    dumbY = np.ones((ydata.shape[0], 1))

    # np.dot is faster than outer product
    xsq = np.dot(sqX[...,None], dumbY.T)
    ysq = np.dot(dumbX, sqY[...,None].T)

    Q = xsq + ysq - 2*XY
    return Q


def _test_(xdata, ydata=None):
    if ydata is None:
        ydata = xdata

    XY = np.dot(xdata, ydata.T)
    sqX = np.sum(xdata**2, axis=1)
    sqY = np.sum(ydata**2, axis=1)
    Q = xsq + ysq - 2*XY
    return Q



def gaukern(data,ydata=None, sigma=1.0, simple=False):
    '''
    Compute the gaussian kernel for a given `sigma` (gaussian width)
    parameter

    Code stolen from apgl - validated against matlab by me (Anwar)
    '''
    is1d = False
    if ydata is not None:
        X1 = data.T
        X2 = ydata.T
    else:
        X1 = data.copy().T
        X2 = data.copy().T

    if X1.shape[-1] != X2.shape[-1]:
        raise ValueError("Invalid matrix dimentions: " + str(X1.shape) + " " + str(X2.shape))

    j1 = np.ones((X1.shape[0], 1))
    j2 = np.ones((X2.shape[0], 1))

    try:
        diagK1 = np.sum(X1**2, 1)
        diagK2 = np.sum(X2**2, 1)
    except ValueError:
        if data.ndim == 1:
            is1d = True
            diagK1 = np.sum(X1**2,0)
            diagK2 = np.sum(X2**2,0)
        else:
            raise ValueError

    X1X2 = np.dot(X1, X2.T)

    if simple is True:
        return (2*X1X2 - np.outer(diagK1, j2) - np.outer(j1, diagK2))

    Q = (2*X1X2 - np.outer(diagK1, j2) - np.outer(j1, diagK2) )/ float((2*sigma**2))

    if is1d:
        return np.exp(Q[0,0])
    else:
        return np.exp(Q)
