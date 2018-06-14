from tikypy.kernels import *


##############################
# some tests
##############################

def test_kernel_switching():
    xdata = np.random.rand(10,5)
    ydata = np.random.rand(6,5)
    param = np.random.randint(2,50)
    lz = lazy_kernel(xdata, ydata, kernel_type='gaussian')
    lz.update(param)
    # Update to multiquad
    K = multiquad_kernel(xdata, ydata=ydata, c=5.0)
    lz.update(5.0, kernel_type='multiquad')
    assert np.allclose(K, lz.kernel)
    # update to gaussian
    K = gaussian_kernel(xdata, ydata=ydata, sigma=10.0)
    lz.update(10.0, kernel_type='gaussian')
    assert np.allclose(K, lz.kernel)

    try:    # cannot switch to linear
        lz.update(10.0, kernel_type='linear')
        raise ValueError('Wrong update to linear')
    except SwitchError:
        pass

    try:    # cannot switch to inhomo poly
        lz.update(10.0, kernel_type='poly')
        raise ValueError('Wrong update to poly')
    except SwitchError:
        pass

    try:    # cannot switch to homo poly
        lz.update(10.0, kernel_type='polyhomo')
        raise ValueError('Wrong update to poly homo')
    except SwitchError:
        pass


def test_kernel_switching_innerprod():
    xdata = np.random.rand(10,5)
    ydata = np.random.rand(6,5)
    param = np.random.randint(2,50)
    lz = lazy_kernel(xdata, ydata)
    lz.update(param)
    # Update to homogeneous poly
    K = polyhomokern(xdata, ydata=ydata, powa=2.0)
    lz.update(2.0, kernel_type='polyhomo')
    assert np.allclose(K, lz.kernel)
    # update to inhomogenous poly
    K = polyinhomo(xdata, ydata=ydata, powa=3.0)
    lz.update(3.0, kernel_type='poly')
    assert np.allclose(K, lz.kernel)
    # update back to linear
    K = linear_kernel(xdata, ydata)
    lz.update(None, kernel_type='linear')
    assert np.allclose(K, lz.kernel)

    try:    # cannot switch to gaussian
        lz.update(10.0, kernel_type='gaussian')
        raise ValueError('Wrong update to gaussian')
    except SwitchError:
        pass

    try:    # cannot switch to multiquad
        lz.update(10.0, kernel_type='multiquad')
        raise ValueError('Wrong update to multiquad')
    except SwitchError:
        pass


def test_multiquad_kernel():
    xdata = np.random.rand(10,5)
    ydata = np.random.rand(6,5)
    param = np.random.randint(1,10)
    K = multiquad_kernel(xdata, ydata=ydata, c=param)
    lz = lazy_kernel(xdata, ydata, kernel_type='multiquad')
    lz.update(param)
    assert np.allclose(lz.kernel, K)
    # test update
    K = multiquad_kernel(xdata, ydata=ydata, c=param*2)
    lz = lazy_kernel(xdata, ydata, kernel_type='multiquad')
    lz.update(param*2)
    assert np.allclose(lz.kernel, K)


def test_linear_kernel():
    xdata = np.random.rand(10,5)
    ydata = np.random.rand(6,5)
    K = linear_kernel(xdata, ydata=ydata)
    lz = lazy_kernel(xdata, ydata, kernel_type='linear')
    assert np.allclose(K, lz.kernel)
    # default is linear
    lz = lazy_kernel(xdata, ydata)
    assert np.allclose(K, lz.kernel)
    # No effect
    lz.update(5.0)
    assert np.allclose(K, lz.kernel)
    lz.update(None)
    assert np.allclose(K, lz.kernel)


def test_polyhomo():
    xdata = np.random.rand(10,5)
    ydata = np.random.rand(6,5)
    param = np.random.randint(2,10)
    K = polyhomokern(xdata, ydata=ydata, powa=param)
    lz = lazy_kernel(xdata, ydata, kernel_type='polyhomo')
    lz.update(param)
    assert np.allclose(K, lz.kernel)
    # update
    K = polyhomokern(xdata, ydata=ydata, powa=param*2)
    lz.update(param*2)
    assert np.allclose(K, lz.kernel)
    # param = 1 is same as linear
    lz.update(1.0)
    assert np.allclose(linear_kernel(xdata, ydata), lz.kernel)


def test_polyinhomo():
    xdata = np.random.rand(10,5)
    ydata = np.random.rand(6,5)
    param = np.random.randint(2,10)
    K = polyinhomo(xdata, ydata=ydata, powa=param)
    lz = lazy_kernel(xdata, ydata, kernel_type='poly')
    lz.update(param)
    assert np.allclose(K, lz.kernel)
    # update
    K = polyinhomo(xdata, ydata=ydata, powa=param*2)
    lz.update(param*2)
    assert np.allclose(K, lz.kernel)
    # linear is approximately the same
    lz.update(1.0)
    assert np.allclose(linear_kernel(xdata, ydata) + 1, lz.kernel)


def test_gaussian_kernel():
    xdata = np.random.rand(10,5)
    ydata = np.random.rand(6,5)
    param = np.random.randint(2,50)
    K = gaussian_kernel(xdata, ydata=ydata, sigma=param)
    lz = lazy_kernel(xdata, ydata, kernel_type='gaussian')
    lz.update(param)
    assert np.allclose(K, lz.kernel)
    K = gaussian_kernel(xdata, ydata=ydata, sigma=param*2)
    lz.update(param*2)
    assert np.allclose(K, lz.kernel)


def test_vector_norm_sq():
    xdata = np.random.rand(10,5)
    ydata = np.random.rand(6,5)

    res = np.zeros((xdata.shape[0], ydata.shape[0]))
    for idx in range(xdata.shape[0]):
        for jdx in range(ydata.shape[0]):
            # d = np.sum((xdata[idx] - ydata[jdx])**2)
            d = np.linalg.norm(xdata[idx] - ydata[jdx])**2
            res[idx,jdx] = d
    Q = vector_norm_sq(xdata, ydata)
    assert np.allclose(Q, res)
