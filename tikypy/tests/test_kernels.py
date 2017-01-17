from tikypy.kernels import *

def test_gaukern_implimentations():
    xdata = np.random.random((20,4))
    ydata = np.random.random((10,4))
    resgaukern = gaukern(xdata.T, ydata=ydata.T)
    resfast_gaukern = fast_gaukern(xdata, ydata=ydata)
    resgaussian = gaussian_kernel(xdata, ydata=ydata)

    assert np.allclose(resgaukern, resfast_gaukern)
    assert np.allclose(resgaukern, resgaussian)
    assert np.allclose(resgaukern.shape, resfast_gaukern.shape)
    assert np.allclose(resgaukern.shape, resgaussian.shape)

    simplegaukern = gaukern(xdata.T)
    simplefast_gaukern = fast_gaukern(xdata)
    simplegaussian = gaussian_kernel(xdata)

    assert np.allclose(simplegaukern, simplefast_gaukern)
    assert np.allclose(simplegaukern, simplegaussian)
    assert np.allclose(simplegaukern.shape, simplefast_gaukern.shape)
    assert np.allclose(simplegaukern.shape, simplegaussian.shape)

    data = np.random.randint(1,5, size=(10,10)).astype(np.float)
    assert np.allclose(gaukern(data.T, sigma=3.0), fast_gaukern(data, sigma=3.0))
    assert np.allclose(gaukern(data.T, sigma=3.0), gaussian_kernel(data, sigma=3.0))
