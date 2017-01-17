import numpy as np

import utils

def test_fast_indexing():
    D = np.random.randn(1000, 1000)
    rows = np.random.randint(0, 1000, (400))
    cols = np.random.randint(0, 1000, (400))

    a = utils.fast_indexing(D, rows, cols)
    b = D[rows, :][:, cols]
    assert np.allclose(a, b)
    a = utils.fast_indexing(D, rows)
    b = D[rows, :]
    assert np.allclose(a, b)
    a = utils.fast_indexing(D.T, cols).T
    b = D[:, cols]
    assert np.allclose(a, b)




def hrf_default_basis(dt=2.0, duration=32):
    '''

    Returns
    --------
    hrf_basis (time-by-3)
    '''
    try:
        import hrf_estimation as he
    except ImportError, e:
        raise(e)

    time = np.arange(0, duration, dt)
    h1 = he.hrf.spm_hrf_compat(time)
    h2 = he.hrf.dspmt(time)
    h3 = he.hrf.ddspmt(time)

    hrf_basis = np.c_[h1, h2, h3]
    return hrf_basis
