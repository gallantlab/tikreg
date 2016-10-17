import numpy as np
import itertools

from scipy.linalg import toeplitz
from scipy.misc import comb


def difference_operator(order, nobs):
    '''Get a finite difference operator matrix of size `nobs`.

    Parameters
    ----------
    order : int
        The order of the derivative (e.g. 2nd derivative)
    nobs : int
        The size of the output matrix

    Returns
    -------
    mat : (`nobs`,`nobs`) np.ndarray
    '''

    depth = order + 1
    kernel = np.asarray([comb(depth-1, idx) for idx in xrange(depth)]) # pascal triangle row
    sign = (-1)**np.arange(len(kernel))
    kernel *= sign
    vec = np.zeros(nobs)
    if order % 2 == 0:
        lkern = len(kernel)/2
        vec[:len(kernel)-lkern] = kernel[lkern:]
        convmat = toeplitz(vec, np.zeros(nobs))
        convmat += np.tril(convmat, -1).T
    elif order == 1:
        vec[:len(kernel)] = kernel
        convmat = toeplitz(vec)
    else:
        raise NotImplementedError
    return convmat



def banded_angles(*models):#,
    offset=0.0
    nangles=6#21#):
    nmodels = len(models)
    print nmodels

    angle = np.linspace(0,90,6)#0+offset, 90 - offset, nangles)
    print angle

    angle = np.deg2rad(angle)
    alpha1 = np.sin(angle)
    alpha2 = np.cos(angle)
    print np.round(alpha1, 2)
    print np.round(alpha2, 2)

    angle1 = np.deg2rad(np.linspace(0,90,6))
    angle2 = np.deg2rad(np.linspace(0,90,6))
    alpha1 = np.sin(angle1)*np.cos(0.0)
    alpha2 = np.sin(angle1)*np.sin(0.0)
    alpha3 = np.cos(angle1)
    print np.round(alpha1, 2)
    print np.round(alpha2, 2)
    print np.round(alpha3, 2)
    return


def spherical_coordinates_n2(offset=0, nsamples=10, spacing=np.linspace):
    nangles = 1
    space = spacing(offset, 90 - offset, nsamples)
    angles = np.asarray(list(itertools.product(space, repeat=nangles))).T
    angles = np.deg2rad(angles)
    return np.cos(angles[0]), np.sin(angles[0])


def spherical_coordinates_n3(offset=0, nsamples=10, spacing=np.linspace):
    nangles = 2
    space = spacing(offset, 90 - offset, nsamples)
    angles = np.asarray(list(itertools.product(space, repeat=nangles))).T
    angles = np.deg2rad(angles)
    x1  = np.cos(angles[0])
    x2 = np.sin(angles[0])*np.cos(angles[1])
    x3 = np.sin(angles[0])*np.sin(angles[1])
    coords = np.asarray([x1,x2,x3])
    return coords


def simple_polar2cartesian(angle, radius=1.0):
    '''Given some polar coordinates, return the cartesian coordinates.

    Parameters
    ----------
    angle : vector (p,)
        The angles should be given in radians
    radius: float-like
        The n-sphere radius

    Returns
    -------
    coords : vector, (p+1,)
        The cartesian coordinate of the points (x_n,...,x_1)
    '''
    angles = np.asarray(angle)
    nmodels = len(angle) + 1

    angles = np.atleast_2d(angles).T    # angle by 1
    # append zeros angle
    angles = np.r_[angles, np.zeros((1,angles.shape[-1]))]
    x0 = np.cos(angles[0])*radius
    coords = [x0]

    for xdx in xrange(1,nmodels-1):
        angle = angles[xdx]
        xi = np.cos(angle)*radius

        if xdx-1 == 0:
            angles_prev = angles[xdx-1]
            xprev = np.sin(angles_prev)
        else:
            angles_prev = angles[:xdx-1]
            xprev = reduce(lambda x,y: np.sin(x)*np.sin(y), angles_prev)

        x = xi*xprev
        coords.append(x)

    if angles[:-1].shape[0] == 1:
        xn = np.sin(angles[0])
    else:
        xn = reduce(lambda x,y: np.sin(x)*np.sin(y), angles[:-1])

    coords.append(xn*radius)
    coords = np.asarray(coords)
    return coords


def sample_uniform_hypersphere(ndimensions, nsamples=10):
    '''
    S2 has a solution:
    http://mathworld.wolfram.com/SpherePointPicking.html

    S3 has a solution:
    http://mathworld.wolfram.com/HyperspherePointPicking.html

    No general solution for higher dimensions
    '''
    raise ValueError


def sample_uniform_sphere(nsamples=10):
    '''
    S2 has a solution:
    http://mathworld.wolfram.com/SpherePointPicking.html
    '''
    # asymuth for octant sphere (0, pi/2), instead of 0,2pi
    asymuth_samples = np.linspace(0,1./4.,nsamples)
    # zenith for octant sphere (0, pi/2) # instead of 0,pi
    zenith_samples = np.linspace(1./2.,1.,nsamples)
    u,v = np.asarray(list(itertools.product(asymuth_samples,zenith_samples))).T
    t1, t2 = 2*np.pi*u, np.arccos(2*v - 1)
    # t1, t2 = 2*np.pi*u, np.arccos(v - 1)
    return np.rad2deg(t2), np.rad2deg(t1)


def sample_spherical_polar(ndimensions, offset=1, nsamples=10, max_angle=90., spacing=np.linspace):
    '''
    '''
    space = spacing(offset, max_angle - offset, nsamples, endpoint=True)

    angles = np.asarray(list(itertools.product(space, repeat=ndimensions-1))).T
    angles = np.deg2rad(angles)
    return angles


def polar2cartesian(angles, radius=1.0, physics_convention=False):
    '''Convert a set of angles of a sphere defined in n-dimensional
    space to cartesian coordinates.

    Parameters
    ----------
    angles : np.ndarray (n-1, k)
        The `k` angles in `n` dimensional space to convert
        from polar to cartesian coordinates.

    raw : bool
        Use hypersphere standard form. This differs from the
        standard 3-dimensional sphere in axis orientation.

    Returns
    -------
    coords : np.ndarray (n, k)
        All the x_i coordinates corresponding to the `k` angles

    Notes
    -----
    theta_1 is angle away from x_{n} (inclination towards x_{n-1}) in [0,pi]
    theta_2 is angle away from x_{n-1} (inclination towards x_{n-2}) in [0,pi]
    ...etc
    theta_{n-1} is the angle between x_1 and x_2 (asymuth) in [0,2pi]


    https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
    '''
    ndimensions = angles.shape[0]+1
    angles = np.r_[angles, np.zeros((1,angles.shape[-1]))]

    x0 = np.cos(angles[0])*radius
    coords = [x0]
    for xdx in xrange(1,ndimensions-1):
        angle = angles[xdx]
        xi = np.cos(angle)*radius

        if xdx-1 == 0:
            angles_prev = angles[xdx-1]
            xprev = np.sin(angles_prev)
        else:
            angles_prev = angles[:xdx-1]
            xprev = reduce(lambda x,y: np.sin(x)*np.sin(y), angles_prev)

        x = xi*xprev
        coords.append(x)

    # for some reason, there's some weirdness going on with np.sin in this context...
    if angles[:-1].shape[0] == 1:
        xn = np.sin(angles[0])
    else:
        xn = reduce(lambda x,y: np.sin(x)*np.sin(y), angles[:-1])

    coords.append(xn*radius)
    coords = np.asarray(coords)
    if physics_convention is True:
        # untested:
        # used to work for 3D... who knows about nD.
        coords = np.roll(coords, 2, axis=0)
    return coords[::-1]


def cartesian2polar(ratios):
    '''
    ratios : np.ndarray (k, n):
        `k` coordinates in `n` dimensional space

    Returns
    -------
    angles : np.ndarray (k, n-1)
        Poolar angles defining the ratios

    [x_1, ..., x_n]
    phi_i is angle
    '''
    all_angles = []

    for cartesian in ratios:
        radius = np.sqrt(np.sum(cartesian**2))
        angles = []

        # easier to reverse the thing for indexing
        reversed_cartesian = cartesian[::-1]
        for xdx in xrange(len(cartesian)-1):
            phi_i = np.arccos(reversed_cartesian[xdx]/np.sqrt(np.sum(reversed_cartesian[xdx:]**2)))
            angles.append(phi_i)

        # phi_1, ..., phi_{n-1}
        all_angles.append(angles)
    all_angles = np.rad2deg(np.asarray(all_angles))
    return all_angles



def _sample_spherical_coordinates(ndimensions, offset=1, nsamples=10, spacing=np.linspace):
    '''Get ``nsamples`` from each (n-1) angle in an n-dimensional sphere and convert them
    to cartesian coordinates.

    Parameters
    ----------
    ndimensions : int, (p,)
        The dimensionality of the sphere
    nsamples : int, (n,)
        The number of samples to take along each angle
    spacing : function
        The function to use when sampling the angles. Defaults
        to np.linspace, can also use np.logspace
    offset : int, (degrees)
        The first angle that will be taken. It will also constrained
        the last angle sampled (i.e. 90 - offset).

    Returns
    -------
    coords : np.ndarray (p, n^(p-1))
        All the x_i coordinates corresponding to the sampled angles

    Notes
    -----
    https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
    '''
    radius = 1.0
    space = spacing(offset, 90. - offset, nsamples)
    angles = np.asarray(list(itertools.product(space, repeat=ndimensions-1))).T
    angles = np.deg2rad(angles)
    angles = np.r_[angles, np.zeros((1,angles.shape[-1]))]

    x0 = np.cos(angles[0])*radius
    coords = [x0]
    for xdx in xrange(1,ndimensions-1):
        angle = angles[xdx]
        xi = np.cos(angle)*radius

        if xdx-1 == 0:
            angles_prev = angles[xdx-1]
            xprev = np.sin(angles_prev)
        else:
            angles_prev = angles[:xdx-1]
            xprev = reduce(lambda x,y: np.sin(x)*np.sin(y), angles_prev)

        x = xi*xprev
        coords.append(x)

    # for some reason, there's some weirdness going on with np.sin in this context...
    if angles[:-1].shape[0] == 1:
        xn = np.sin(angles[0])
    else:
        xn = reduce(lambda x,y: np.sin(x)*np.sin(y), angles[:-1])

    coords.append(xn*radius)
    coords = np.asarray(coords)
    # standard format is reversed (this returns x_n, ..., x_2, x_1)
    return coords


def simple_sphere_coord(radius=1., theta1=45., theta2=45.):
    theta1 = np.deg2rad(theta1)
    theta2 = np.deg2rad(theta2)
    # x = radius*np.cos(theta1)*np.sin(theta2)
    # y = radius*np.sin(theta1)*np.sin(theta2)
    # z = radius*np.cos(theta2)
    x = radius*np.cos(theta1)
    y = radius*np.sin(theta1)*np.cos(theta2)
    z = radius*np.sin(theta1)*np.sin(theta2)
    return x,y,z

def standard_sphere_coord(angle, radius=1.0):#theta1=45., theta2=45., radius=1.):
    '''
    standard physics way
    theta1: inclination (angle between z and x/y
    theta2: asymuth (angle on x-y plane)
    '''
    theta1, theta2 = np.deg2rad(angle)
    # theta1 = np.deg2rad(theta1)
    # theta2 = np.deg2rad(theta2)
    x = radius*np.sin(theta1)*np.cos(theta2)
    y = radius*np.sin(theta1)*np.sin(theta2)
    z = radius*np.cos(theta1)
    return x,y,z


def simple_sphere_angle(x,y,z):
    coords = np.asarray([x,y,z])
    radius = ((coords**2).sum())**0.5
    theta1 = np.arctan(y/x)
    theta2 = np.arccos(z/(radius))
    return radius, theta1, theta2


def test_spherical_coords():
    v3d = spherical_coordinates_n3(nsamples=3, offset=1)
    v2d = spherical_coordinates_n2(nsamples=3, offset=1)
    threed = sample_spherical_coordinates(3, nsamples=3, offset=1.)
    twod = sample_spherical_coordinates(2, nsamples=3, offset=1.)
    assert np.allclose(v3d, threed)
    assert np.allclose(v2d, twod)

    # Test 3D
    angles = np.asarray([45., 45.])
    angles = np.deg2rad(angles)
    x1  = np.cos(angles[0])
    x2 = np.sin(angles[0])*np.cos(angles[1])
    x3 = np.sin(angles[0])*np.sin(angles[1])
    coords = np.asarray([x1,x2,x3])

    z, y, x = coords
    c = (x**2 + y**2)**0.5
    assert np.allclose(c, z)
    d = (c**2 + np.sin(np.deg2rad(45.))**2)**0.5
    assert np.allclose(d, 1.)
    xx,yy,zz = simple_sphere_coord(radius=1.0, theta1=45.0, theta2=45.0)
    assert np.alltrue([np.allclose(xx,x), np.allclose(yy,y), np.allclose(zz,z)])

    coords = simple_sphere_coord(radius=1.0, theta1=50.0, theta2=10.0)
    simple_xyz = np.asarray(coords)
    xyz = polar2cartesian(np.deg2rad([10.,50.])).squeeze()
    print simple_xyz
    print xyz

    coords = simple_sphere_coord(radius=1.0, theta1=0.0, theta2=10.0)
    simple_xyz = np.asarray(coords)
    xyz = polar2cartesian(np.deg2rad([10.,0.])).squeeze()
    print simple_xyz
    print xyz


if 0:

    offset = 1.0
    angle = np.linspace(0+offset, 90 - offset, 21)
    angle = np.deg2rad(angle)

    alpha1 = np.sin(angle)
    alpha2 = np.cos(angle)
    alphas = zip(alpha1, alpha2)
    ratios = alpha1/alpha2


def show_spherical_angles(theta1=30., theta2=60., physics_convention=False):
    '''Draw a vector on the unit sphere defined by the angles
    theta1 (inclination on last axis, x_3, x_3->x1) and theta2 (asymuth x_1->x_2 plane).
    '''
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np
    from itertools import product, combinations

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,projection='3d')
    ax.set_aspect("equal")

    # draw cube
    if 0:
        r = [-1, 1]
        for s, e in combinations(np.array(list(product(r,r,r))), 2):
            if np.sum(np.abs(s-e)) == r[1]-r[0]:
                ax.plot3D(*zip(s,e), color="k")

    # draw sphere
    upsample = 2.
    # u, v = np.mgrid[0:2*np.pi:20j*upsample, 0:np.pi:10j*upsample]
    u, v = np.mgrid[0:np.pi/2.:20j*upsample, 0:np.pi/2.:10j*upsample]
    x=np.cos(u)*np.sin(v)
    y=np.sin(u)*np.sin(v)
    z=np.cos(v)
    ax.plot_wireframe(x, y, z, color="k", alpha=0.2)

    #draw a vector
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d

    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
            FancyArrowPatch.draw(self, renderer)

    if np.isscalar(theta1) and np.isscalar(theta2):
        coords = polar2cartesian(np.deg2rad(np.asarray([theta1,theta2])[...,None]), physics_convention=physics_convention)
        # coords = polar2cartesian(np.deg2rad(np.asarray([theta1,theta2])[...,None]))
    else:
        mangles = np.vstack([theta1, theta2])
        # coords = np.asarray([standard_sphere_coord(angle) for angle in mangles.T]).T
        coords = polar2cartesian(np.deg2rad(mangles), physics_convention=physics_convention)

    for idx, xyz in enumerate(coords.T):
        if 0: print xyz
        x,y,z = xyz
        a = Arrow3D([0,x],[0,y],[0,z], mutation_scale=20, lw=1, arrowstyle="-|>", color="r")
        ax.add_artist(a)
        plt.show()
        if np.isscalar(theta1) and np.isscalar(theta2):
            ax.text(x,y,z, r'$\theta_1=%0.f, \theta_2=%0.f$' % (theta1, theta2))
        else:
            ax.text(x,y,z, r'$\theta_1=%0.f, \theta_2=%0.f$' % (theta1[idx], theta2[idx]))
    # axes
    angles = np.deg2rad([[0.0, 90.0],
                         [90.00, 90.0],
                         [45.00, 90.0],
                         [90., 45.],
                         [90., 90.],
                         [90., 0.0],
                         ]).T
    coords = polar2cartesian(angles)

    for idx, xyz in enumerate(coords.T):
        x,y,z = xyz
        if 0: print xyz
        a = Arrow3D([0,x],[0,y],[0,z], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
        ax.add_artist(a)
        plt.show()
        # theta1, theta2 = tuple(np.rad2deg(angles[:,idx]))
        # ax.text(x,y,z, r'$\theta_1=%0.f, \theta_2=%0.f$' % (theta1, theta2))

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')
    ax.view_init(elev=90, azim=-90)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_zlim(0,1)
    return ax


if 0:
    # https://en.wikipedia.org/wiki/3-sphere#Hyperspherical_coordinates
    psi, theta, phi = 30., 40., 60., # phi is asymuth
    angles = np.asarray([psi, theta, phi])[...,None]
    radius = np.sin(np.deg2rad(psi))
    sphere_angles = np.asarray([theta, phi])[...,None]
    coords_3sphere = polar2cartesian(np.deg2rad(angles), physics_convention=True)
    coords_2sphere = polar2cartesian(np.deg2rad(sphere_angles), radius=radius, physics_convention=True)
    print coords_3sphere
    print coords_2sphere
    x0, x1, x2, x3 = coords_3sphere
    x,y,z = coords_2sphere
    rad,t1,t2 = simple_sphere_angle(x,y,z)

    print radius
    x,y,z = simple_sphere_coord(theta1=phi, theta2=theta, radius=radius)
    rad,t1,t2 = simple_sphere_angle(x,y,z)
    print rad
    print (np.asarray([x0,x1,x2])**2).sum()**0.5






if __name__ == '__main__':

    angles = np.deg2rad([[50.0, 10.],
                         # [50.0, 50.],
                         # [50.0, 70.],
                         # [50.0, 89.],
                         ]).T
    coords = polar2cartesian(angles)
    # angles = sample_spherical_polar(3, nsamples=3)

    # test points
    ##############################
    ratios = np.asarray([[1.0, 1.0, 1.0],
                         [1.0, 1.0, 0.0],
                         [1.0, 0.0, 1.0],
                         [0.0, 1.0, 1.0],
                         [1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0],
                         ])
    angles = np.nan_to_num(cartesian2polar(ratios))
    show_spherical_angles(angles[:,0], angles[:,1])

    # log points
    ##############################
    import itertools

    ridges = np.logspace(0,1,5)#np.logspace(0,3,5)
    ratios = np.asarray(list(itertools.product(*[ridges]*3)))
    print ratios.shape

    angles = np.nan_to_num(cartesian2polar(ratios))
    show_spherical_angles(angles[:,0], angles[:,1])


    # linear points
    ##############################
    ridges = np.linspace(0,1,5)#np.logspace(0,3,5)
    ratios = np.asarray(list(itertools.product(*[ridges]*3)))
    print ratios.shape

    angles = np.nan_to_num(cartesian2polar(ratios))
    show_spherical_angles(angles[:,0], angles[:,1])

    # BUG: does not sample near top well =S
    angles = np.asarray(sample_uniform_sphere(nsamples=10))
    show_spherical_angles(angles[0], angles[1])


    # for idx, xyz in enumerate(coords.T):
    #     if 0: print xyz
    #     x,y,z = xyz
    #     a = Arrow3D([0,x],[0,y],[0,z], mutation_scale=20, lw=1, arrowstyle="-|>", color="r")
    #     ax.add_artist(a)
    #     plt.show()
    #     theta1, theta2 = tuple(np.rad2deg(angles[:,idx]))
    #     ax.text(x,y,z, r'$\theta_1=%0.f, \theta_2=%0.f$' % (theta1, theta2))

    # ax.set_zlabel('Z')
    # ax.set_ylabel('Y')
    # ax.set_xlabel('X')
