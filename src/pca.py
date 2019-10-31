# import numpy as np
import autograd.numpy as np
from scipy import linalg as la

from pymanopt import Problem
# from pymanopt.solvers import TrustRegions
from pymanopt.solvers import SteepestDescent
from pymanopt.manifolds import Rotations


def centering(Z):
    """Centering an array of matrix Z."""
    if len(Z.shape) == 2:
        Z = Z[np.newaxis, :, :]
    s, N, M = Z.shape

    z = Z.reshape(s, -1) # vectorize the matrices
    zc = z - np.mean(z, axis=1)[:, np.newaxis]

    return zc

def cov(zc):
    """Compute covariance matrix of a centered zc."""
    s, NM = zc.shape
    C = np.dot(zc, zc.T.conj()) / NM

    return C.real # C should be real, as C = A A^T and A is real

def z2y(zc, nc=None):
    """Transform a centered z to y by truncate small eigen-values."""
    Czz = cov(zc)
    e, U = la.eigh(Czz)
    if nc is None:
        nc = len(e[e>0.0])
    es = e[-nc:]**0.5
    Us = U[:, -nc:]
    y = np.dot((Us/es).T, zc)
    # NOTE: we have Cyy = I, already whitened

    return y, es, Us

def variance(x):
    """Compute variance c4 of a random variable x."""
    T = x.shape[0]
    c4 = np.sum(x**2) / T - (np.sum(x) / T)**2

    return c4

def Fhat_variance(x):
    """The Fhat function for free variance."""
    return -variance(x)

def pcf(Fhat, Z, nc=None, return_Fhat=False):
    """Principal Component Factorization (PCF) of an array of matrices Z."""
    # get centered vec(Z) such that zc = A xc = A Cxx^(1/2) xc^w
    zc = centering(Z)

    # U, s, VT = la.svd(zc)
    # e, U = la.eigh(np.dot(zc, zc.T))
    e, U = la.eigh(cov(zc))
    if nc is None:
        nc = len(e[e>0.0])
    es = e[-nc:]**0.5
    Us = U[:, -nc:]
    Ahat = Us
    xhat = np.dot(Us.T, zc)
    # print variance(xhat[0]), variance(xhat[1])
    # print es
    s = len(e[e>0])
    _, N, M = Z.shape

    # re=order xhat and Ahat, from more non-Gaussian to more Gaussian
    Fhat_values = np.array([ Fhat(xhat[i]) for i in range(s) ])
    inds = np.argsort(Fhat_values)
    Ahat = Ahat[:, inds]
    xhat = xhat[inds]
    # assert np.allclose(zc, np.dot(Ahat, xhat)), 'Something may be wrong as zc != Ahat xhat'
    # reshape xhat to an array of matrices Xhat
    Xhat = xhat.reshape((s, N, M))

    if return_Fhat:
        return Ahat, Xhat, Fhat_values[inds]

    return Ahat, Xhat
