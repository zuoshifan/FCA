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

def kurtosis(x):
    """Compute kurtosis c4 of a random variable x."""
    T = x.shape[0]
    c4 = np.sum(x**4) / T - 3 * (np.sum(x**2) / T)**2

    return c4

def Fhat_kurtosis(x):
    """The Fhat function for free kurtosis."""
    return -np.abs(kurtosis(x))

def negentropy(x):
    """Compute the engentropy of a random variable x."""
    T = x.shape[0]

    return (np.sum(x**3) / T)**2 / 12.0 + kurtosis(x) / 48.0

def Fhat_negentropy(x):
    """The Fhat function for engentropy."""
    return -negentropy(x)

def icf(Fhat, Z, nc=None, return_Fhat=False):
    """Independent Component Factorization (ICF) of an array of matrices Z."""
    # get centered vec(Z) such that zc = A xc = A Cxx^(1/2) xc^w
    zc = centering(Z)
    # get y = Vs^T xc^w, where A Cxx^(1/2) = Us \Sigma_s Vs^T
    y, es, Us = z2y(zc, nc)
    _, N, M = Z.shape
    s, NM = y.shape

    def cost(W):
        WTy = np.dot(W.T, y)
        return np.sum([ Fhat(WTy[i]) for i in range(s) ])

    # A solver that involves the hessian
    # solver = TrustRegions(mingradnorm=1e-8)
    solver = SteepestDescent(mingradnorm=1e-8)

    # O(s)
    manifold = Rotations(s, 1)

    # Solve the problem with pymanopt
    problem = Problem(manifold=manifold, cost=cost)
    # get What = Vs^T P S
    Wopt = solver.solve(problem)

    # get Ahat and xhat such that zc = Ahat xhat
    # get Ahat, which is actually = A Cxx^(1/2) P S
    Ahat = np.dot(Us*es, Wopt)
    # get xhat, which is actually = S^-1 P^-1 xc^w
    # xhat = np.dot(la.inv(Wopt), y)
    # Wopt is orthogonal, so Wopt.T = la.inv(Wopt)
    xhat = np.dot(Wopt.T, y)
    # assert np.allclose(zc, np.dot(Ahat, xhat)), 'Something may be wrong as zc != Ahat xhat'

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
