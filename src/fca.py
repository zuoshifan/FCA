# import numpy as np
import autograd.numpy as np
from scipy import linalg as la

from pymanopt import Problem
# from pymanopt.solvers import TrustRegions
from pymanopt.solvers import SteepestDescent
from pymanopt.manifolds import Rotations


def matrix_cenntering(X, type='rectangular'):
    """"Centering a matrix X."""
    assert type in ('self-adjoint', 'rectangular'), 'Unknown matrix type'
    N, M = X.shape
    if type == 'self-adjoint':
        assert np.allclose(X, X.T.conj()), 'Matrix X is not self-adjoint'
        X_bar = np.trace(X) * np.eye(N) / N
    else:
        # X_bar = np.mean(X) * np.ones_like(X)
        X_bar = np.mean(X)# * np.ones_like(X)

    return X - X_bar

def centering(Z, type='rectangular'):
    """Centering an array of matrix Z."""
    if len(Z.shape) == 2:
        Z = Z[np.newaxis, :, :]
    s, N, M = Z.shape

    Zc = np.zeros_like(Z) # to save centered Z
    for i in range(s):
        Zc[i] = matrix_cenntering(Z[i], type)

    return Zc

def cov(Zc, type='rectangular'):
    """Compute covariance matrix of a centered Zc."""
    s, N, M = Zc.shape

    # # not efficient way
    # C = np.zeros((s, s), dtype=Zc.dtype)
    # for i in range(s):
    #     for j in range(s):
    #         C[i, j] = np.trace(np.dot(Zc[i], Zc[j].T.conj())) / N

    # more efficient way
    Z1 = Zc.reshape(s, -1)
    C = np.dot(Z1, Z1.T.conj()) / N

    return C.real # C should be real, as C = A A^T and A is real

def z2y(Zc, type='rectangular', nc=None):
    """Transform a centered z to y by truncate small eigen-values."""
    Czz = cov(Zc, type)
    e, U = la.eigh(Czz)
    if nc is None:
        nc = len(e[e>0.0])
    es = e[-nc:]**0.5
    Us = U[:, -nc:]
    Y = np.tensordot((Us/es).T, Zc, axes=(1, 0))
    # NOTE: we have Cyy = I, already whitened

    return Y, es, Us

# def free_whiten(Z, type='rectangular'):
#     if len(Z.shape) == 2:
#         Z = Z[np.newaxis, :, :]
#     s, N, M = Z.shape

#     Zc = np.zeros_like(Z) # to save centered Z
#     for i in range(s):
#         Zc[i] = matrix_cenntering(Z[i], type)

#     C = cov(Z)
#     e, U = la.eigh(C)
#     eh = e**0.5
#     Y = np.tensordot(np.dot(U/eh, U.T.conj()), Zc, axes=(1, 0))

#     return Y, eh, U

def free_kurtosis(X):
    """Compute free kurtosis k4 of a matrix X. This works for both self-adjoint and rectangular matrices."""
    N, M = X.shape
    # XXH = np.dot(X, X.T.conj())
    XXH = np.dot(X, np.conj(X.T))
    k4 = np.trace(np.dot(XXH, XXH)) / N - (1 + N / M) * (np.trace(XXH) / N)**2

    return k4

def Fhat_free_kurtosis(X, type='rectangular'):
    """The Fhat function for free kurtosis."""
    return -np.abs(free_kurtosis(X))

def free_entropy_selfadj(X):
    """Compute free entropy chi of a self-adjoint matrix X."""
    N, M = X.shape
    assert N == M and np.allclose(X, np.conj(X.T)), 'X is not a self-adjoint matrix'
    e, U = la.eigh(X)
    chi = np.sum([ np.log(np.abs(e[i] - e[j])) for i in range(N) for j in range(i+1, N) ]) / (N*(N-1))

    return chi

def free_entropy_rectangular(X):
    """Compute free entropy chi of a rectrangular matrix X."""
    N, M = X.shape
    # XXH = np.dot(X, X.T.conj())
    XXH = np.dot(X, np.conj(X.T))
    # e, U = la.eigh(XXH)
    e, U = np.linalg.eigh(XXH)
    a, b = 1.0*N / (N + M), 1.0*M / (N + M)
    sum_log_diff = np.sum([ np.log(np.abs(e[i] - e[j])) for i in range(N) for j in range(i+1, N) ])
    sum_log = np.sum(np.log(e))
    chi = (a**2 / N*(N - 1)) * sum_log_diff + ((b - a)*a / N) * sum_log

    return chi

def Fhat_free_entropy(X, type='rectangular'):
    """The Fhat function for free entropy."""
    if type == 'self-adjoint':
        return free_entropy_selfadj(X)
    elif type == 'rectangular':
        return free_entropy_rectangular(X)
    else:
        raise ValueError('Unkown matrix type: %s' % type)

def fcf(Fhat, Z, type='rectangular', nc=None, return_Fhat=False):
    """Free Component Factorization (FCF) of an array of matrices Z."""
    # get centered Z such that Zc = A Xc = A Cxx^(1/2) Xc^w
    Zc = centering(Z, type)
    # get Y = Vs^T Xc^w, where A Cxx^(1/2) = Us \Sigma_s Vs^T
    Y, es, Us = z2y(Zc, type, nc)
    s, N, M = Y.shape

    def cost(W):
        WTY = np.tensordot(W.T, Y, axes=(1, 0))
        return np.sum([ Fhat(WTY[i], type) for i in range(s) ])

    # A solver that involves the hessian
    # solver = TrustRegions(mingradnorm=1e-8)
    solver = SteepestDescent(mingradnorm=1e-8)

    # O(s)
    manifold = Rotations(s, 1)

    # Solve the problem with pymanopt
    problem = Problem(manifold=manifold, cost=cost)
    # get What = Vs^T P S
    Wopt = solver.solve(problem)

    # get Ahat and Xhat such that Zc = Ahat Xhat
    # get Ahat, which is actually = A Cxx^(1/2) P S
    Ahat = np.dot(Us*es, Wopt)
    # get Xhat, which is actually = S^-1 P^-1 Xc^w
    Xhat = np.tensordot(la.inv(Wopt), Y, axes=(1, 0))
    # assert np.allclose(Zc, np.tensordot(Ahat, Xhat, axes=(1, 0))), 'Something may be wrong as Zc != Ahat Xhat'

    # re=order Xhat and Ahat, from more non-Gaussian to more Gaussian
    Fhat_values = np.array([ Fhat(Xhat[i]) for i in range(s) ])
    inds = np.argsort(Fhat_values)
    Ahat = Ahat[:, inds]
    Xhat = Xhat[inds]
    # assert np.allclose(Zc, np.tensordot(Ahat, Xhat, axes=(1, 0))), 'Something may be wrong as Zc != Ahat Xhat'

    if return_Fhat:
        return Ahat, Xhat, Fhat_values[inds]

    return Ahat, Xhat
