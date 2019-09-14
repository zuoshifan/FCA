import autograd.numpy as np
from pymanopt import Problem


def cost(X):
    return np.exp(-np.sum(X**2))

problem = Problem(manifold=manifold, cost=cost)
