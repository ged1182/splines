import numpy as np


def characteristic_fn(x, a=0.0, b=None, include_left=False, include_right=False):
    """
    The characteristic function on an interval or at a point.
    :param x: array of shape (n,) : the values at which to compute the characteristic function values
    :param a: float : the left endpoint of the interval (default: 0.0)
    :param b: float or None: the right endpoint of the interval (default: None)
    :param include_left: bool : whether or not to include a in the interval (default: False)
    :param include_right: bool: whether or not to include b in the interval (default : False)
    :return:
    """
    if b is None:
        return np.array(x == 0.0, dtype=float)
    else:
        mask_left = (x >= a) if include_left else (x > a)
        mask_right = (x <= b) if include_right else (x < b)
        chi = np.array(mask_left * mask_right, dtype=float)
        return chi


def batch_kron(x, y):
    """
    compute Kronecker products in batch
    :param x: array of shape (n,a,b) or (n,a)
    :param y: array of shape (n,c,d) or (n,c)
    :return: kron_products : array of shape (n,a*c, b*d) or (n,a*c) : Kronecker products
    """
    assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
    assert x.shape[0] == y.shape[0]
    n = x.shape[0]

    kron_products = np.array(list(map(lambda i: np.kron(x[i, :], y[i, :]), range(n))))
    return kron_products