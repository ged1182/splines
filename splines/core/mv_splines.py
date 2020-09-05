import numpy as np
from .uv_splines import b_spline_vector, b_spline_vector_grad
from .utils import batch_kron


def mv_b_spline_vector(x, knot_sequences, orders):
    """
    compute the values of the tensor-product multivariate B-splines
    :param x: array of shape (n,d) - the input features
    :param knot_sequences: list of array of length d
    :param orders: list of ints of length d
    :return: mv_b_spline : array of shape (n,j) :  the values of the tensor-product B-splines
    """
    assert isinstance(x, np.ndarray)
    # print(knot_sequences)
    if knot_sequences is None:
        assert len(orders) == x.shape[1]
    else:
        assert len(knot_sequences) == len(orders) == x.shape[1]
    q = len(orders)
    u = [None for _ in range(x.shape[1])] if knot_sequences is None else knot_sequences

    uv_b_splines = [b_spline_vector(x[:, i], u_, k) for i, (u_, k) in enumerate(zip(u, orders))]
    mv_b_spline = uv_b_splines[0]
    for i in range(1, q):
        mv_b_spline = batch_kron(mv_b_spline, uv_b_splines[i])
    return mv_b_spline


def mv_spline(x, knot_sequences, orders, c):
    """
    compute the values of a tensor-product multi-variate spline
    :param x: array of shape (n,d) - the values at which to compute the spline
    :param knot_sequences: list of array of length d : each item is an array containing the interior knots sequence
    of a dimension
    :param orders: list of ints of length d : each item is the order of the B-splines of a dimensions
    :param c: array of shape (card,) : the vector of coefficients of the basis functions
    :return: y_spline : array of shape (n,) : the values of the multivariate-spline function
    """
    mv_b_splines = mv_b_spline_vector(x, knot_sequences, orders)
    y_spline = mv_b_splines.dot(c)
    return y_spline


def mv_b_spline_grad(x, knot_sequences, orders):
    """
    compute the values of a tensor-product multi-variate spline
    :param x: array of shape (n,d) - the values at which to compute the spline
    :param knot_sequences: list of array of length d : each item is an array containing the interior knots sequence of
        a dimension
    :param orders: list of ints of length d : each item is the order of the B-splines of a dimensions
    :return: mv_grads, a list of length d containing arrays : the qth element of this list is an array of
        shape (n,card,p), where card is the dimension of the multi-variate tensor and p is the number of knots of the
        uni-variate spline in
    position q
    """
    n, d = x.shape
    p = np.array(list(map(lambda u: u.shape[0], knot_sequences)))
    # card = np.prod(p+orders)
    uv_b_splines = [np.expand_dims(b_spline_vector(x[:, i], np.array(u, dtype=float), k), axis=2) for i, (u, k) in
                    enumerate(zip(knot_sequences, orders))]
    uv_grads = [b_spline_vector_grad(x[:, i], np.array(u, dtype=float), k) for i, (u, k) in
                enumerate(zip(knot_sequences, orders))]
    mv_grads = []
    for dim in np.arange(d):
        if (p == 0)[dim]:
            grad = np.zeros((n, orders[dim], 0))
        else:
            temp = uv_b_splines.copy()
            temp[dim] = uv_grads[dim]
            grad = temp[0]
            for i in range(1, d):
                grad = batch_kron(grad, temp[i])
        mv_grads.append(grad)
    return mv_grads


def mv_spline_grad(x, knot_sequences, orders, c):
    """
    Compute the gradient of the multi-variate spline with respect to the knots
    :param x: array of shape (n,d) - the values at which to compute the spline
    :param knot_sequences: list of array of length d : each item is an array containing the interior knots sequence of
        a dimension
    :param orders: list of ints of length d : each item is the order of the B-splines of a dimensions
    :param c: array of shape (k+p,) : the vector of coefficients of the basis functions
    :return: d_mv_spline : list of arrays of shape(n, p_q) - the gradients
    """
    n, d = x.shape
    p = np.array([u.shape[0] for u in knot_sequences])
    db_mat = mv_b_spline_grad(x, knot_sequences, orders)
    c_ = c.reshape(-1, )

    d_mv_spline = [np.dot(np.transpose(db_mat[i], axes=(0, 2, 1)), c_) if (p > 0)[i] else np.array([])
                   for i in np.arange(d)]
    # ds = [ds_ for ds_ in ds]
    return d_mv_spline
