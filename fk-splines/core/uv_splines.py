import numpy as np
from .utils import characteristic_fn


def h_weight_vector(x, u, k):
    """
    Computes the values of the weight functions used to compute the kth order B-splines
    Inputs:
    x : array of shape (n,) : the values at which to compute the weight functions
    u : array of shape (p,) : the interior knots
    k : the order
    Output:
    h : ndarray of shape (n,k+p) the values of the (k+p-1) weight functions at x
    """
    assert k >= 2 and (isinstance(k, int) or isinstance(k, np.int32) or isinstance(k, np.int64))
    # n = x.shape[0]
    u = u if not(u is None) else np.array([], dtype=np.float)
    p = u.shape[0]

    u_tilde = np.sort(np.concatenate([np.repeat([0], k - 1), u], axis=0))

    u_tilde_tilde = np.sort(np.concatenate([np.repeat([1], k - 1), u_tilde], axis=0))
    diff_mat = np.zeros((k + p - 1, 2 * k + p - 2))
    for j in range(k + p - 1):
        diff_mat[j, j] = -1
        diff_mat[j, j + k - 1] = 1
    diff_u = diff_mat.dot(u_tilde_tilde)
    mask_repeated_knots = characteristic_fn(diff_u)
    multiplier = 1 - mask_repeated_knots
    divisor = diff_u + mask_repeated_knots

    h = np.expand_dims(x, axis=1) - u_tilde.reshape(1, -1)

    h = multiplier * h / divisor
    return h


def build_a_mat_and_a_tilde_mat(k, p):
    """
    Build the A and A_tilde matrices used in the computations
    :param k: int :  the order of the spline, at least 2
    :param p: int : the number of interior knots
    :return:
    """
    a_mat = np.zeros((k + p, k + p - 1))
    for i in range(1, a_mat.shape[0]):
        a_mat[i - 1, i - 1] = -1
        a_mat[i, i - 1] = 1

    # build "A tilde" matrix
    a_tilde_mat = np.zeros((k + p, k + p - 1))
    a_tilde_mat[0:k + p - 1, 0:k + p - 1] = np.eye(k + p - 1)
    return a_mat, a_tilde_mat


def b_spline_vector(x, u, k):
    """
    compute the values of the B-splines of order k with interior knots u
    :param x: array of shape (n,) : the values at which to compute the B-splines
    :param u: array of shape (p,) : the interior knots
    :param k: int : the order of the fk-splines
    :return: b: array of shape (n,k+p) : the values of the k+p fk-splines at x
    """
    assert k >= 1 and (isinstance(k, int) or isinstance(k, np.int32), isinstance(k, np.int64))

    n = x.shape[0]
    u = u if not(u is None) else np.array([], dtype=np.float)
    p = u.shape[0]
    # the complete knot sequence
    t = np.sort(np.concatenate([np.repeat([0, 1], k), u], axis=0))
    b = np.zeros((n, k + p))

    if k == 1:
        for i in range(k + p):
            right = True if i == k + p - 1 else False
            b[:, i] = characteristic_fn(x, t[i], t[i + 1], include_left=True, include_right=right)
    else:
        # build the "H" matrix

        h = np.array(h_weight_vector(x, u, k), dtype=np.float)

        h_mat = np.repeat(h.reshape((n, 1, k + p - 1)), k + p, axis=1)

        a_mat, a_tilde_mat = build_a_mat_and_a_tilde_mat(k, p)

        # order (k-1) b-splines
        b_k_minus_1 = b_spline_vector(x, u, k - 1)
        b_k_minus_1 = np.expand_dims(b_k_minus_1, 2)

        # order k b-splines

        b = np.matmul(h_mat * a_mat + np.expand_dims(a_tilde_mat, 0), b_k_minus_1)
        b = np.squeeze(b, 2)
    return b


def h_weight_vector_grad(x, u, k, freeze_knot_at_positions=None):
    """
    compute the gradient of the h_weight_vector functions order with respect to the knots u
    :param x: array of shape (n,) the values at which to evaluate the gradients
    :param u: array of shape (p,) the interior knots
    :param k: int : the order of the b-splines with which these weight functions are used for
    :param freeze_knot_at_positions:
    :return: dh : array of shape (n,k+p-1,p) : the gradients of the weight functions
    """
    assert k >= 2 and (isinstance(k, int) or isinstance(k, np.int32) or isinstance(k, np.int64))
    n = x.shape[0]
    u = u if not(u is None) else np.array([], dtype=np.float)
    p = u.shape[0]
    # u_prime = np.sort(np.concatenate([np.repeat([0], k - 1), u], axis=0))

    u_tilde = np.sort(np.concatenate([np.repeat([0, 1], k - 1), u], axis=0))
    diff_mat = np.zeros((k + p - 1, 2 * k + p - 2))
    for i in range(k + p - 1):
        diff_mat[i, i] = -1
        diff_mat[i, i + k - 1] = 1
    diff_u = diff_mat.dot(u_tilde)
    mask_repeated_knots = characteristic_fn(diff_u)
    multiplier = np.expand_dims(np.expand_dims(1 - mask_repeated_knots, axis=0), axis=2)
    divisor = np.expand_dims(np.expand_dims(diff_u + mask_repeated_knots, axis=0), axis=2)
    e_mat = np.zeros((n, k + p - 1, p))
    e_tilde_mat = np.zeros((n, k + p - 1, p))
    for i in range(p):
        e_mat[:, i + k - 1, i] = 1
        e_tilde_mat[:, i + k - 1, i] = 1
        e_tilde_mat[:, i, i] = -1

    h = h_weight_vector(x, u, k)

    dh = np.repeat(np.expand_dims(h, axis=2), p, axis=2) * e_tilde_mat - e_mat
    dh = multiplier * dh / divisor
    if not (freeze_knot_at_positions is None):
        assert isinstance(freeze_knot_at_positions, np.ndarray) and freeze_knot_at_positions.shape <= u.shape
        dh[:, :, freeze_knot_at_positions] = 0
    return dh


def b_spline_vector_grad(x, u, k, freeze_knot_at_positions=None):
    """
    compute the gradient of the b-splines of order k with respect to the knots u
    :param x: array of shape (n,) : the values at which to compute the gradients
    :param u: array of shape (p,) : the interior knots
    :param k: int : the order of the spline
    :param freeze_knot_at_positions: array of shape (s,) : the positions of the knots that should be frozen (s<=p)
    :return: du : array of shape (n,k+p,p) : the gradients values of the k+p fk-splines at x
    """
    assert k >= 2 and (isinstance(k, int) or isinstance(k, np.int32) or isinstance(k, np.int64))
    n = x.shape[0]
    if u is None:
        return None
    p = u.shape[0]
    if p == 0:
        return np.zeros((n, k, 0))
    db_mat = np.zeros((n, k + p, p))
    dh = h_weight_vector_grad(x, u, k, freeze_knot_at_positions=freeze_knot_at_positions)

    dh_mat = np.transpose(np.kron(np.transpose(dh, axes=(0, 2, 1)), np.ones((k + p, 1))), axes=(0, 2, 1))

    # build "A" matrix
    a_mat, _ = build_a_mat_and_a_tilde_mat(k, p)
    a_mat = np.expand_dims(a_mat, axis=0)
    b = b_spline_vector(x, u, k - 1)
    b = np.expand_dims(b, axis=1)

    for i in range(p):
        db_mat[:, :, i] = np.matmul(np.transpose(dh_mat[:, :, i * (k + p):(i + 1) * (k + p)], axes=(0, 2, 1)) * a_mat,
                                    np.transpose(b, axes=(0, 2, 1))).squeeze()
    if k > 2:
        a_mat, a_tilde_mat = build_a_mat_and_a_tilde_mat(k, p)
        # build the "H" matrix
        h = h_weight_vector(x, u, k)
        h_mat = np.repeat(h.reshape((n, 1, k + p - 1)), k + p, axis=1)

        # gradients of order (k-1) b-splines
        db_k_minus_1 = b_spline_vector_grad(x, u, k - 1, freeze_knot_at_positions=freeze_knot_at_positions)
        for i in range(p):
            db_mat[:, :, i] += np.matmul(h_mat * a_mat + np.expand_dims(a_tilde_mat, 0), db_k_minus_1)[:, :, i]

    return db_mat


def spline(x, u, k, c):
    """
    Compute the spline for inputs in x with knot sequence u order k and coefficients c
    :param x: array of shape (n,) : the values at which to compute the spline (the input features)
    :param u: array of shape (p,) : the knot sequence, must be ordered
    :param k: int : the order of the B-splines to use
    :param c: array of shape (k+p,) : the coefficients of the B-splines
    :return: y_spline : array of shape (n,) : the values of the spline function
    """
    u = u if not(u is None) else np.array([], dtype=np.float)
    b = b_spline_vector(x, u, k)
    y_spline = np.dot(b, c).squeeze()
    return y_spline


def spline_grad(x, u, k, c, freeze_knot_at_positions=None):
    """
    Compute the gradient of the spline with respect to the knots
    :param x: array of shape (n,) : the values at which to compute the gradients (the input features)
    :param u: array of shape (p,) : the knot sequence, must be ordered
    :param k: int : the order of the B-splines to use, must be at least 2
    :param c: array of shape (k+p,) : the vector of coefficients of the basis functions
    :param freeze_knot_at_positions: array of shape (s,) : the position of the knots in u to freeze (s<=p)
    :return: d_spline : array of shape(n, p) - the gradients
    """
    u = u if not(u is None) else np.array([], dtype=np.float)
    db_mat = b_spline_vector_grad(x, u, k, freeze_knot_at_positions=freeze_knot_at_positions)

    c_ = c.reshape(-1, 1)
    d_spline = np.matmul(np.transpose(db_mat, axes=(0, 2, 1)), c_)
    d_spline = d_spline.squeeze()
    return d_spline
