import numpy as np
from ..metrics import mse, mse_grad
from .mv_splines import mv_b_spline_vector, mv_spline_grad
from tqdm import trange
from scipy.sparse import coo_matrix


def fit_spline(x_train, y_train,
               x_val, y_val,
               regressor,
               optimizer,
               knot_init=None,
               p=None,
               k=None,
               max_iters=100,
               patience=25,
               batch_size=None,
               verbose=False
               ):

    num_of_vars = 1 if np.ndim(x_train) == 1 else x_train.shape[-1]
    if p is None and knot_init is None:
        p = np.array([1 for _ in range(num_of_vars)], dtype=np.int)
    if k is None:
        k = np.array([2 for _ in range(num_of_vars)], dtype=np.int)
    if knot_init is None:

        u = [np.array([i / (p_ + 1) for i in range(1, p_ + 1)]) for p_ in p]
    else:
        if not(knot_init is None):
            u = knot_init #if num_of_vars > 1 else [knot_init]

    u_history = {d+1: [np.array(u[d], dtype='float')] for d in range(num_of_vars)}

    b_splines_train = mv_b_spline_vector(x_train.reshape(-1, 1), u, k) if num_of_vars == 1 \
        else mv_b_spline_vector(x_train, u, k)

    regressor.fit(coo_matrix(b_splines_train), y_train)
    c = regressor.coef_
    c_history = [c]
    y_train_hat = regressor.predict(b_splines_train)
    b_splines_val = mv_b_spline_vector(x_val.reshape(-1, 1), u, k) if num_of_vars == 1\
        else mv_b_spline_vector(x_val, u, k)
    y_val_hat = regressor.predict(b_splines_val)
    mse_train = mse(y_train, y_train_hat)
    mse_val = mse(y_val, y_val_hat)
    r2_train = regressor.score(b_splines_train, y_train)
    r2_val = regressor.score(b_splines_val, y_val)
    mse_train_history = [mse_train]
    mse_val_history = [mse_val]
    r2_train_history = [r2_train]
    r2_val_history = [r2_val]

    history = {'u': u_history, 'c': c_history, 'mse_train': mse_train_history, 'mse_val': mse_val_history,
               'r2_train': r2_train_history, 'r2_val': r2_val_history}
    best_index = 0
    epochs_range = trange(max_iters) if verbose else range(max_iters)
    for i in epochs_range:
        if batch_size is None:
            index_batches = [range(x_train.shape[0])]

        else:
            num_of_complete_batches = x_train.shape[0] // batch_size
            shuffled_indices = np.random.permutation(x_train.shape[0])
            index_batches = [shuffled_indices[batch_size * i:batch_size * (i + 1)] for i in
                             range(num_of_complete_batches)]
            if batch_size * num_of_complete_batches < x_train.shape[0]:
                index_batches.append(shuffled_indices[batch_size * num_of_complete_batches:])
        for idx in index_batches:

            x = x_train[idx]
            y = y_train[idx]
            basis_splines = mv_b_spline_vector(x.reshape(-1, 1), u, k) if num_of_vars == 1\
                else mv_b_spline_vector(x, u, k)
            regressor.fit(coo_matrix(basis_splines), y)
            c = regressor.coef_
            y_hat = regressor.predict(basis_splines)
            dy = mv_spline_grad(x.reshape(-1, 1), u, k, c) if num_of_vars == 1\
                else mv_spline_grad(x, u, k, c)
            grad = mse_grad(y, y_hat, dy)

            u = optimizer.step(u, grad)

        b_splines_train = mv_b_spline_vector(x_train.reshape(-1, 1), u, k) if num_of_vars == 1\
            else mv_b_spline_vector(x_train, u, k)
        regressor.fit(coo_matrix(b_splines_train), y_train)
        c = regressor.coef_
        b_splines_val = mv_b_spline_vector(x_val.reshape(-1, 1), u, k) if num_of_vars == 1\
            else mv_b_spline_vector(x_val, u, k)
        y_val_hat = regressor.predict(b_splines_val)
        y_train_hat = regressor.predict(b_splines_train)
        mse_train = mse(y_train, y_train_hat)
        mse_val = mse(y_val, y_val_hat)
        r2_train = regressor.score(b_splines_train, y_train)
        r2_val = regressor.score(b_splines_val, y_val)

        for d in range(num_of_vars):
            history['u'][d+1].append(np.array(u[d], dtype='float'))
        history['c'].append(c)
        history['mse_train'].append(mse_train)
        history['mse_val'].append(mse_val)
        history['r2_train'].append(r2_train)
        history['r2_val'].append(r2_val)
        best_index = int(np.argmin(np.array(history['mse_val'])))

        # Early Stopping:
        if not (patience is None) and i >= patience and best_index <= i - patience:
            break

        u_best = [history['u'][d+1][best_index] for d in range(num_of_vars)]
        b_splines_train = mv_b_spline_vector(x_train.reshape(-1, 1), u_best, k) if num_of_vars == 1 \
            else mv_b_spline_vector(x_train, u_best, k)

        regressor.fit(coo_matrix(b_splines_train), y_train)

    return best_index, regressor, history
