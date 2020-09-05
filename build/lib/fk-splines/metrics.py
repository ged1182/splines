import numpy as np


def mse(y, y_hat):
    """
    Compute the mean-squared-error between y and y_hat
    :param y: array of shape (n,) : the target values
    :param y_hat: array of shape (n,) - the predicted values
    :return:
    """
    loss = np.mean((y - y_hat) ** 2)
    return loss


def mse_grad(y, y_hat, d_y_hat):
    """
    Compute the gradient of the mean-squared-error
    :param y: array of shape (n,) : the true targets
    :param y_hat: array of shape (n,) : the predicted targets
    :param d_y_hat: array of shape (n,) or a list of arrays: the gradient of the predicted targets
    :return: loss_grad : array of shape (1,) or list of arrays : the gradient of the mse loss
    """
    n = y.shape[0]
    error = (y_hat - y) / n

    if isinstance(d_y_hat, list):
        # print([d_y_hat_.shape for d_y_hat_ in d_y_hat])
        loss_grad = [np.dot(error.T, d_y_hat_) if d_y_hat_.shape[0] > 0 else np.array([]) for d_y_hat_ in d_y_hat]
        return loss_grad
    else:
        loss_grad = np.dot(error.T, d_y_hat)
        return loss_grad


def r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    compute the r2 score
    :param y: array of shape (n,) : the true targets
    :param y_hat: array of shape (n,) : the predicted targets
    :return: r2_value : float : the r2 score
    """
    n = y.shape[0]
    sse = n * mse(y, y_hat)
    y_mean = np.mean(y)
    sst = np.sum((y - y_mean)**2)
    r2_value = 1 - sse / sst
    return r2_value


def bic(n, mse_value, num_parameters):
    """

    :param n: int : number of data points
    :param mse_value: float : the mean-squared-error
    :param num_parameters: int : the number of free parameters of the model
    :return: bic_score : float : the bayes information criterion score:
    n * np.log(sse) + num_parameters * np.log(n)
    """
    bic_score = n * np.log(n * mse_value) + np.log(n) * num_parameters
    return bic_score
