from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .core.optimizers import SGD, Adam
from .core.training_algorithms import fit_spline
from .core.uv_splines import b_spline_vector, spline
from .metrics import *

METRICS = {'r2': r2, 'mse': mse, 'rmse': lambda y, y_hat: np.sqrt(mse(y, y_hat))}
NO_HISTORY_MESSAGE = 'There is no history. Please train using the `fit` method with `optimize_knots=True`'


class UVSpline:
    """
    Free-Knot B-Spline Regression for Uni-Variate Data

    Args:
        order: int - the order of the B-fk-splines (default: 4)
            must be at least 2

        interval: list or tuple of length 2 - the interval of definition of the B-fk-splines (default: (0,1))
            must be in the form (a,b) or [a,b] where a < b

        knots: np.ndarray - (optional) the interior knots of the B-fk-splines
            must satisfy a < knots.min() <= knots.max() < b

        alpha: float - (optional) the regularization parameter to use in finding the coefficients with
            Ridge regression (default: None). When `None` ordinary least squares is used to find the coefficients.

        normalize: bool - whether or not to standardize the output before fitting (default: False)

    Properties:
        order: int - the order the B-fk-splines

        interval: tuple - the interval of definition

        knots: np.ndarray - the interior knots of the spline (B-fk-splines)

        alpha: float - the regularization parameter used in finding the optimal coefficients

        normalize: bool - whether or not to normalize the output

        knots_scaled: np.ndarray - the interior knots of the B-fk-splines scaled to lie in the interval (0,1)

        coefficients: np.ndarray - the coefficients of the B-fk-splines that define the Spline
            you can set this manually but is optimized during training using the `fit` method

        mse_train_history: np.ndarray - the mean-squared error history of the training data
            use this to visualize the learning curve of the training data

        mse_val_history: np.ndarray - the mean-squared error history of the training data
            use this to visualize the learning curve of the validation data

        knot_history: np.ndarray - a list of knot sequences of the course of the training
            use this to visualize the paths of each of the interior knots from their initial values

        coefficients_history: np.ndarray - a list of arrays of the coefficients during the course of the training

        r2_score_train_history: np.ndarray - the r-squared score history of the training data
            use this to visualize the performance curve of the training data

        r2_score_train_history: np.ndarray - the r-squared score history of the validation data
            use this to visualize the performance curve of the validation data

    """

    def __init__(self, order=4, interval=(0, 1), knots=None, alpha=None, normalize=False):

        self.__order = order
        self.__interval = interval
        self.__knots = knots
        self.__alpha = alpha
        self.__normalize = normalize

        self.__knots_scaled = None
        self.__coefficients = None
        self.__best_index_of_fit = 0
        self.__fit_history = {}
        self.__regressor = None

        self.__input_scaler = MinMaxScaler()
        self.__input_scaler.fit(np.array(interval).reshape(-1, 1))
        self.__output_scaler = StandardScaler()

    @property
    def order(self):
        return self.__order

    @order.setter
    def order(self, order):
        self.__order = order

    @property
    def interval(self):
        return self.__interval

    @interval.setter
    def interval(self, interval):
        assert len(interval) == 2 and interval[0] < interval[1]
        self.__interval = interval
        self.__input_scaler.fit(np.array(interval).reshape(-1, 1))
        if self.__knots:
            self.__knots_scaled = self.__input_scaler.transform(self.__knots.reshape(-1, 1)).reshape(-1, )

    @property
    def knots(self):
        return self.__knots

    @knots.setter
    def knots(self, knots):
        assert isinstance(knots, np.ndarray) and np.ndim(knots) == 1
        assert self.__interval[0] < knots.min() <= knots.max() < self.__interval[1]
        self.__knots = np.sort(knots)
        self.__knots_scaled = self.__input_scaler.transform(self.__knots.reshape(-1, 1)).reshape(-1, )

    @property
    def knots_scaled(self):
        return self.__knots_scaled

    @property
    def alpha(self):
        return self.__alpha

    @alpha.setter
    def alpha(self, alpha):
        assert isinstance(alpha, float) and alpha > 0
        self.__alpha = alpha

    @property
    def normalize(self):
        return self.__normalize

    @normalize.setter
    def normalize(self, normalize):
        assert isinstance(normalize, bool)
        self.__normalize = normalize

    @property
    def coefficients(self):
        return self.__coefficients

    @coefficients.setter
    def coefficients(self, coefficients):
        assert isinstance(coefficients, np.ndarray)
        assert coefficients.shape == (self.__order + self.__knots.shape[0],)
        self.__coefficients = coefficients

    @property
    def mse_train_history(self):
        if 'mse_train' in self.__fit_history.keys():
            return self.__fit_history['mse_train']
        else:
            print(NO_HISTORY_MESSAGE)

    @property
    def mse_val_history(self):
        if 'mse_val' in self.__fit_history.keys():
            return self.__fit_history['mse_val']
        else:
            print(NO_HISTORY_MESSAGE)

    @property
    def knot_history(self):
        if 'u' in self.__fit_history.keys():
            u = np.array(self.__fit_history['u'][0]).reshape(-1, self.__knots.shape[0])
            u_shape = u.shape
            u = self.__input_scaler.inverse_transform(u.reshape(-1,1)).reshape(u_shape)
            return u
        else:
            print(NO_HISTORY_MESSAGE)

    @property
    def coefficients_history(self):
        if 'c' in self.__fit_history.keys():
            return self.__fit_history['c']
        else:
            print(NO_HISTORY_MESSAGE)

    @property
    def r2_score_train_history(self):
        if 'r2_train' in self.__fit_history.keys():
            return self.__fit_history['r2_train']
        else:
            print(NO_HISTORY_MESSAGE)

    @property
    def r2_score_val_history(self):
        if 'r2_val' in self.__fit_history.keys():
            return self.__fit_history['r2_val']
        else:
            print(NO_HISTORY_MESSAGE)

    @property
    def best_index_of_fit(self):
        if len(self.__fit_history.keys()) > 0:
            return self.__best_index_of_fit

    def fit(self, x_train, y_train,
            x_val, y_val,
            optimize_knots=True, num_of_knots=None,
            optimizer='ADAM',
            max_epochs=100,
            lr=1e-3,
            lr_decay_rate=1.0,
            lr_decay_every=100,
            batch_size=None,
            patience=100,
            verbose=False):

        """
        Fit a Uni-Variate spline to the given data by optimizing the coefficients.
        If optimize_knots=True, also will find the interior knots given the number of knots `num_of_knots`.

        Args:
            x_train: np.ndarray of shape (n_train,) - the input training data

            y_train: np.ndarray of shape (n_train,) - the output training data

            x_val: np.array of shape (n_val,) - the input validation data

            y_val: np.ndarray of shape (n_val,) - the output validation data

            optimize_knots: bool - whether to optimize the knots (default: True)

            num_of_knots: int - (optional unless the initial knots aren't set for the UVSpline Regressor Object)
                the number of interior knots (default: None)

            optimizer: string - either 'SGD' for stochastic gradient descent of 'ADAM' for Adaptive Moment Estimation
                from the paper

            max_epochs: int - the maximum number of epochs (default: 100)

            lr: float - the learning rate for the optimizer (default: 1e-3)

            lr_decay_rate: float - the decay rate of the learning rate (default: 0.9)
                set this to 1.0 to keep the learning rate constant

            lr_decay_every: int - will decay the learning rate by `lr_decay_rate` every this many epochs (default: 100)

            batch_size: int - (optional) the batch size for the training data (default: None)
                None will do full-batch optimization.

            patience: int - the number of epochs to wait with no improvement in the validation loss before stopping
                the training.

        Returns: self
        """

        assert (num_of_knots is None and not (self.__knots_scaled is None)) \
               or (isinstance(num_of_knots, int) and num_of_knots > 0)
        x_all = np.concatenate([x_train, x_val], axis=-1).reshape(-1, 1)
        x_all = self.__input_scaler.fit_transform(x_all)
        x_train_scaled = x_all[:x_train.shape[0]].reshape(-1, )
        x_val_scaled = x_all[x_train.shape[0]:].reshape(-1, )
        y_all = np.concatenate([y_train, y_val], axis=-1).reshape(-1, 1)
        if self.__normalize:
            y_all = self.__output_scaler.fit_transform(y_all)
        y_train_scaled = y_all[:y_train.shape[0]].reshape(-1, )
        y_val_scaled = y_all[y_train.shape[0]:].reshape(-1, )

        if self.__alpha is None:
            self.__regressor = LinearRegression(fit_intercept=False)
        else:
            if isinstance(self.__alpha, float) or isinstance(self.__alpha, np.float):
                self.__regressor = Ridge(fit_intercept=False, alpha=self.__alpha)
            else:
                if isinstance(self.__alpha, np.ndarray):
                    self.__regressor = RidgeCV(fit_intercept=False, alphas=self.__alpha)

        if not optimize_knots:

            b_splines_train = b_spline_vector(x_train_scaled, self.__knots_scaled, self.order)

            self.__regressor.fit(b_splines_train, y_train_scaled)
            self.__coefficients = self.__regressor.coef_
        else:
            assert (optimizer in ['SGD', 'ADAM'])
            opt = SGD(lr=lr, lr_decay_rate=lr_decay_rate, lr_decay_every=lr_decay_every) if optimizer == 'SGD' \
                else Adam(p=(num_of_knots,), lr=lr, lr_decay_rate=lr_decay_rate, lr_decay_every=lr_decay_every)

            self.__best_index_of_fit, self.__regressor, self.__fit_history = fit_spline(x_train_scaled, y_train_scaled,
                                                                                        x_val_scaled, y_val_scaled,
                                                                                        regressor=self.__regressor,
                                                                                        optimizer=opt,
                                                                                        knot_init=self.__knots_scaled,
                                                                                        p=[num_of_knots],
                                                                                        k=[self.order],
                                                                                        max_iters=max_epochs,
                                                                                        batch_size=batch_size,
                                                                                        patience=patience,
                                                                                        verbose=verbose)
            self.__knots_scaled = self.__fit_history['u'][0][self.__best_index_of_fit]

            self.__knots = self.__input_scaler.inverse_transform(self.__knots_scaled.reshape(-1, 1)).reshape(-1, )
            self.__coefficients = self.__fit_history['c'][self.__best_index_of_fit]

    def predict(self, x, y=None, metrics=None):
        assert not(self.__coefficients is None)
        x_scaled = self.__input_scaler.transform(x.reshape(-1, 1)).reshape(-1, )
        assert (metrics is None) or np.all(np.array([m in METRICS.keys() for m in metrics], dtype=bool)) or \
               metrics == 'All'
        y_hat = spline(x_scaled, self.__knots_scaled, self.order, self.__coefficients)
        if self.__normalize:
            y_hat = self.__output_scaler.inverse_transform(y_hat.reshape(-1, 1)).reshape(-1, )
        if not (metrics is None) and not (y is None):
            if metrics == 'All':
                metrics_ = {m: METRICS[m](y, y_hat) for m in METRICS.keys()}
            else:
                metrics__ = np.array([metrics], dtype=object).reshape(-1, )
                metrics_ = {m: METRICS[m](y, y_hat) for m in metrics__}

            return y_hat, metrics_
        else:
            return y_hat

    def b_splines(self, x):
        """
        Compute the values of the Basis Splines at x

        Args:
            x: np.ndarray of shape (n,) the values at which to compute the B-fk-splines

        Returns: an array of shape (n, p+k) where n p is the number of interior knots and k is the order

        """

        x_scaled = self.__input_scaler.transform(x.reshape(-1,1)).reshape(-1,)
        y_scaled = b_spline_vector(x_scaled, self.__knots_scaled, self.order)
        if self.__normalize:
            y = self.__output_scaler.inverse_transform(y_scaled)
        else:
            y = y_scaled
        return y
