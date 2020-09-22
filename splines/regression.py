from .core.mv_splines import *
from .core.optimizers import SGD, Adam
from .core.training_algorithms import fit_spline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from .metrics import *

METRICS = {'r2': r2, 'mse': mse, 'rmse': lambda y, y_hat: np.sqrt(mse(y, y_hat))}
NO_HISTORY_MESSAGE = 'There is no history. Please train using the `fit` method with `optimize_knots=True`'


class SplineRegression:

    def __init__(self, num_of_vars=2, orders=None, intervals=((0, 1), (0, 1)), alpha=None):
        assert num_of_vars == len(orders) == len(intervals)
        if orders is None:
            orders = [2, 2]
        self.__num_of_vars = num_of_vars
        self.__orders = orders
        self.__intervals = intervals
        # self.__knots = None if knots is None else np.array(list(map(np.array, knots)))
        self.__alpha = alpha
        # self.__normalize = normalize

        self.__input_scaler = MinMaxScaler()
        self.__input_scaler.fit(np.array(intervals).T)
        # self.__output_scaler = StandardScaler()

        self.__knots_lengths = None if self.__knots is None else np.array(list(map(len, self.__knots)))
        self.__knots_scaled = self.__scale_knots()
        self.__coefficients = None
        self.__best_index_of_fit = 0
        self.__fit_history = {}
        if self.__alpha is None:
            self.regressor = LinearRegression(fit_intercept=False)
        else:
            if isinstance(self.__alpha, float) or isinstance(self.__alpha, np.float):
                self.regressor = Ridge(fit_intercept=False, alpha=self.__alpha)
            else:
                if isinstance(self.__alpha, np.ndarray):
                    self.regressor = RidgeCV(fit_intercept=False, alphas=self.__alpha)

    def __scale_knots(self):
        if self.__knots is None:
            return None
        if np.max(self.__knots_lengths) == 0:
            return self.__knots
        else:
            longest_knot_sequence = np.max(self.__knots_lengths)

            knots_padded_ = np.array(
                [np.concatenate([u, np.repeat(np.nan, longest_knot_sequence - len(u))]) for u in self.__knots]
            )

            knots_padded_ = self.__input_scaler.transform(knots_padded_.T).T
            return np.array([u[~np.isnan(u)] for u in knots_padded_], dtype=np.object)

    def __inverse_scale_knots(self):
        if self.__knots_scaled is None:
            return None
        self.__knots_lengths = np.array(list(map(len, self.__knots_scaled)))
        if np.max(np.array(list(map(len, self.__knots_scaled)))) == 0:
            return self.__knots_scaled
        else:
            longest_knot_sequence = np.max(self.__knots_lengths)

            knots_padded_ = np.array(
                [np.concatenate([u, np.repeat(np.nan, longest_knot_sequence - len(u))]) for u in self.__knots_scaled]
            )

            knots_padded_ = self.__input_scaler.inverse_transform(knots_padded_.T).T
            return np.array([u[~np.isnan(u)] for u in knots_padded_], dtype=np.object)

    @property
    def num_of_vars(self):
        return self.__num_of_vars

    @property
    def orders(self):
        return self.__orders

    @orders.setter
    def orders(self, orders):
        assert isinstance(orders, np.ndarray)
        assert orders.dtype in [int, np.int]
        assert np.all(orders >= 2)
        self.__orders = orders

    @property
    def intervals(self):
        return self.__intervals

    @intervals.setter
    def intervals(self, intervals):
        assert isinstance(intervals, tuple) or isinstance(intervals, np.ndarray) or isinstance(intervals, list)
        intervals_ = np.array(intervals, dtype=np.float)
        assert intervals_.shape == (2, self.__num_of_vars)
        assert np.all(intervals_.T[:, 0] < intervals_.T[:, 1])
        self.__input_scaler.fit(intervals_.T)
        self.__intervals = intervals

    @property
    def knots(self):
        if self.__knots is None:
            print('Knots are not set.')
        else:
            return self.__knots

    @knots.setter
    def knots(self, knots):
        assert isinstance(knots, np.ndarray)
        assert knots.shape[0] == self.__num_of_vars
        for i in range(self.__num_of_vars):
            if len(knots[i] > 0):
                assert np.all((self.__intervals[i][0] < knots[i]) * (knots[i] < self.__intervals[i][1]))

        self.__knots = knots
        self.__knots_lengths = np.array(list(map(len, self.__knots)))
        self.__knots_scaled = self.__scale_knots()

    @property
    def knots_scaled(self):
        if self.__knots_scaled is None:
            print('Knots are not set.')
        else:
            return self.__knots_scaled

    @property
    def alpha(self):
        return self.__alpha

    @alpha.setter
    def alpha(self, alpha):
        assert isinstance(alpha, float) or isinstance(alpha, np.float)
        assert alpha > 0
        self.__alpha = alpha

    @property
    def normalize(self):
        return self.__normalize

    @normalize.setter
    def normalize(self, normalize):
        assert isinstance(normalize, bool) or isinstance(normalize, np.bool)
        self.__normalize = normalize

    @property
    def coefficients(self):
        if self.__coefficients is None:
            print('Coefficients are not set.')
        else:
            return self.__coefficients

    @coefficients.setter
    def coefficients(self, coefficients):
        assert isinstance(coefficients, np.ndarray)
        p = 0 if self.__knots_lengths is None else np.array(self.__knots_lengths, dtype=np.int)
        assert coefficients.shape == np.product(np.array(self.__orders, dtype=np.int) + p)
        assert coefficients.dtype == np.float
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
    def knots_history(self):
        u_scaled = {d+1: np.array(self.__fit_history['u'][d+1]) for d in range(self.__num_of_vars)}
        knots_history = {d+1: u_scaled[d+1] * (self.__intervals[d][1]-self.__intervals[d][0]) + self.__intervals[d][0]
                         for d in range(self.__num_of_vars)}
        return knots_history


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
            x_val=None, y_val=None,
            optimize_knots=True, num_of_knots=None,
            optimizer='ADAM',
            max_epochs=100,
            lr=1e-3,
            lr_decay_rate=1.0,
            lr_decay_every=100,
            batch_size=None,
            patience=None,
            verbose=False):
        """

        Args:
            patience:
            batch_size:
            lr_decay_every:
            lr_decay_rate:
            lr:
            max_epochs:
            optimizer:
            optimize_knots:
            x_val:
            y_val:
            y_train:
            x_train:
            num_of_knots (np.ndarray): an array of integers of the knots lengths of each of the variables (dimensions)
        """
        if num_of_knots is None:
            num_of_knots = [1, 1]
        if x_val is None or y_val is None:
            x_val = x_train
            y_val = y_train

        x_train_scaled = self.__input_scaler.transform(x_train)
        x_val_scaled = self.__input_scaler.transform(x_val)

        if not optimize_knots:

            b_splines_train = self.b_splines(x_train)

            self.regressor.fit(b_splines_train, y_train)
            self.__coefficients = self.regressor.coef_
        else:
            assert (optimizer in ['SGD', 'ADAM'])
            opt = SGD(lr=lr, lr_decay_rate=lr_decay_rate, lr_decay_every=lr_decay_every) if optimizer == 'SGD' \
                else Adam(p=num_of_knots, lr=lr, lr_decay_rate=lr_decay_rate, lr_decay_every=lr_decay_every)
            self.__best_index_of_fit, self.regressor, self.__fit_history = fit_spline(x_train_scaled, y_train,
                                                                                      x_val_scaled, y_val,
                                                                                      regressor=self.regressor,
                                                                                      optimizer=opt,
                                                                                      knot_init=self.__knots_scaled,
                                                                                      p=num_of_knots,
                                                                                      k=self.orders,
                                                                                      max_iters=max_epochs,
                                                                                      batch_size=batch_size,
                                                                                      patience=patience,
                                                                                      verbose=verbose)
            self.__knots_scaled = [self.__fit_history['u'][d+1][self.__best_index_of_fit]
                                   for d in range(self.__num_of_vars)]
            self.__knots = self.__inverse_scale_knots()
            self.__coefficients = self.__fit_history['c'][self.__best_index_of_fit]

    def predict(self, x, y=None, metrics=None):
        assert not (self.__coefficients is None)
        x_scaled = self.__input_scaler.transform(x)
        assert (metrics is None) or metrics == 'All' or len(set(metrics).intersection(set(METRICS.keys()))) > 0

        y_hat = mv_spline(x_scaled, self.__knots_scaled, self.__orders, self.__coefficients)

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
            x: np.ndarray of shape (n, num_of_vars) the values at which to compute the B-splines

        Returns: an array of shape (n, p+k) where n p is the number of interior knots and k is the order

        """

        x_scaled = self.__input_scaler.transform(x)
        y = mv_b_spline_vector(x_scaled, self.__knots_scaled, self.__orders)
        return y
