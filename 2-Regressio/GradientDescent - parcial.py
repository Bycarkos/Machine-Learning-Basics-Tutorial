import numpy as np
import matplotlib.pylab as plt
from sklearn.metrics import mean_squared_error


class Regressor(object):
    def __init__(self, alpha=1e-2):
        # Inicialitzem w
        self.w = None
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None

    def __str__(self):
        return "Parametres Regressor \n\t bias (o intercept_): {} \n\t w (o coef_): {}".format(self.w[0], self.w[1:])

    def __initialize_w(self, n_vars):
        self.d = n_vars
        # diferents formes d'inicialitzar els pesos
        # self.w = np.random.random(n_vars)
        # self.w = np.random.normal(0, .1, size=n_vars)
        self.w = np.zeros(n_vars)

    def score(self, x, y):
        return self.__loss(self.predict(x), y)

    def predict(self, x):
        # implementar aqui la funció de prediccio
        return np.dot(x, self.w[1:]) + self.w[0]

    def __loss(self, h, y):
        # mse
        return np.mean((h-y)**2)

    def __update(self, x, y):
        # actualitzar aqui els pesos donada la x i la y real.
        h = self.predict(x)
        n_samples = len(y)

        # calcular gradient w..
        gradient = np.zeros(self.d)  #modificar aquesta linia

        # actualitzar w0 amb alpha
        self.w = self.w - self.alpha * gradient

        # retornar el error
        return self.score(x, y)

    def fit(self, x, y, max_iter=1000, epsilon=1e-5):
        if len(x.shape) > 1:
            n_vars = x.shape[1]
        else:
            raise AttributeError("Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.")

        y = y.ravel()

        self.__initialize_w(n_vars+1)

        current_error = self.score(x, y)
        last_error = current_error + epsilon * 10

        iter = 0
        # Entrenar durant max_iter iteracions o fins que la millora sigui inferior a epsilon
        while iter < max_iter and abs(last_error - current_error) > epsilon:
            last_error = current_error
            current_error = self.__update(x, y)
            iter += 1
            print("Iter {}: error {}".format(iter, current_error))

        print("Trained in {} iters: error {} ({:.7f} epsilon)".format(iter, current_error, abs(last_error - current_error)))
        self.coef_ = self.w[1:]
        self.intercept_ = self.w[0]

        pass

if __name__ == "__main__":
    # GENERATE DATA

    ns = 300  # samples
    nf = 2    # attributes

    # rng = np.random.RandomState(0)
    # X_sample = rng.randn(ns, nf)

    X1 = np.random.randn(ns, nf)  # Random points sampled from a univariate “normal” (Gaussian) distribution
    A = np.array([[0.6, .4], [.4, 0.6]])
    X2 = np.dot(X1, A)

    # VISUALIZE DATA
    # plt.plot(X2[:, 0], X2[:, 1], "o", alpha=0.3)  # alpha, blending value, between 0 (transparent) and 1 (opaque).
    # plt.show()

    X = X2[:, 0].reshape(-1, 1)
    Y = X2[:, 1]

    our_model = Regressor()
    our_model.fit(X, Y)
    y_our_pred = our_model.predict(X)

    print(our_model)
    print("ERROR IN MY MODEL:", mean_squared_error(Y, y_our_pred))

    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(X, Y)
    y_pred = model.predict(X)

    print("ERROR IN SKLEARN MODEL:", mean_squared_error(Y, y_pred))


    from sklearn.linear_model import SGDRegressor

    sgd_model = SGDRegressor()
    sgd_model.fit(X, Y)
    y_sgd_pred = sgd_model.predict(X)

    print("ERROR IN SKLEARN SGD MODEL:", mean_squared_error(Y, y_sgd_pred))

    sgd_model2 = SGDRegressor(loss='squared_loss', learning_rate='constant', eta0=1e-3, alpha=0)
    sgd_model2.fit(X, Y)
    y_sgd_sim_pred = sgd_model2.predict(X)

    print("ERROR IN SKLEARN SGD MODEL similar:", mean_squared_error(Y, y_sgd_sim_pred))
    print("COEF:", sgd_model2.coef_)
    print("INTERCEPT:", sgd_model2.intercept_)
