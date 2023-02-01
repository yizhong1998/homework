import numpy as np


class LinearRegression:

    w: np.ndarray
    b: float

    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, y):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        coeffs = np.linalg.pinv(X.T @ X) @ X.T @ y
        self.b = coeffs[0]
        self.w = coeffs[1:]

    def predict(self, X):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return X @ np.hstack([self.b, self.w])


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def __init__(self):
        self.coeffs = None

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        y = y.reshape(-1, 1)
        self.coeffs = np.zeros((X.shape[1], 1))
        for i in range(epochs):
            y_pred = X @ self.coeffs
            er_pred = y - y_pred
            par_diff = -2 * (X.T @ er_pred) / X.shape[0]
            self.coeffs = self.coeffs - par_diff * lr

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        y_pred = X @ self.coeffs
        return y_pred
