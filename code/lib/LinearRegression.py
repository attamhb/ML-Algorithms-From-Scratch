import numpy as np

class LinearRegression:

    def __init__(self, learning_rate=0.001, num_iterations=1000, criterion='gd'):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.criterion = criterion
        self.coefficients = None
        self.intercept = None

    def fit(self, features, target):
        num_samples, num_features = features.shape
        self.coefficients = np.zeros(num_features)
        self.intercept = 0

        if self.criterion == 'gd':
            self._gradient_descent(features, target, num_samples)
        elif self.criterion == 'sgd':
            self._stochastic_gradient_descent(features, target)
        elif self.criterion == 'normal_eq':
            self._normal_equation(features, target)
        elif self.criterion == 'pseudoinverse':
            self._pseudo_inverse(features, target)
        else:
            raise ValueError("Invalid criterion. Choose from 'gd', 'sgd', 'normal_eq', 'pseudoinverse'.")

    def _gradient_descent(self, features, target, num_samples):
        for _ in range(self.num_iterations):
            predictions = np.dot(features, self.coefficients) + self.intercept
            gradients_weights, gradients_bias = self.compute_gradients(features, target, predictions)
            self._update_parameters(gradients_weights, gradients_bias)

    def _update_parameters(self, gradients_weights, gradients_bias):
        self.coefficients -= self.learning_rate * gradients_weights
        self.intercept -= self.learning_rate * gradients_bias

    def _stochastic_gradient_descent(self, features, target):
        num_samples = features.shape[0]
        for _ in range(self.num_iterations):
            for i in range(num_samples):
                random_index = np.random.randint(num_samples)
                X_i = features[random_index:random_index + 1]
                y_i = target[random_index:random_index + 1]
                prediction = np.dot(X_i, self.coefficients) + self.intercept
                gradients_weights, gradients_bias = self.compute_gradients(X_i, y_i, prediction)
                self._update_parameters(gradients_weights, gradients_bias)

    def _normal_equation(self, features, target):
        X_b = np.c_[np.ones((features.shape[0], 1)), features]  # Add bias term
        self.coefficients = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(target)
        self.intercept = self.coefficients[0]
        self.coefficients = self.coefficients[1:]

    def _pseudo_inverse(self, features, target):
        X_b = np.c_[np.ones((features.shape[0], 1)), features]  # Add bias term
        self.coefficients = np.linalg.pinv(X_b).dot(target)
        self.intercept = self.coefficients[0]
        self.coefficients = self.coefficients[1:]

    def compute_gradients(self, features, target, predictions):
        num_samples = features.shape[0]
        gradients_weights = (1 / num_samples) * np.dot(features.T, (predictions - target))
        gradients_bias = (1 / num_samples) * np.sum(predictions - target)
        return gradients_weights, gradients_bias

    def predict(self, features):
        predictions = np.dot(features, self.coefficients) + self.intercept
        return predictions
