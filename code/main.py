import numpy as np
import pandas as pd

X_traing = np.load("../datasets/housing_training_set_prepared.npy")
y_traing = np.load("../datasets/housing_training_labels.npy")


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf_node(self):
        return self.value is not None

def parameters_with_normal_equations(X, y):
    """
    return weights for linear regression using normal equations.
    X : The input feature matrix
    y : The output target vector
    """
    # Calculate the parameters using the normal equation
    return np.linalg.inv(X.T @ X) @ X.T @ y


def compute_mse(y_true, y_pred):
    mse = np.sum((y_true - y_pred) ** 2) / len(y_true)
    return mse


def partial_diff_mse(X, y, theta, j):
    predictions = X @ theta
    errors = predictions - y
    mse_thetaj = (2 / len(y)) * np.sum(errors * X[:, j])
    return mse_thetaj


def gradient_descent(X, y, theta, n_iterations, eta):
    m = len(y)  # Number of training examples
    for itr in range(n_iterations):
        # Calculate predictions and errors
        predictions = X.dot(theta)
        errors = predictions - y

        # Calculate gradients
        gradients = (2 / m) * X.T.dot(errors)

        # Update theta
        theta -= eta * gradients
        # print(f"theta:-> {theta[10:]}")

    return theta


def learning_schedule(t, t0=5, t1=50):
    return t0 / (t + t1)


def stochastic_gradient_descent(X, y, theta, n_iterations, n_epochs):
    m = len(y)  # Number of training examples

    for epoch in range(n_epochs):
        # Shuffle the data at the beginning of each epoch
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(m):  # Iterate through each training example
            X_i = X_shuffled[
                i : i + 1
            ]  # Select the corresponding training example
            y_i = y_shuffled[i]  # Corresponding target value

            # Calculate predictions and errors for the selected example
            predictions = X_i.dot(theta)
            errors = predictions - y_i

            # Calculate gradients
            gradients = 2 * X_i.T.dot(errors)

            # Update learning rate
            eta = learning_schedule(epoch * m + i)
            # Update theta
            theta -= eta * gradients

            # Uncomment to see theta updates
            print(f"theta:-> {theta[10:]}")

    return theta


import numpy as np


def wb_support_vector_machine(X_train, y_train, n_iterations, lmbda, lr):
    N, n_X = X_train.shape
    labels = np.where(y_train <= 0, -1, 1)

    w = np.zeros(n_X)
    b = 0

    for iteration in range(n_iterations):
        for idx, x_i in enumerate(X_train):
            condition = labels[idx] * (np.dot(x_i, w) - b) >= 1

            if condition:
                # Only update weights to maintain the regularization
                w -= lr * (2 * lmbda * w)
            else:
                # Update weights and bias for misclassified points
                w -= lr * (2 * lmbda * w - np.dot(x_i, labels[idx]))
                b -= lr * labels[idx]

        # Optional: Print current weights and bias at specific intervals
        if iteration % 100 == 0:  # Print every 100 iterations
            print(f"Iteration :==> {iteration}: w :==> {w}, b = {b}")

    return w, b


def predict(X, w, b):
    approx = np.dot(X, w) - b
    return np.sign(approx)


#############################################################################

W = parameters_with_normal_equations(X_traing, y_traing)

X_example = X_traing[10]
y_example = y_traing[10]

y_pred_example = np.dot(X_example, W)

y_pred = np.dot(X_traing, W)

training_mse = np.sum((y_pred - y_traing) ** 2) / len(X_traing[:, 0])
training_ame = np.sum(np.abs(y_pred - y_traing)) / len(X_traing[:, 0])

print(training_mse)
print(training_ame)


theta_init = np.random.randn(X_traing.shape[1])
n_iterations = 1000
eta = 0.2
n_epochs = 10

w_gd = gradient_descent(X_traing, y_traing, theta_init, n_iterations, eta)

w_sgd = stochastic_gradient_descent(
    X_traing, y_traing, theta_init, n_iterations, n_epochs
)


y_pred_gd = np.dot(X_traing, w_gd)
y_pred_sgd = np.dot(X_traing, w_sgd)

for i in range(11):
    print(y_traing[i], "   ", y_pred[i], "   ", y_pred_gd[i], y_pred_sgd[i])
    print()

#################################################################################


def svm_fit(X, y, lr=0.001, lambda_param=0.01, n_iters=1000):
    n_samples, n_features = X.shape
    y_ = np.where(y <= 0, -1, 1)  # Convert labels to -1, 1

    w = np.zeros(n_features)  # Initialize weights
    b = 0  # Initialize bias

    for _ in range(n_iters):
        for idx, x_i in enumerate(X):
            condition = (
                y_[idx] * (np.dot(x_i, w) - b) >= 1
            )  # Check margin condition
            print("w:", w, "b:", b)
            if condition:
                # Update weights for correctly classified samples
                w -= lr * (2 * lambda_param * w)
            else:
                # Update weights and bias for misclassified samples
                w -= lr * (2 * lambda_param * w - np.dot(x_i, y_[idx]))
                b -= lr * y_[idx]

    return w, b


def svm_predict(X, w, b):
    approx = np.dot(X, w) - b  # Calculate the decision boundary
    return np.sign(approx)  # Return -1 or 1 based on the sign


n_iterations = 1000
lmbda = 0.5
lr = 0.12

w_svm, b_svm = svm_fit(X_traing, y_traing)
y_svm = svm_predict(X_traing, w_svm, b_svm)
