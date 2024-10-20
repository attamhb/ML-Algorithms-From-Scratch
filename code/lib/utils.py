import numpy as np 
import matplotlib.pyplot as plt  


def visualize_svm(svm_model, features, labels):
    def calculate_hyperplane_value(feature_x, weights, bias, offset):
        return (-weights[0] * feature_x + bias + offset) / weights[1]

    figure = plt.figure()
    axis = figure.add_subplot(1, 1, 1)
    plt.scatter(features[:, 0], features[:, 1], marker="o", c=labels)

    x_min = np.amin(features[:, 0])
    x_max = np.amax(features[:, 0])

    y_min_hyperplane = calculate_hyperplane_value(x_min, svm_model.weights, svm_model.bias, 0)
    y_max_hyperplane = calculate_hyperplane_value(x_max, svm_model.weights, svm_model.bias, 0)

    y_min_margin = calculate_hyperplane_value(x_min, svm_model.weights, svm_model.bias, -1)
    y_max_margin = calculate_hyperplane_value(x_max, svm_model.weights, svm_model.bias, -1)

    y_min_margin_positive = calculate_hyperplane_value(x_min, svm_model.weights, svm_model.bias, 1)
    y_max_margin_positive = calculate_hyperplane_value(x_max, svm_model.weights, svm_model.bias, 1)

    axis.plot([x_min, x_max], [y_min_hyperplane, y_max_hyperplane], "y--")
    axis.plot([x_min, x_max], [y_min_margin, y_max_margin], "k")
    axis.plot([x_min, x_max], [y_min_margin_positive, y_max_margin_positive], "k")

    y_min = np.amin(features[:, 1])
    y_max = np.amax(features[:, 1])
    axis.set_ylim([y_min - 3, y_max + 3])

    plt.show()


def compute_accuracy(true_labels, predicted_labels):
    return np.sum(true_labels == predicted_labels) / len(true_labels)


def compute_mean_squared_error(true_labels, predicted_labels):
    return np.mean((true_labels - predicted_labels) ** 2)

