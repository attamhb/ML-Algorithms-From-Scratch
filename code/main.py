# imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from lib.DecisionTree import DecisionTreeClassifier
from lib.RandomForest import RandomForestClassifier
from lib.LinearRegression import LinearRegression
from lib.LogisticRegression import LogisticRegression


from lib.KNearestNeighbors import KNearestNeighbors
from matplotlib.colors import ListedColormap
from lib.utils import compute_accuracy, compute_mean_squared_error

#############################################################################
# Load and split the iris dataset
iris_data = datasets.load_iris()
features, labels = iris_data.data, iris_data.target

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=1234
)

# Decision Tree Classifier
decision_tree_classifier = DecisionTreeClassifier(max_depth=10)
decision_tree_classifier.fit(X_train, y_train)
decision_tree_predictions = decision_tree_classifier.predict(X_test)
decision_tree_accuracy = compute_accuracy(y_test, decision_tree_predictions)

# Random Forest Classifier
random_forest_classifier = RandomForestClassifier(n_trees=20)
random_forest_classifier.fit(X_train, y_train)
random_forest_predictions = random_forest_classifier.predict(X_test)
random_forest_accuracy = compute_accuracy(y_test, random_forest_predictions)

# Logistic Regression Model
breast_cancer_data = datasets.load_breast_cancer()
features_bc, labels_bc = breast_cancer_data.data, breast_cancer_data.target
X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(
    features_bc, labels_bc, test_size=0.2, random_state=1234
)

logistic_regression_classifier = LogisticRegression(learning_rate=0.01)
logistic_regression_classifier.fit(X_train_bc, y_train_bc)
logistic_regression_predictions = logistic_regression_classifier.predict(
    X_test_bc
)

logistic_regression_accuracy = compute_accuracy(
    y_test_bc, logistic_regression_predictions
)

# knearestneighbors


cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])


plt.figure()
plt.scatter(features[:,2],features[:,3], c=labels, cmap=cmap, edgecolor='k', s=20)
plt.show()


clf = KNearestNeighbors(num_neighbors=5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print(predictions)

acc_knn = np.sum(predictions == y_test) / len(y_test)
print(acc_knn)


#############################################################################
# Print classification results
print("Decision Tree Accuracy:", decision_tree_accuracy)
print("Random Forest Accuracy:", random_forest_accuracy)
print("Logistic Regression Accuracy:", logistic_regression_accuracy)
print("Knearestneighbors Accuracy", logistic_regression_accuracy)
print()

#################################################################################

# Generate synthetic regression data
features, target = datasets.make_regression(
    n_samples=100, n_features=1, noise=20, random_state=4
)

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=1234
)

# Visualize the generated data
fig = plt.figure(figsize=(8, 6))
plt.scatter(features, target, color="b", marker="o", s=30)
plt.title("Generated Regression Data")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.show()

# Train the Linear Regression model
linear_regression_model = LinearRegression(learning_rate=0.01)
linear_regression_model.fit(X_train, y_train)

# Make predictions on the test set
predictions = linear_regression_model.predict(X_test)

# Calculate Mean Squared Error
mean_squared_error = compute_mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mean_squared_error)

# Visualize predictions
predicted_line = linear_regression_model.predict(features)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8, 6))

# Scatter plot for training and test data
plt.scatter(X_train, y_train, color=cmap(0.9), s=10, label='Training Data')
plt.scatter(X_test, y_test, color=cmap(0.5), s=10, label='Test Data')

# Plotting the prediction line
plt.plot(
    features,
    predicted_line,
    color='black',
    linewidth=2,
    label='Prediction Line',
)
plt.title("Linear Regression Predictions")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()
plt.show()
#
###
