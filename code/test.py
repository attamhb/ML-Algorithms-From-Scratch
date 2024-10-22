# imports
from sklearn.model_selection import train_test_split
from sklearn import datasets
from lib.utils import visualize_svm
from lib.utils import compute_accuracy
import numpy as np

# from lib.SupportVectorMachines import SupportVectorMachine
from lib.SupportVectorMachines import SupportVectorMachine

X, y = datasets.make_blobs(
    n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40
)
y = np.where(y == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

clf_svm = SupportVectorMachine()

clf_svm.fit(X_train, y_train)
predictions = clf_svm.predict(X_test)

clf_svm_accuracy = compute_accuracy(y_test, predictions)

print("SVM classification accuracy", clf_svm_accuracy)


# visualize_svm(clf_svm, X_test, y_test)
# visualize_svm(clf_svm, X_train, y_train)
visualize_svm(clf_svm, X, y)

# Naive Bayes



# compute_accuracy(true_labels, predicted_labels)

features, labels = datasets.make_classification(
    n_samples=1000, n_features=10, n_classes=2, random_state=123
)
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=123
)

from lib.NaiveBayes import NaiveBayesClassifier
# from lib.NaiveBayesClassifier import NaiveBayesClassifier
nb_classifier = NaiveBayesClassifier()
nb_classifier.fit(X_train, y_train)

# Make predictions
predictions = nb_classifier.predict(X_test)

# Calculate and print accuracy
print("Naive Bayes classification accuracy:", compute_accuracy(y_test, predictions))
