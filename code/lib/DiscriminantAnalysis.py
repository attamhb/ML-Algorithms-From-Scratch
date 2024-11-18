import numpy as np


class LDA:
    def __init__(self):
        self.mean_vec = None
        self.cov_mat = None
        self.class_priors = None
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean_vec = np.array(
            [X[y == cls].mean(axis=0) for cls in self.classes]
        )
        self.cov_mat = np.cov(X, rowvar=False)  # assumes equal covariance
        self.class_priors = np.array(
            [np.mean(y == cls) for cls in self.classes]
        )

    def predict(self, X):
        preds = []
        for x in X:
            posteriors = []
            for idx, cls in enumerate(self.classes):
                mean = self.mean_vec[idx]
                cov = self.cov_mat
                prior = self.class_priors[idx]

                # Calculate the posterior probability
                likelihood = np.exp(
                    -0.5 * (x - mean).T @ np.linalg.inv(cov) @ (x - mean)
                ) / np.sqrt(np.linalg.det(cov))
                posterior = likelihood * prior
                posteriors.append(posterior)
            preds.append(self.classes[np.argmax(posteriors)])
        return np.array(preds)


class QDA:
    def __init__(self):
        self.mean_vec = None
        self.cov_mats = None
        self.class_priors = None
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean_vec = np.array(
            [X[y == cls].mean(axis=0) for cls in self.classes]
        )
        self.cov_mats = np.array(
            [np.cov(X[y == cls], rowvar=False) for cls in self.classes]
        )
        self.class_priors = np.array(
            [np.mean(y == cls) for cls in self.classes]
        )

    def predict(self, X):
        preds = []
        for x in X:
            posteriors = []
            for idx, cls in enumerate(self.classes):
                mean = self.mean_vec[idx]
                cov = self.cov_mats[idx]
                prior = self.class_priors[idx]

                # Calculate the posterior probability
                likelihood = np.exp(
                    -0.5 * (x - mean).T @ np.linalg.inv(cov) @ (x - mean)
                ) / np.sqrt(np.linalg.det(cov))
                posterior = likelihood * prior
                posteriors.append(posterior)
            preds.append(self.classes[np.argmax(posteriors)])
        return np.array(preds)
