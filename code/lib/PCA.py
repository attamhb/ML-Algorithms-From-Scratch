import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.eigenvalues = None
        self.eigenvectors = None
        self.mean = None

    def fit(self, X):
        # Step 1: Standardize the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Step 2: Calculate the covariance matrix
        covariance_matrix = np.cov(X_centered, rowvar=False)
        
        # Step 3: Eigen decomposition
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # Step 4: Sort the eigenvalues and eigenvectors
        sorted_indices = np.argsort(self.eigenvalues)[::-1]
        self.eigenvalues = self.eigenvalues[sorted_indices]
        self.eigenvectors = self.eigenvectors[:, sorted_indices]
        
        # Step 5: Select the top 'n_components' eigenvectors
        self.eigenvectors = self.eigenvectors[:, :self.n_components]

    def transform(self, X):
        # Center the data using the mean
        X_centered = X - self.mean
        # Project the data onto the new subspace
        return X_centered.dot(self.eigenvectors)

# Example usage:
if __name__ == "__main__":
    # Sample data (e.g., 5 samples with 3 features)
    X = np.array([[2.5, 2.4, 3.1],
                  [0.5, 0.7, 0.8],
                  [2.2, 2.9, 2.6],
                  [1.9, 2.2, 1.5],
                  [3.1, 3.0, 4.0]])

    # Create a PCA instance and fit the data
    pca = PCA(n_components=2)
    pca.fit(X)

    # Transform the data
    X_transformed = pca.transform(X)
    
    print("Original Data:\n", X)
    print("Transformed Data:\n", X_transformed)
