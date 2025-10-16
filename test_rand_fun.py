import numpy as np


def rbf_kernel(x, y, sigma=1000):
    """Compute the RBF (Gaussian) kernel between two vectors."""
    sq_dist = np.linalg.norm(x - y) ** 2
    return np.exp(-sq_dist / (2 * sigma**2))


def kernel_matrix(X):
    K = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            K[i, j] = rbf_kernel(X[i], X[j])
    # Kronecker product with identity matrix
    return K


def sample_multivariate_normal(mean, cov, size=1):
    """Sample from a multivariate normal distribution."""
    return np.random.multivariate_normal(mean, cov, size)


def compute_noise_function(X):
    K = kernel_matrix(X)
    print(K)
    K_inv = np.linalg.inv(np.kron(K, np.eye(X.shape[1])))
    print(np.kron(K, np.eye(X.shape[1])))
    print(K_inv)
    print(np.kron(K, np.eye(X.shape[1])) @ K_inv)
    d = X.shape[1]
    m = X.shape[0]
    alphas = sample_multivariate_normal(np.zeros((m * d)), K_inv).reshape(m, d)
    return X + K @ alphas


if __name__ == "__main__":
    # Example usage
    X = np.array([[13.8462, 73.1265], [-4.14494, -22.7931]])
    print("Original Data:\n", X)
    noisy_X = compute_noise_function(X)
    print("Noisy Data:\n", noisy_X)
