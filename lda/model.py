from numpy import ndarray, zeros, unique, argsort
from numpy.linalg import eig, inv


class LDA:
    """    Linear Discriminant Analysis (LDA)
    for feature extraction and dimensionality reduction.

    Parameters
    ----------
    n_components : int
        The number of linear discriminants (components) to keep.
        Must be <= number of classes - 1

    """

    def __init__(self, n_components: int, /):

        self.n_components = n_components

    def fit(self, X: ndarray, y: ndarray, /) -> ndarray:
        """Computes the within-class scatter matrix (S_W)
         and between-class scatter matrix (S_B),
        solves the generalized eigenvalue problem,
         and projects the input data X
        onto the LDA subspace defined by the top
         `n_components` eigenvectors"""

        n_feature = X.shape[1]
        S_W, S_B = zeros((n_feature, n_feature)), zeros((n_feature, n_feature))
        mean = X.mean(axis=0)
        classes = unique(y)
        for c in classes:
            X_c = X[y == c]
            mean_c = X_c.mean(axis=0)
            S_W += (X_c - mean_c).T @ (X_c - mean_c)
            diff = (mean_c - mean).reshape(-1, 1)
            S_B += X_c.shape[0] * (diff @ diff.T)
        eigenvls, eigenvecs = eig(inv(S_W) @ S_B)
        idx = argsort(eigenvls)[::-1]
        eigenvecs, eigenvls = eigenvecs[:, idx], eigenvls[idx]
        W = eigenvecs[:, :self.n_components]
        X_LDA = X @ W
        return X_LDA