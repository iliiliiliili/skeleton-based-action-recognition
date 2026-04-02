
"""
dirichlet_estimator.py

This module provides a function to estimate the parameters of a Dirichlet distribution
from the mean and variance of sampled categorical distributions.
"""
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import dirichlet
import matplotlib.tri as tri
from tqdm import tqdm
from scipy.special import psi, gammaln


_corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
_AREA = 0.5 * 1 * 0.75**0.5
_triangle = tri.Triangulation(_corners[:, 0], _corners[:, 1])

# For each corner of the triangle, the pair of other corners
_pairs = [_corners[np.roll(range(3), -i)[1:]] for i in range(3)]
# The area of the triangle formed by point xy and another pair or points
tri_area = lambda xy, pair: 0.5 * np.linalg.norm(np.cross(*(pair - xy)))


def estimate_mle_dirichlet(X, tol=1e-7, maxiter=1000):
    """
    Maximum Likelihood Estimation (MLE) of Dirichlet parameters.

    Args:
        X (np.ndarray): Data matrix of shape (N, K), each row is a probability vector (sum to 1).
        tol (float): Convergence tolerance.
        maxiter (int): Maximum number of iterations.

    Returns:
        np.ndarray: Estimated Dirichlet concentration parameters (alpha) of shape (K,).
    """
    from scipy.special import psi, polygamma
    X = np.asarray(X)
    N, K = X.shape
    log_p = np.log(X + 1e-10)
    mean_log_p = np.mean(log_p, axis=0)

    # Initialize alpha with method of moments
    mean = np.mean(X, axis=0)
    var = np.var(X, axis=0)
    alpha0 = (mean[0] * (1 - mean[0]) / (var[0] + 1e-10) - 1)
    alpha0 = max(alpha0, K * 0.1)
    alpha = mean * alpha0

    for _ in range(maxiter):
        alpha0 = np.sum(alpha)
        grad = N * (psi(alpha0) - psi(alpha) + mean_log_p)
        hess = -N * polygamma(1, alpha)
        z = N * polygamma(1, alpha0)
        c = np.sum(grad / hess) / (1.0 / z + np.sum(1.0 / hess))
        update = (grad - c) / hess
        alpha_new = alpha - update
        # Ensure positivity
        alpha_new = np.maximum(alpha_new, 1e-6)
        if np.all(np.abs(alpha_new - alpha) < tol):
            break
        alpha = alpha_new
    return alpha


def estimate_mle_dirichlet_batched(X, tol=1e-7, maxiter=1000):
    """
    Batched Maximum Likelihood Estimation (MLE) of Dirichlet parameters.

    Args:
        X (np.ndarray): Data tensor of shape (B, N, K), each batch is a matrix of N samples of K probabilities.
        tol (float): Convergence tolerance.
        maxiter (int): Maximum number of iterations.

    Returns:
        np.ndarray: Estimated Dirichlet concentration parameters (alpha) of shape (B, K).
    """
    from scipy.special import psi, polygamma
    X = np.asarray(X)
    B, N, K = X.shape
    log_p = np.log(X + 1e-10)
    mean_log_p = np.mean(log_p, axis=1)  # (B, K)

    # Initialize alpha with method of moments for each batch
    mean = np.mean(X, axis=1)  # (B, K)
    var = np.var(X, axis=1)    # (B, K)
    alpha0 = (mean[:, 0] * (1 - mean[:, 0]) / (var[:, 0] + 1e-10) - 1)  # (B,)
    alpha0 = np.maximum(alpha0, K * 0.1)
    alpha = mean * alpha0[:, None]  # (B, K)

    for _ in tqdm(range(maxiter), desc="Estimating Dirichlet parameters"):
        alpha0 = np.sum(alpha, axis=1)  # (B,)
        grad = N * (psi(alpha0)[:, None] - psi(alpha) + mean_log_p)  # (B, K)
        hess = -N * polygamma(1, alpha)  # (B, K)
        z = N * polygamma(1, alpha0)  # (B,)
        c = np.sum(grad / hess, axis=1) / (1.0 / z + np.sum(1.0 / hess, axis=1))  # (B,)
        update = (grad - c[:, None]) / hess  # (B, K)
        alpha_new = alpha - update
        # Ensure positivity
        alpha_new = np.maximum(alpha_new, 1e-6)
        if np.all(np.abs(alpha_new - alpha) < tol):
            break
        alpha = alpha_new
    return alpha

def estimate_dirichlet_params(logits_mean, logits_var, eps=1e-6):
    """
    Estimate Dirichlet parameters (alpha) from the mean and variance of logits.

    Args:
        logits_mean (np.ndarray): Mean logits vector of shape (K,) for K categories.
        logits_var (np.ndarray): Variance of logits vector of shape (K,) for K categories.
        eps (float): Small value to avoid division by zero.

    Returns:
        np.ndarray: Estimated Dirichlet concentration parameters (alpha) of shape (K,).
    """
    logits_mean = np.asarray(logits_mean)
    logits_var = np.asarray(logits_var)
    logits_var = np.clip(logits_var, eps, None)

    # Convert logits to probabilities via softmax
    mean = np.exp(logits_mean) / np.sum(np.exp(logits_mean))

    K = mean.shape[0]
    # Compute alpha0 (sum of alphas)
    alpha0 = (mean * (1 - mean) / logits_var - 1).sum()
    alpha0 = np.clip(alpha0, eps, None)
    
    # Compute each alpha_k
    alpha = mean * (alpha0 + K)
    return alpha


def dirichlet_entropy(alpha):
    alpha0 = np.sum(alpha)
    K = alpha.shape[0]
    entropy = (
        gammaln(alpha0)
        - np.sum(gammaln(alpha))
        - (alpha0 - K) * psi(alpha0)
        + np.sum((alpha - 1) * psi(alpha))
    )
    return entropy

def dirichlet_max_confidence(alpha):
    mean = alpha / np.sum(alpha)
    return np.max(mean)

def is_ood_dirichlet(alpha, threshold=0.8, method="max_confidence"):
    """
    Returns True if the Dirichlet alpha values indicate an out-of-distribution (OOD) sample.

    Args:
        alpha (np.ndarray): Dirichlet concentration parameters (alpha) of shape (K,).
        threshold (float): Threshold for OOD detection. Default is 0.8.
        method (str): Method for OOD detection. Options:
            - "max_confidence": OOD if max(alpha / sum(alpha)) < threshold
            - "entropy": OOD if Dirichlet mean entropy > threshold

    Returns:
        bool: True if sample is OOD, False otherwise.
    """
    alpha = np.asarray(alpha)
    K = alpha.shape[0]
    alpha0 = np.sum(alpha)
    mean = alpha / alpha0
    if method == "max_confidence":
        max_conf = np.max(mean)
        return max_conf < threshold
    elif method == "entropy":
        entropy = dirichlet_entropy(alpha)
        return entropy > threshold
    else:
        raise ValueError(f"Unknown method: {method}")

class Dirichlet(object):
    def __init__(self, alpha):
        '''Creates Dirichlet distribution with parameter `alpha`.'''
        from math import gamma
        from operator import mul
        self._alpha = np.array(alpha)
        self._coef = gamma(np.sum(self._alpha)) / \
                     np.multiply.reduce([gamma(a) for a in self._alpha])
    def pdf(self, x):
        '''Returns pdf value for `x`.'''
        from operator import mul
        return self._coef * np.multiply.reduce([xx ** (aa - 1)
                                                for (xx, aa)in zip(x, self._alpha)])
    def sample(self, N):
        '''Generates a random sample of size `N`.'''
        return np.random.dirichlet(self._alpha, N)


def xy2bc(xy, tol=1.e-4):
    '''Converts 2D Cartesian coordinates to barycentric.
    Arguments:
        `xy`: A length-2 sequence containing the x and y value.
    '''
    coords = np.array([tri_area(xy, p) for p in _pairs]) / _AREA
    return np.clip(coords, tol, 1.0 - tol)


def plot_dirichlet_list(alpha_list, nrows=None, figsize=(12, 18), name="dirichlet_distributions", upper_limit = 50):
    """
    Plot a list of Dirichlet distributions in an n x 6 grid with top-3 categories below each.

    Args:
        alpha_list (list of np.ndarray): List of Dirichlet concentration parameters (alpha).
        nrows (int, optional): Number of rows in the grid. If None, computed automatically.
        figsize (tuple, optional): Figure size for the plot.
        name (str, optional): Name of the file to save the plot as.
    """
    num = len(alpha_list)
    ncols = 6
    if nrows is None:
        nrows = (num + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows * 2, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1, ncols)
    x = np.linspace(0.001, 0.999, 200)
    
    for idx, alpha in enumerate(alpha_list):
        # Normalize alphas so sum is not higher than upper_limit
        alpha_sum = np.sum(alpha)
        if alpha_sum > upper_limit:
            alpha = alpha * (upper_limit / alpha_sum)
        
        row, col = divmod(idx, ncols)
        ax = axes[row * 2, col]
        K = len(alpha)
        
        if K == 2:
            y = dirichlet.pdf(np.stack([x, 1-x], axis=-1), alpha)
            ax.plot(x, y)
            ax.set_title(f"{idx}")
            ax.set_xlim(0, 1)
        elif K == 3:
            n_grid = 100
            x1 = np.linspace(0.001, 0.999, n_grid)
            x2 = np.linspace(0.001, 0.999, n_grid)
            X1, X2 = np.meshgrid(x1, x2)
            X3 = 1 - X1 - X2
            mask = (X3 > 0)
            pdf = np.zeros_like(X1)
            pdf[mask] = dirichlet.pdf(np.stack([X1[mask], X2[mask], X3[mask]], axis=-1), alpha)
            ax.contourf(X1, X2, pdf, levels=20, cmap='viridis')
            ax.set_title(f"{idx}")
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
        else:
            mean = alpha / np.sum(alpha)
            ax.bar(range(K), mean)
            ax.set_title(f"{idx}")
            ax.set_xlabel('Category')
            ax.set_ylabel('Mean')
        ax.set_xticks([])

        if np.isnan(alpha).any() or np.isinf(alpha).any():
            print(f"Warning: alpha contains NaN or Inf for index {idx}")
            alpha = np.nan_to_num(alpha, nan=1e-6, posinf=1e6, neginf=1e-6)
        
        # Plot ternary plot with gradient for K=3, otherwise top-3 categories
        ax_bottom = axes[row * 2 + 1, col]
        # Plot ternary triangle with gradient (top-3 categories)
        top_3_idx = np.argsort(alpha)[-3:]
        alpha_top3 = alpha[top_3_idx]

        dist = Dirichlet(alpha_top3)
        refiner = tri.UniformTriRefiner(_triangle)
        trimesh = refiner.refine_triangulation(subdiv=8)
        pvals = [dist.pdf(xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)]
        ax_bottom.tricontourf(trimesh, pvals, 200, cmap='jet')
        ax_bottom.axis('equal')
        ax_bottom.set_xlim(0, 1)
        ax_bottom.set_ylim(0, 0.75**0.5)
        ax_bottom.axis('off')
    
    # Hide unused axes
    for idx in range(num, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row * 2, col].axis('off')
        axes[row * 2 + 1, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"plots/{name}.png")
    print(f"Saved plot to plots/{name}.png")

