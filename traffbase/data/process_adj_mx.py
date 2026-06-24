import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg


def calculate_symmetric_normalized_laplacian(adj: np.ndarray) -> sp.spmatrix:
    """
    Calculate the symmetric normalized Laplacian.

    The symmetric normalized Laplacian matrix is given by:
    L^{Sym} = I - D^{-1/2} A D^{-1/2}, where L is the unnormalized Laplacian,
    D is the degree matrix, and A is the adjacency matrix.

    Args:
        adj (np.ndarray): Adjacency matrix A.

    Returns:
        sp.spmatrix: Symmetric normalized Laplacian L^{Sym}.
    """
    adj = sp.coo_matrix(adj)
    degree = np.array(adj.sum(1)).flatten()
    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0
    matrix_degree_inv_sqrt = sp.diags(degree_inv_sqrt)

    laplacian = sp.eye(adj.shape[0]) - matrix_degree_inv_sqrt.dot(adj).dot(matrix_degree_inv_sqrt).tocoo()

    return laplacian


def calculate_scaled_laplacian(
        adj: np.ndarray,
        lambda_max: float | None = 2,
        undirected: bool = True) -> sp.spmatrix:
    """
    Scale the normalized Laplacian for use in Chebyshev polynomials.

    Rescale the Laplacian matrix such that its eigenvalues are within the range [-1, 1].

    Args:
        adj (np.ndarray): Adjacency matrix A.
        lambda_max (float | None, optional): Maximum eigenvalue; if None it is
            estimated via eigsh. Defaults to 2.
        undirected (bool, optional): If True, treats the graph as undirected, defaults to True.

    Returns:
        sp.spmatrix: Scaled Laplacian matrix.
    """
    if undirected:
        adj = np.maximum(adj, adj.T)

    laplacian = calculate_symmetric_normalized_laplacian(adj)

    if lambda_max is None:
        eigenvalues, _ = linalg.eigsh(laplacian, 1, which='LM')
        lambda_max = float(eigenvalues[0])

    laplacian = sp.csr_matrix(laplacian)
    identity = sp.identity(laplacian.shape[0], format='csr', dtype=laplacian.dtype)

    scaled_laplacian = (2 / lambda_max) * laplacian - identity

    return scaled_laplacian


def calculate_symmetric_message_passing_adj(adj: np.ndarray) -> sp.spmatrix:
    """
    Calculate the renormalized message-passing adjacency matrix as proposed in GCN.

    The message-passing adjacency matrix is defined as A' = D^{-1/2} (A + I) D^{-1/2}.

    Args:
        adj (np.ndarray): Adjacency matrix A.

    Returns:
        sp.spmatrix: Renormalized message-passing adjacency matrix.
    """
    adj = adj + np.eye(adj.shape[0], dtype=np.float32)
    adj = sp.coo_matrix(adj)

    row_sum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(row_sum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0

    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    mp_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).astype(np.float32)

    return mp_adj


def calculate_transition_matrix(adj: np.ndarray) -> np.ndarray:
    """
    Calculate the transition matrix as proposed in DCRNN and Graph WaveNet.

    The transition matrix is defined as P = D^{-1} A, where D is the degree matrix.

    Args:
        adj (np.ndarray): Adjacency matrix A.

    Returns:
        np.ndarray: Transition matrix P.
    """
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(row_sum, -1)
    d_inv[np.isinf(d_inv)] = 0.0

    d_mat = sp.diags(d_inv)
    prob_matrix = d_mat.dot(adj).astype(np.float32).toarray()

    return prob_matrix