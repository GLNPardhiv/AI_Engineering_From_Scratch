import numpy as np


def power_iteration(M, num_iters=100):
    n = M.shape[1]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)

    for _ in range(num_iters):
        Mv = M @ v
        v = Mv / np.linalg.norm(Mv)

    eigenvalue = v @ M @ v
    return eigenvalue, v

def svd_from_scratch(A, k=None):
    m, n = A.shape
    if k is None:
        k = min(m, n)

    sigmas = []
    us = []
    vs = []

    A_residual = A.copy().astype(float)

    for _ in range(k):
        AtA = A_residual.T @ A_residual
        eigenvalue, v = power_iteration(AtA, num_iters=200)

        if eigenvalue < 1e-10:
            break

        sigma = np.sqrt(eigenvalue)
        u = A_residual @ v / sigma

        sigmas.append(sigma)
        us.append(u)
        vs.append(v)

        A_residual = A_residual - sigma * np.outer(u, v)

    U = np.column_stack(us) if us else np.empty((m, 0))
    S = np.array(sigmas)
    V = np.column_stack(vs) if vs else np.empty((n, 0))

    return U, S, V

if __name__ == "__main__":
    np.random.seed(42)
    A = np.random.randn(5, 4)

    U_ours, S_ours, V_ours = svd_from_scratch(A)
    U_np, S_np, Vt_np = np.linalg.svd(A, full_matrices=False)

    print("Our singular values:", np.round(S_ours, 4))
    print("NumPy singular values:", np.round(S_np, 4))

    A_reconstructed = U_ours @ np.diag(S_ours) @ V_ours.T
    print(f"Reconstruction error: {np.linalg.norm(A - A_reconstructed):.8f}")
