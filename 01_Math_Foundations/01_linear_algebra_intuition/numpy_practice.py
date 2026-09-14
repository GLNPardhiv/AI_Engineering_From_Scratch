import numpy as np

a = np.array([1, 2, 3], dtype=float)
b = np.array([4, 5, 6], dtype=float)

print(f"a + b = {a + b}")
print(f"a · b = {np.dot(a, b)}")
print(f"|a| = {np.linalg.norm(a):.4f}")
print(f"cosine = {np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)):.4f}")

W = np.random.randn(2, 3) * 0.1
print(f"W = {W}")
x = np.array([1.0, 0.5, -0.3])
print(f"Wx = {W @ x}")


# Rank, Projection, and QR with NumPy
A = np.array([[1, 2], [2, 4]])
print(f"Rank: {np.linalg.matrix_rank(A)}")

a = np.array([3, 4])
b = np.array([1, 0])
proj = (np.dot(a, b) / np.dot(b, b)) * b
print(f"Projection of {a} onto {b}: {proj}")

# np.random.randn(3, 3) Creates a 3 X 3 matrix filled with random numbers from a normal distribution.
# np.linalg.qr(...) Splits (decomposes) the random matrix into two new matrices, called Q and R, such that the original matrix equals Q multiplied by R (A = Q dot R).
Q, R = np.linalg.qr(np.random.randn(3, 3))
# np.eye(3) Creates a 3 X 3 identity matrix (ones on the diagonal, zeros elsewhere).
# np.allclose(...) Compares two matrices and returns True if they are nearly identical, ignoring tiny computer rounding errors. This line prints True because Q is orthogonal.
print(f"Q is orthogonal: {np.allclose(Q @ Q.T, np.eye(3))}")
# np.triu(R) Takes matrix R and keeps only the numbers on and above the main diagonal, turning everything below the diagonal into zeros.
print(f"R is upper triangular: {np.allclose(R, np.triu(R))}")


# PyTorch -- Tensors Are Vectors with Autodiff
# import torch

# x = torch.randn(3, requires_grad=True): Creates a list of 3 random numbers called x.
# requires_grad=True tells PyTorch to watch this variable so it can calculate math derivatives (gradients) later.
# x = torch.randn(3, requires_grad=True)
# y = torch.tensor([1.0, 0.0, 0.0])

# similarity = torch.dot(x, y)
# similarity.backward(): Computes the derivative of the dot product with respect to x.
# This tells us how much the final result changes if we change each number in x.
# similarity.backward()

# print(f"x = {x.data}")
# print(f"y = {y.data}")
# print(f"dot product = {similarity.item():.4f}")
# print(f"d(dot)/dx = {x.grad}")
