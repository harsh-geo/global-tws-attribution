import numpy as np
import matplotlib.pyplot as plt

n = 100
t = np.arange(n)
true_tws = np.sin(2 * np.pi * t / 12)

twsc = np.zeros(n)
for i in range(1, n-1):
    twsc[i] = (true_tws[i+1] - true_tws[i-1]) / 2

# Add noise
twsc_pred = twsc + np.random.randn(n) * 0.05

# Reconstruct using matrix A
A = np.zeros((n, n))
for i in range(1, n-1):
    A[i, i-1] = -0.5
    A[i, i+1] = 0.5
A[0, 0] = 1; A[0, 1] = 0
A[-1, -2] = -1; A[-1, -1] = 1

B = twsc_pred.copy()
B[0] = true_tws[0]

tws_rec_unstable = np.linalg.solve(A, B)

# Reconstruct with smoothing
tws_rec_smooth = np.convolve(tws_rec_unstable, np.ones(3)/3, mode='same')

# Reconstruct with Tikhonov
D = np.eye(n)
for i in range(1, n):
    D[i, i-1] = -1
lam = 0.1
tws_rec_reg = np.linalg.solve(A.T@A + lam*(D.T@D), A.T@B)

print("Error Unstable:", np.mean((tws_rec_unstable - true_tws)**2))
print("Error Smooth:", np.mean((tws_rec_smooth[1:-1] - true_tws[1:-1])**2))
print("Error Reg:", np.mean((tws_rec_reg - true_tws)**2))

plt.plot(t, true_tws, 'k-', lw=2, label='True')
plt.plot(t, tws_rec_unstable, 'r--', label='Unstable')
plt.plot(t, tws_rec_reg, 'b-', label='Tikhonov')
plt.legend()
plt.savefig('scratch/test_python.png')
