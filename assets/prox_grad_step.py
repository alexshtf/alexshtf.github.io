import matplotlib.pyplot as plt
import numpy as np

def grad_step(A, x, step_size):
    grad = 2 * np.matmul(A, x)
    return x - step_size * grad


def prox_step(A, x, step_size):
    # min: u^T A u + (1/2eta) ||u - x||^2
    # solve: 2 A u + (1/eta)(u - x) = 0
    #        2 A u + (1/eta) u = (1/eta) x
    #        (2 eta A + I) u = x
    #        u = inv(2 eta A + I) x
    return np.linalg.solve(2 * step_size * A + np.eye(2), x)


delta = 0.0025
x = np.arange(-3.0, 3.0, delta)
y = np.arange(-2.0, 2.0, delta)
X, Y = np.meshgrid(x, y)
Z = (2 * X**2 - 3*X*Y + 4*Y**2)

A = np.array([[2., -1.5], [-1.5, 4.]])
x = np.array([[1.5], [1.8]])

# choose either prox_step or grad_step
# z = grad_step(A, x, 0.1)
z = prox_step(A, x, 0.4)

lx = np.matmul(np.matmul(x.transpose(), A), x).item()
lz = np.matmul(np.matmul(z.transpose(), A), z).item()

plt.contour(X, Y, Z,
            levels=[lz, lx],
            colors=['black', 'black'])
dz = z - x
plt.arrow(x[0, 0], x[1, 0], dz[0, 0], dz[1, 0],
          color='blue',
          width=0.005,
          head_width=0.12,
          overhang=0.3,
          length_includes_head=True)
plt.scatter(x.item(0), x.item(1), s=50, color='b')
plt.scatter(z.item(0), z.item(1), s=50, color='b')
plt.annotate('$x_t$',
             xy=(x.item(0), x.item(1)),
             xytext=(x.item(0)+0.05, x.item(1)+0.05),
             fontsize=16)
plt.annotate('$x_{t+1}$',
             xy=(z.item(0), z.item(1)),
             xytext=(z.item(0)+0.15, z.item(1)-0.1),
             fontsize=16)
plt.axis('off')
plt.show()

