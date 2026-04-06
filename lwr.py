import numpy as np
import matplotlib.pyplot as plt
def local_regression(x0, X, Y, tau):
    x0 = np.array([1, x0])
    X = np.array([[1, x] for x in X])
    weights = np.exp(-np.sum((X - x0) ** 2, axis=1) / (2 * tau**2))
    W = np.diag(weights)
    XT_W = X.T @ W
    beta = np.linalg.pinv(XT_W @ X) @ XT_W @ Y
    return beta @ x0
def draw(tau):
    prediction = [local_regression(x0, X, Y, tau) for x0 in domain] 
    plt.scatter(X, Y, color='black')
    plt.plot(domain, prediction, color='red')
    plt.title(f"Tau = {tau}")
    plt.show()
X = np.linspace(-3, 3, num=1000)
domain = X
Y = np.log(np.abs(X**2 - 1) + 0.5)
draw(10)
draw(0.1)
draw(0.01)
draw(0.001)
