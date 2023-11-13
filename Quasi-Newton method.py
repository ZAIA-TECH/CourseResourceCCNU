import numpy as np
import matplotlib.pyplot as plt

# Rosenbrock函数
def rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

# 梯度方向
def grad(x):
    return np.array([400 * x[0]**3 - 400 * x[0] * x[1] + 2 * x[0] -2, -200 * x[0]**2 + 200 * x[1]])

# wolfe条件计算alpha
def wolfe(f, df, p, x, alpham, c1, c2, t):
    flag = 0
    a = 0
    b = alpham
    fk = f(x)
    alpha = b * np.random.uniform(0, 1)
    gk = df(x)
    gk1 = df(x+alpha*p)
    phi0 = fk
    dphi0 = np.dot(gk,p)
    dphi = np.dot(gk1,p)
    while (flag == 0):
        newfk = f(x + alpha * p)
        phi = newfk
        if phi <= phi0 + c1*alpha*dphi0:
            if dphi >= c2*dphi0:
                flag = 1
            else:
                a = alpha
                b = b
                if(b < alpham):
                    alpha = (a+b)/2
                else:
                    alpha = t*alpha
        else:
            a = a
            b = alpha
            alpha = (a+b)/2
        gk1 = df(x+alpha*p)
        dphi = np.dot(gk1,p)
    return alpha

# 求Hk+1
def bfgs(f, df, dfk1, dfk, xk1, xk, Hk):
    I = np.identity(2)
    yk = dfk1 - dfk
    sk = xk1 - xk
    rhok = 1 / np.dot(yk.T, sk)
    Hk1 = np.dot(np.dot(I - rhok*np.dot(sk, yk.T), Hk), I - rhok*np.dot(yk, sk.T)) + rhok*np.dot(sk, sk.T)
    Hk = Hk1
    return Hk

# 拟牛顿法迭代过程
def quasinewton(x0):
    maxk = 100000
    W = np.zeros((2, maxk))
    W[:, 0] = x0
    epsilon = 1e-5
    x = x0
    i = 0
    xn = np.array([1, 1])
    delta = np.linalg.norm(x - xn)
    Hk = np.identity(2)
    while delta > epsilon:
        xk = x.copy()
        dfk = grad(xk)
        p = -np.dot(Hk, grad(x))
        alpha = wolfe(rosenbrock, grad, p, x, 1, 0.4, 0.9, 2)
        x += alpha * p
        W[:, i] = x
        xk1 = x
        dfk1 = grad(xk1)
        Hk = bfgs(rosenbrock, grad, dfk1, dfk, xk1, xk, Hk)
        delta = np.linalg.norm(x - xn)
        i += 1
    print("迭代次数为：\n", i)
    print("近似解为：\n", x)
    print("误差为：\n", delta)
    W = W[:, 0:i]
    return W, x

if __name__ == "__main__":
    X1 = np.arange(-1.5, 1.5+0.05, 0.05)
    X2 = np.arange(-0.1, 1.5+0.05, 0.05)
    [x1, x2] = np.meshgrid(X1, X2)
    f = 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2

    # 调整等高线密度，设置levels参数
    levels = np.logspace(0, 10, 20)

    # 绘制等值线，并设置标签
    contour = plt.contour(x1, x2, f, levels=levels, colors='gray', alpha=0.5)
    plt.clabel(contour, inline=True, fontsize=8)

    x0 = np.array([-1.2, 1])
    W, x_opt = quasinewton(x0)

    # 绘制优化路径
    plt.plot(W[0, :], W[1, :], 'r*-')

    # 标记初始点和最终解
    plt.scatter(x0[0], x0[1], color='blue', marker='o', label='Initial Point')
    plt.scatter(x_opt[0], x_opt[1], color='green', marker='o', label='Final Solution')

    plt.legend()
    plt.show()
