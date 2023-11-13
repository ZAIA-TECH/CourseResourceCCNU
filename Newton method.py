import numpy as np
import random
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
    alpha = b * random.uniform(0, 1)
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

# Hessian矩阵
def hess(x):
    return np.array([[1200 * x[0]**2 - 400 * x[1] + 2, -400 * x[0]], [-400 * x[0], 200]])

# 牛顿法迭代过程
def newton(x0):
    maxk = 10000
    W = np.zeros((2, maxk))
    W[:, 0] = x0
    epsilon = 1e-5
    x = x0
    i = 0
    xn = np.array([1, 1])
    delta = np.linalg.norm(x - xn)
    while delta > epsilon:
        H = hess(x)
        p = -np.dot(np.linalg.inv(H), grad(x))
        alpha = wolfe(rosenbrock, grad, p, x, 1, 0.4, 0.9, 2)
        x += alpha * p
        W[:, i] = x
        delta = np.linalg.norm(x - xn)
        i += 1
    print("迭代次数为：", i)
    print("近似解为：", x)
    print("误差为：", delta)
    W = W[:, 0:i]
    return W

if __name__ == "__main__":
    X1 = np.arange(-1.5, 1.5+0.05, 0.05)
    X2 = np.arange(-0.1, 1.5+0.05, 0.05)
    [x1, x2] = np.meshgrid(X1, X2)
    f = 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2

    # 调整等高线密度，设置levels参数
    levels = np.logspace(0, 10, 20)
    plt.contour(x1, x2, f, levels=levels, colors='gray', alpha=0.5)

    x0 = np.array([-1.2, 1])
    W = newton(x0)

    # 绘制等值线
    plt.contour(x1, x2, f, levels=levels, colors='gray', alpha=0.5)

    # 绘制优化路径
    plt.plot(W[0, :], W[1, :], 'r*-', label='Optimization Path')

    # 标记初始点和最终解
    plt.plot(x0[0], x0[1], 'go', label='Initial Point')
    plt.plot(W[0, -1], W[1, -1], 'bo', label='Final Solution')

    # 添加等值线标签
    plt.clabel(plt.contour(x1, x2, f, levels=levels, colors='gray', alpha=0.5), inline=True, fontsize=8)

    plt.legend()
    plt.show()


















''' 旧版本
# 输入
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 可以显示中文
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False

# 获取用户输入的初始点
x0 = float(input('输入初始点：(例如1, 2, 3, ...) \n'))

# 设置阈值，当迭代变化小于阈值时停止
theta = 1e-5

# 定义初始函数和导数
init_fun = lambda x: np.exp(x) - x**2
derivative = lambda x: np.exp(x) - 2*x

# 生成x和y的取值范围
x = np.linspace(-1, 3, 400)
y = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x, y)
Z = init_fun(X)

# 创建 3D 图
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

# 迭代法
def iterative(x0=x0, theta=theta):
    results = []  # 用于存储每次迭代的结果
    number = 0
    xi = x0
    while True and number <= 100:
        # 计算下一个迭代点
        xi = xi - init_fun(xi) / derivative(xi)

        # 在图上标注迭代点
        ax.scatter(xi, init_fun(xi), number, c='red', marker='o', s=30)

        # 判断是否满足停止条件
        if abs(init_fun(xi)) < theta:
            results.append((xi, number))
            return results
        results.append((xi, number))
        number += 1

# 使用迭代法计算求解x0
results = iterative(x0, theta)

# 打印每次迭代的结果
for result in results:
    xi, number = result
    print(f'迭代结果：{xi}, 迭代次数：{number}')

ax.set_title('牛顿法迭代 3D 图')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
'''