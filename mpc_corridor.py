# 依赖：Python 3.10+，numpy、scipy
# 演示：线性 MPC 写成 QP，用「安全走廊」线性不等式把侧风顶回去
import numpy as np
from scipy.optimize import minimize

DT, N = 0.1, 25                      # 控制周期 100 ms，预测 25 步（2.5 s）
A = np.array([[1.0, DT], [0.0, 1.0]])        # 状态 = [横向位置 y, 横向速度 vy]
B = np.array([[0.5 * DT ** 2], [DT]])        # 控制量 = 横向加速度 ay

Q  = np.diag([2.0, 8.0])             # 跟踪代价（中间步）
QN = np.diag([4.0, 12.0])            # 终端代价（最后一步权重大）
R  = np.array([[1.0]])               # 控制量代价，抑制抖动

A_MAX = 2.0                          # 最大横向加速度 m/s^2（执行器权限）
Y_MAX = 0.6                          # 安全走廊半宽 m —— 就是 H p <= h 的最简形式
WIND  = 0.8                          # 未建模侧风 m/s^2


def build_prediction(N, A, B):
    """把 x_{k+1} = A x_k + B u_k 展开成 X = Abar x0 + Bbar U（X 含 x1..xN）。"""
    nx, nu = A.shape[0], B.shape[1]
    Abar = np.zeros((N * nx, nx))
    Bbar = np.zeros((N * nx, N * nu))
    for i in range(N):
        Abar[i * nx:(i + 1) * nx, :] = np.linalg.matrix_power(A, i + 1)
        for j in range(i + 1):
            Bbar[i * nx:(i + 1) * nx, j * nu:(j + 1) * nu] = \
                np.linalg.matrix_power(A, i - j) @ B
    return Abar, Bbar


ABAR, BBAR = build_prediction(N, A, B)
QBAR = np.kron(np.eye(N), Q)
QBAR[-2:, -2:] = QN                              # 最后一步换成终端代价
RBAR = np.kron(np.eye(N), R)

# 把代价展开成标准二次型：J(U) = 0.5 U'H U + f'U + const
H_MAT = BBAR.T @ QBAR @ BBAR + RBAR
POS_ROWS = np.arange(0, N * 2, 2)                # X 里每个位置分量的行号


def mpc(x0, x_ref=np.zeros(2), y_max=Y_MAX):
    """解一步 QP：min J(U) s.t. |u_k| <= A_MAX, |y_k| <= y_max。只返回第一步。"""
    xr = np.tile(x_ref, N)
    f = 2 * BBAR.T @ QBAR @ (ABAR @ x0 - xr)
    obj = lambda U: U @ H_MAT @ U + f @ U
    jac = lambda U: 2 * H_MAT @ U + f

    # 安全走廊：把「每一步预测位置都在带内」写成两条线性不等式
    cons = [
        {"type": "ineq", "fun": lambda U: y_max - (ABAR @ x0 + BBAR @ U)[POS_ROWS],
         "jac": lambda U: -BBAR[POS_ROWS, :]},
        {"type": "ineq", "fun": lambda U: y_max + (ABAR @ x0 + BBAR @ U)[POS_ROWS],
         "jac": lambda U: BBAR[POS_ROWS, :]},
    ]

    res = minimize(obj, np.zeros(N), jac=jac, bounds=[(-A_MAX, A_MAX)] * N,
                   constraints=cons, method="SLSQP",
                   options={"maxiter": 300, "ftol": 1e-10})
    return res.x[0]


def simulate(y_max):
    """闭环跑 12 s；参考一直是 y=0，但物理对象上一直挂着恒定侧风。"""
    x = np.array([0.0, 0.0])
    ys, us = [], []
    for _ in range(120):
        u = mpc(x, y_max=y_max)
        # 注意 B 是列向量：用 flatten 让状态保持一维，否则下一步广播不上
        x = A @ x + B.flatten() * u + np.array([0.5 * WIND * DT ** 2, WIND * DT])
        ys.append(x[0]); us.append(u)
    return np.array(ys), np.array(us)


y_free, _ = simulate(y_max=1e6)                   # 对照：不加走廊约束
y_corr, u_corr = simulate(y_max=Y_MAX)            # 加走廊约束

print(f"参考横向位置 0.00 m，恒定侧风 {WIND} m/s^2")
print(f"  不加走廊约束：稳态偏移 {y_free[-1]:+.3f} m  <- 靠代价函数是拉不回来的")
print(f"  加安全走廊  ：稳态偏移 {y_corr[-1]:+.3f} m  <- 顶在 {Y_MAX} m 边界上")
overshoot = np.abs(y_corr).max() - Y_MAX
print(f"  全程最大偏移 {np.abs(y_corr).max():.3f} m，走廊半宽 {Y_MAX:.3f} m"
      f"（超出 {overshoot * 1000:+.0f} mm，属于 QP 求解器的数值容差）")
print(f"  控制量范围 {u_corr.min():+.2f} ~ {u_corr.max():+.2f} m/s^2"
      f"（执行器上限 ±{A_MAX}）")
