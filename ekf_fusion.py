# 依赖：Python 3.10+，numpy
# 演示：IMU 纯积分 vs EKF 融合（IMU 高频 + 位置传感器低频）
import numpy as np

rng = np.random.default_rng(0)

DT      = 0.005          # IMU 周期 200 Hz
GPS_EVERY = 20           # 每 20 个 IMU 采样来一次位置观测 -> 100 ms
STEPS   = 4000           # 20 s

# ---- 真值：匀加速直线运动 ----
A_TRUE  = 0.3            # m/s^2
BIAS_TRUE = 0.05         # IMU 零偏，纯积分就是被它拖垮的

# ---- 传感器（含噪）----
acc_meas = A_TRUE + BIAS_TRUE + rng.normal(0.0, 0.15, STEPS)   # IMU 加速度（含零偏 + 噪声）
pos_meas = np.full(STEPS, np.nan)
p_true = np.zeros(STEPS); v_true = np.zeros(STEPS)
for k in range(1, STEPS):
    v_true[k] = v_true[k - 1] + A_TRUE * DT
    p_true[k] = p_true[k - 1] + v_true[k - 1] * DT + 0.5 * A_TRUE * DT ** 2
for k in range(0, STEPS, GPS_EVERY):                 # 低频位置观测，单独加 1 m 噪声
    pos_meas[k] = p_true[k] + rng.normal(0.0, 1.0)

# ---- EKF：状态 [p, v, b]，把 IMU 零偏也一起估出来 ----
F = np.array([[1.0, DT, -0.5 * DT ** 2],
              [0.0, 1.0,        -DT],
              [0.0, 0.0,        1.0]])
B = np.array([0.5 * DT ** 2, DT, 0.0])
Q = np.diag([1e-6, 1e-4, 1e-9])       # 过程噪声
R = np.array([[1.0 ** 2]])            # 位置观测噪声

H = np.array([[1.0, 0.0, 0.0]])
x = np.zeros(3)                       # 估计的状态
P = np.eye(3) * 1.0

p_raw, v_raw = 0.0, 0.0               # 纯积分对照
err_raw, err_ekf, bias_est = [], [], []

for k in range(STEPS):
    # --- 预测：用 IMU 推一步 ---
    x = F @ x + B * acc_meas[k]
    P = F @ P @ F.T + Q

    # --- 更新：有位置观测就校正 ---
    if not np.isnan(pos_meas[k]):
        y = pos_meas[k] - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(3) - K @ H) @ P

    # --- 对照：IMU 纯积分（不知道有零偏，误差会二次方发散）---
    p_raw += v_raw * DT + 0.5 * acc_meas[k] * DT ** 2
    v_raw += acc_meas[k] * DT

    err_raw.append(abs(p_raw - p_true[k]))
    err_ekf.append(abs(x[0] - p_true[k]))
    bias_est.append(x[2])

print(f"20 s 后位置误差：IMU 纯积分 {err_raw[-1]:6.2f} m  |  EKF {err_ekf[-1]:5.2f} m")
print(f"位置误差均值：   IMU 纯积分 {np.mean(err_raw):6.2f} m  |  EKF {np.mean(err_ekf):5.2f} m")
print(f"零偏估计：真值 {BIAS_TRUE:.3f}  →  EKF 收敛到 {bias_est[-1]:.3f} m/s^2")
