# 依赖：Python 3.10+，numpy、scipy
# 演示：体素占据栅格 -> ESDF -> 用势场法绕障
import numpy as np
from scipy.ndimage import distance_transform_edt

RES = 0.2                      # 每格 0.2 m
H = W = 120                    # 24 m x 24 m 的局部地图
maps = np.zeros((H, W), dtype=bool)          # True = 被占据

# 两堵墙 + 一根柱子（真实环境里这些格由点云投影得到）
maps[30:34, 15:100] = True
maps[72:76, 20:105] = True
maps[50:62, 58:66] = True

# ---- ESDF：每个空闲格到最近障碍的距离（米）----
# distance_transform_edt 给出「到最近 True 的欧氏距离」，单位是格，乘 RES 换成米
esdf = distance_transform_edt(~maps) * RES


def esdf_grad(esdf, pos):
    """用双线性插值取 ESDF 值，并用中心差分取梯度（指向远离障碍的方向）。"""
    x, y = pos / RES
    x0, y0 = int(x), int(y)
    tx, ty = x - x0, y - y0

    def sample(ix, iy):
        ix = np.clip(ix, 0, W - 1); iy = np.clip(iy, 0, H - 1)
        return esdf[iy, ix]

    val = ((1 - tx) * (1 - ty) * sample(x0, y0) + tx * (1 - ty) * sample(x0 + 1, y0)
           + (1 - tx) * ty * sample(x0, y0 + 1) + tx * ty * sample(x0 + 1, y0 + 1))

    gx = (sample(x0 + 1, y0) - sample(x0 - 1, y0)) / (2 * RES)
    gy = (sample(x0, y0 + 1) - sample(x0, y0 - 1)) / (2 * RES)
    return val, np.array([gx, gy])


# ---- 势场法：引力拉向目标 + 斥力推开障碍 ----
def attractive(pos, goal, k=3.0):
    return k * (goal - pos)

def repulsive(pos, esdf, d0=2.0, k=12.0):
    d, grad = esdf_grad(esdf, pos)
    if d >= d0 or d < 1e-6:
        return np.zeros(2)
    return k * (1.0 / d - 1.0 / d0) / d ** 2 * grad     # 越近推得越猛


start = np.array([1.0, 0.5])
goal  = np.array([22.0, 22.0])
pos, vel = start.copy(), np.zeros(2)
DT, VMAX = 0.02, 3.0

min_clear, path_len, steps = np.inf, 0.0, 0

for _ in range(12000):
    f = attractive(pos, goal) + repulsive(pos, esdf)
    vel = np.clip(0.9 * vel + f * DT, -VMAX, VMAX)
    pos = pos + vel * DT
    steps += 1

    clear, _g = esdf_grad(esdf, pos)
    min_clear = min(min_clear, clear)
    path_len += np.linalg.norm(vel) * DT

    if np.linalg.norm(pos - goal) < 0.3:
        break
    if clear <= 0.0:                       # 撞进障碍格
        print(f"❌ 在第 {steps} 步撞上障碍（位置 {pos.round(2)}）")
        break

reached = np.linalg.norm(pos - goal) < 0.3
print(f"到达目标：{reached}   用时 {steps * DT:.1f} s   路径长度 {path_len:.1f} m")
print(f"全程离障碍最近：{min_clear:.2f} m   终点 {pos.round(2)}（目标 {goal}）")
