# UAV Sensor Fusion Demos

配套博客文章的三段**可直接运行**示意代码，对应《中国科技人才》2024 年第 1 期论文：

> 《基于传感器融合的自主避障无人机导航与控制算法研究》

博客完整讲解（含公式、示意图与工程清单）：  
https://ura2039.xyz/blog/posts/uav-sensor-fusion-paper/

## 包含什么

| 文件 | 在讲什么 |
|---|---|
| `ekf_fusion.py` | IMU 高频 + 位置低频 → **EKF** 融合，顺带估计 IMU 零偏 |
| `voxel_esdf_potential.py` | 体素占据栅格 → **ESDF** → 势场法绕障 |
| `mpc_corridor.py` | 线性 **MPC** 写成二次规划，用「安全走廊」顶住侧风 |

根目录扁平放置，没有子目录。

## 环境

- Python **3.10+**
- `numpy`、`scipy`

```bash
pip install -r requirements.txt
```

## 跑起来

```bash
python3 ekf_fusion.py
python3 voxel_esdf_potential.py
python3 mpc_corridor.py
```

期望大致输出：

- EKF：20 s 后 IMU 纯积分位置误差约数米，融合后亚米级，零偏估到约 `0.03`
- ESDF 势场：到达目标 `True`，全程离障最近约 `0.45 m`
- MPC：无走廊稳态偏移约 `+0.81 m`，加走廊后顶在 `0.6 m` 边界（约 `+4 mm` 属 SLSQP 数值容差）

## 说明

这些是**教学/演示脚本**，不是飞控量产代码：动力学被刻意压成 1D / 2D，数值也调过以便一眼看出「加约束前后差在哪」。工程落地请看博客文末的开源参考（`ethzasl_msf`、`ego-planner` 等）。

## License

MIT
