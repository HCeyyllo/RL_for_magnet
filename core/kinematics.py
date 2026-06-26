import numpy as np


def forward_positions(beta: np.ndarray, l: float) -> np.ndarray:
    """
    根据每段绕 z 轴的弯曲角 beta 和分段长度 l，计算从基点到末端的所有段节点坐标。
    返回形状为 (N+1, 3) 的数组，其中第 0 行为基点 (0,0,0)。

    向量化实现：由于全部旋转都绕 z 轴且位于 z=0 平面，第 i 段的全局朝向角即
    theta_i = cumsum(beta)_i，节点坐标是各段方向向量的累加。结果与逐段矩阵
    连乘的旧实现在数值上完全一致，但去除了 Python 循环与 3x3 矩阵构造。
    """
    beta = np.asarray(beta, dtype=np.float64)
    N = beta.size
    positions = np.zeros((N + 1, 3), dtype=np.float64)
    if N == 0:
        return positions

    theta = np.cumsum(beta)
    steps_xy = l * np.stack([np.cos(theta), np.sin(theta)], axis=1)
    positions[1:, :2] = np.cumsum(steps_xy, axis=0)
    return positions

