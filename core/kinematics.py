import numpy as np


def forward_positions(beta: np.ndarray, l: float) -> np.ndarray:
    """
    根据每段绕 z 轴的弯曲角 beta 和分段长度 l，计算从基点到末端的所有段节点坐标。
    返回形状为 (N+1, 3) 的数组，其中第 0 行为基点 (0,0,0)。
    """
    beta = np.asarray(beta)
    N = beta.size
    positions = [np.zeros(3)]
    R = np.eye(3)
    for i in range(N):
        b = beta[i]
        cb, sb = np.cos(b), np.sin(b)
        Rz = np.array([[cb, -sb, 0.0], [sb, cb, 0.0], [0.0, 0.0, 1.0]])
        R = R @ Rz
        pi = positions[-1] + R @ np.array([l, 0.0, 0.0])
        positions.append(pi)
    return np.asarray(positions)

