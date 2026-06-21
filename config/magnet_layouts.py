import numpy as np

from .physical_constants import N_default


def W_same_direction(N: int | None = None) -> np.ndarray:
    """
    对应：1229 永磁体工作空间_同向 中的 W 配置（全段同向）
    """
    if N is None:
        N = N_default
    W = np.zeros((N, 3))
    W[:] = [1, 0, 0]
    return W


def W_opposite_direction(N: int | None = None) -> np.ndarray:
    """
    对应：1229 永磁体工作空间_反向 中的 W 配置（前半段 -x，后半段 +x）
    """
    if N is None:
        N = N_default
    W = np.zeros((N, 3))
    for i in range(N):
        if i < N / 2:
            W[i] = [-1, 0, 0]
        else:
            W[i] = [1, 0, 0]
    return W


def W_by_ratio(N: int, positive_ratio: float) -> np.ndarray:
    """
    按正向磁矩占比生成磁矩分布 W（工作空间 V2 磁段比例扫描用）。
    前 n_neg 段为 -x，后 n_pos 段为 +x。
    """
    n_pos = int(round(N * positive_ratio))
    n_neg = N - n_pos
    W = np.zeros((N, 3))
    for i in range(n_neg):
        W[i] = [-1, 0, 0]
    for i in range(n_neg, N):
        W[i] = [1, 0, 0]
    return W


def sigma_m_list_workspace() -> list[np.ndarray]:
    """
    工作空间计算中使用的永磁体朝向列表（同向/反向脚本一致）。
    """
    return [
        np.array([0, 1, 0]),
        np.array([0, -1, 0]),
        np.array([1, 0, 0]),
        np.array([-1, 0, 0]),
        np.array([1, 1, 0]),
        np.array([-1, -1, 0]),
        np.array([1, -1, 0]),
        np.array([-1, 1, 0]),
    ]


def sigma_m_list_workspace_dense_15deg() -> list[np.ndarray]:
    """
    工作空间计算的高密度永磁体朝向列表：
    从 +x 方向 0° 开始，每 15° 一档，扫描至 360°（含 360°）。
    """
    angles_deg = np.arange(0.0, 360.0 + 1e-12, 15.0)
    angles_rad = np.radians(angles_deg)
    return [np.array([np.cos(a), np.sin(a), 0.0]) for a in angles_rad]


def sigma_m_single_magnet() -> np.ndarray:
    """
    对应：1226_永磁体变形.py 中单个永磁体示例的朝向。
    """
    return np.array([0, -1, 0])

