"""
工作空间面积计算（凸包面积）。
对应工作空间 V2 中「工作空间面积计算」功能。
"""
import numpy as np
from scipy.spatial import ConvexHull, QhullError



def compute_workspace_area_grid(points: np.ndarray, grid_size: float = 5e-4) -> float:
    """
    基于栅格占据统计工作空间面积（二维）。
    参数:
        points: (M, 3) 或 (M, 2)，单位 m
        grid_size: 栅格边长，单位 m（默认 0.5 mm）
    返回:
        面积 (m²)。若需 mm² 则乘以 1e6。
    """
    pts_2d = np.asarray(points)[:, :2]
    if len(pts_2d) == 0:
        return 0.0
    if grid_size <= 0:
        raise ValueError("grid_size must be positive.")

    # 将点映射到离散网格坐标，统计被占据的唯一网格数
    grid_idx = np.floor(pts_2d / grid_size).astype(np.int64)
    occupied = np.unique(grid_idx, axis=0).shape[0]
    return occupied * (grid_size**2)
