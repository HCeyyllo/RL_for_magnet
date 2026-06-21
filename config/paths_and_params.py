import numpy as np

from .physical_constants import l_default, N_default


def workspace_radius_config_same():
    """
    对应：1229 永磁体工作空间_同向 的半径和角度配置
    """
    r_shape = 0.03
    r_values = np.arange(0.03, 0.080 + 1e-12, 0.005)
    angles = np.radians(np.linspace(0, 90, 90))
    return dict(r_shape=r_shape, r_values=r_values, angles=angles)


def workspace_radius_config_opposite():
    """
    对应：1229 永磁体工作空间_反向 的半径和角度配置
    """
    r_shape = 0.04
    r_values = np.arange(0.04, 0.080 + 1e-12, 0.01)
    angles = np.radians(np.linspace(0, 90, 90))
    return dict(r_shape=r_shape, r_values=r_values, angles=angles)


def build_p_magnets(radius: float, angles: np.ndarray) -> np.ndarray:
    """
    根据半径和角度生成永磁体路径点 (x, y, z=0)
    """
    return np.stack(
        [radius * np.cos(angles), radius * np.sin(angles), np.zeros_like(angles)],
        axis=1,
    )


# 典型弹性系数配置（可按需要调整/扩展）
k_workspace_same: float = 4.5e-4
k_workspace_opposite: float = 3.5e-4
k_single_magnet: float = 1.5e-4


def default_segment_params():
    """
    提供统一获取段参数的方式，方便后续如需修改 N 或 l。
    """
    return dict(l=l_default, N=N_default)

