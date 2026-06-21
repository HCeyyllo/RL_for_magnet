import numpy as np

from config.physical_constants import mu0, Br, Dm, Lm


def magnetic_moment_magnet() -> float:
    """
    计算永磁体总磁矩 m_m.
    """
    return (Br / mu0) * np.pi * (Dm / 2) ** 2 * Lm


def magnetic_field_dipole(
    p: np.ndarray,
    p_magnet: np.ndarray,
    sigma_m: np.ndarray,
    m_m: float | None = None,
) -> np.ndarray:
    """
    计算磁偶极子在点 p 处产生的磁场 B.
    参数:
        p        : (3,) 空间点坐标
        p_magnet : (3,) 永磁体中心位置
        sigma_m  : (3,) 永磁体磁矩方向（单位向量或未归一化向量）
        m_m      : 标量磁矩大小，若为 None 则使用由 Br, Dm, Lm 计算得到的值
    返回:
        B : (3,) 磁感应强度
    """
    if m_m is None:
        m_m = magnetic_moment_magnet()

    d = p - p_magnet
    d_norm = np.linalg.norm(d)
    if d_norm < 1e-10:
        return np.zeros(3)

    dot = np.dot(sigma_m, d)
    term1 = 3.0 * d * dot / (d_norm**5)
    term2 = sigma_m / (d_norm**3)
    B = (mu0 / (4.0 * np.pi)) * m_m * (term1 - term2)
    return B

