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
        p        : (3,) 单点坐标，或 (M, 3) 一批点坐标
        p_magnet : (3,) 永磁体中心位置
        sigma_m  : (3,) 永磁体磁矩方向（单位向量或未归一化向量）
        m_m      : 标量磁矩大小，若为 None 则使用由 Br, Dm, Lm 计算得到的值
    返回:
        B : 与 p 形状相同的磁感应强度 (3,) 或 (M, 3)
    支持批量输入，单点与逐点循环的旧实现在数值上完全一致。
    """
    if m_m is None:
        m_m = magnetic_moment_magnet()

    p = np.asarray(p, dtype=np.float64)
    p_magnet = np.asarray(p_magnet, dtype=np.float64)
    sigma_m = np.asarray(sigma_m, dtype=np.float64)
    single = p.ndim == 1
    points = p[None, :] if single else p

    d = points - p_magnet[None, :]
    d_norm = np.linalg.norm(d, axis=1)
    safe = d_norm >= 1e-10
    d_norm_safe = np.where(safe, d_norm, 1.0)

    dot = d @ sigma_m
    term1 = 3.0 * d * (dot / d_norm_safe**5)[:, None]
    term2 = sigma_m[None, :] / (d_norm_safe**3)[:, None]
    B = (mu0 / (4.0 * np.pi)) * m_m * (term1 - term2)
    B[~safe] = 0.0

    return B[0] if single else B

