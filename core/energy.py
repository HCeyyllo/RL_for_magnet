import numpy as np
from scipy.optimize import minimize

from core.magnetic_field import magnetic_field_dipole
from core.kinematics import forward_positions


def total_energy(
    beta: np.ndarray,
    p_magnet: np.ndarray,
    sigma_m: np.ndarray,
    W: np.ndarray,
    k: float,
    l: float,
    m_seg: float,
) -> float:
    """
    总能量：弹性能 + 磁能。
    beta   : (N,) 各段弯曲角
    p_magnet, sigma_m : 单个永磁体位置与方向
    W     : (N,3) 每段磁矩方向（局部坐标系下）
    k     : 弹性系数
    l     : 分段长度
    m_seg : 每段等效磁矩大小
    """
    beta = np.asarray(beta)
    positions = forward_positions(beta, l)

    VE = 0.0
    VB = 0.0
    R = np.eye(3)

    for i, b in enumerate(beta):
        VE += 0.5 * k * b**2

        cb, sb = np.cos(b), np.sin(b)
        Rz = np.array([[cb, -sb, 0.0], [sb, cb, 0.0], [0.0, 0.0, 1.0]])
        R = R @ Rz
        m_global = m_seg * (R @ W[i])

        pi = positions[i + 1]
        B = magnetic_field_dipole(pi, p_magnet, sigma_m)
        VB -= float(np.dot(m_global, B))

    return VE + VB


def optimize_for_radius(
    p_magnets: np.ndarray,
    sigma_m_list: list[np.ndarray],
    W: np.ndarray,
    k: float,
    l: float,
    m_seg: float,
) -> dict:
    """
    针对给定半径的一条永磁体路径，遍历所有永磁体位置和 sigma_m 朝向，
    返回与原脚本结构兼容的结果字典：
        {sigma_idx: {'theta_all': (M,N), 'end_positions': (M,3), 'sigma_m': ...}}
    """
    N = W.shape[0]
    num_sigma = len(sigma_m_list)
    M = len(p_magnets)

    radius_results: dict[int, dict] = {}
    for idx in range(num_sigma):
        radius_results[idx] = {
            "theta_all": np.zeros((M, N)),
            "end_positions": np.zeros((M, 3)),
            "sigma_m": sigma_m_list[idx],
        }

    for i, p_m in enumerate(p_magnets):
        p_magnet = np.array([p_m[0], p_m[1], 0.0])

        for sigma_idx, sigma_m in enumerate(sigma_m_list):
            if i > 0:
                beta0 = radius_results[sigma_idx]["theta_all"][i - 1]
            else:
                beta0 = np.zeros(N)

            res = minimize(
                lambda b: total_energy(b, p_magnet, sigma_m, W, k, l, m_seg),
                beta0,
                method="L-BFGS-B",
                bounds=[(-np.pi, np.pi)] * N,
                options={"maxiter": 3000, "ftol": 1e-9},
            )

            radius_results[sigma_idx]["theta_all"][i] = res.x
            positions = forward_positions(res.x, l)
            radius_results[sigma_idx]["end_positions"][i] = positions[-1]

    return radius_results

