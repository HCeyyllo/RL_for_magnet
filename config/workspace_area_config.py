"""
工作空间面积–磁段比例 扫描配置。
对应工作空间 V2 中「不同正向磁矩占比对应工作空间面积」功能。
"""
from datetime import datetime
from pathlib import Path

import numpy as np

from .paths_and_params import build_p_magnets


# 正向磁矩占比扫描范围（与 V2 一致：0.1 ~ 1.0，步长 0.1）
RATIOS_DEFAULT: np.ndarray = np.arange(0.1, 1.01, 0.1)

# 扫描时使用的永磁体轨迹：半径与角度（与 V2 一致）
R_SHAPE_RATIO: float = 0.03
ANGLES_RATIO: np.ndarray = np.radians(np.linspace(0, 90, 90))

# 工作空间图像根目录；实际保存路径为「根目录 + 时间戳子文件夹」
WORKSPACE_IMGS_BASE_DIR: str = "workspace_imgs"


def workspace_imgs_dir(timestamp: str | None = None) -> str:
    """
    单次运行的工作空间图像目录：`<WORKSPACE_IMGS_BASE_DIR>/<timestamp>/`
    timestamp 为 YYYYMMDD_HHMMSS；省略则使用当前时间。
    """
    ts = timestamp if timestamp is not None else datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path(WORKSPACE_IMGS_BASE_DIR) / ts)

# 占比–面积关系图保存路径
RATIO_VS_AREA_FIG: str = "ratio_vs_workspace_area.png"


def workspace_radius_config_ratio():
    """
    工作空间面积扫描用的半径与角度配置。
    """
    return dict(
        r_shape=R_SHAPE_RATIO,
        angles=ANGLES_RATIO,
    )


def get_p_magnets_shape_ratio():
    """扫描时使用的永磁体路径点 (仅形状半径)。"""
    cfg = workspace_radius_config_ratio()
    return build_p_magnets(cfg["r_shape"], cfg["angles"])
