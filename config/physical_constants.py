import numpy as np


mu0: float = 4 * np.pi * 1e-7  # 真空磁导率 (H/m)
Br: float = 1.44               # 永磁体剩磁 (T)
Dm: float = 24e-3              # 永磁体直径 (m)
Lm: float = 24e-3              # 永磁体长度 (m)

# 分段参数
l_default: float = 0.0008      # 默认分段长度 (m)
N_default: int = 20            # 默认伪刚体段数
m_seg_default: float = 1e-4    # 默认每段等效磁矩 (A·m²)

