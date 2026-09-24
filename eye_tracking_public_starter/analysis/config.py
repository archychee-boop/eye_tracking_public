# -*- coding: utf-8 -*-
"""全局配置:路径、设计/组别/内容映射、AOI 面积占比(论文表 2-2)。"""

import sys
from pathlib import Path

# Windows 默认控制台/管道编码为 GBK,会把中文与 χ²/σ_b 等统计符号写坏或直接报错;
# 统一改为 UTF-8 输出,保证 00–07 脚本在任何终端下都能正常打印。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError, OSError):
        pass

# ---------- 路径 ----------
PKG_ROOT = Path(__file__).resolve().parent                      # 复现分析/
PROJECT_ROOT = PKG_ROOT.parent                                  # public starter/

DATA_MASTER = PROJECT_ROOT / "data" / "synthetic_master.csv"
DATA_MATERIAL = PROJECT_ROOT / "data" / "material_measurements.xlsx"

OUT_DIR = PROJECT_ROOT / "outputs_synthetic"
TABLE_DIR = OUT_DIR / "表格"
FIG_DIR = OUT_DIR / "图表"

SHEET_MASTER = "总表_完整版"

# ---------- 设计/组别/内容 ----------
DESIGNS = list("ABCDEFGH")          # 8 个实验版面
ANCHOR = "Z"                        # 锚点(白鹈鹕)
DESIGNS_ALL = DESIGNS + [ANCHOR]

GROUP_DESIGNS = {
    "组1": ["A", "C", "F", "Z"],
    "组2": ["B", "E", "G", "Z"],
    "组3": ["A", "D", "H", "Z"],
}
CONTENT_DESIGNS = {
    "造山运动": ["A", "B"],
    "红耳鹎": ["C", "D", "E"],
    "红果仔": ["F", "G", "H"],
    "白鹈鹕(锚点)": ["Z"],
}
# 设计代号 -> 内容类型
DESIGN_CONTENT = {d: c for c, ds in CONTENT_DESIGNS.items() for d in ds}

# 设计代号 -> 版式类型标签(论文表 3-3)
DESIGN_LABELS = {
    "A": "内容导向型", "B": "内容平衡型", "C": "视觉吸引型", "D": "图文均衡型",
    "E": "内容清晰型", "F": "详情导向型", "G": "要点明确型", "H": "内容极简型",
    "Z": "锚点",
}

# 设计 -> 所属区组(用于区组内比较)
DESIGN_BLOCK = {}
for c, ds in CONTENT_DESIGNS.items():
    if c == "白鹈鹕(锚点)":
        continue
    for d in ds:
        DESIGN_BLOCK[d] = c

# ---------- AOI 面积占比(%, 论文表 2-2,顺序 标题/要点/详情/关键图片) ----------
LAYERS = ["标题层", "要点层", "详情层", "关键图片"]
DESIGN_AREA_PCT = {
    "A": {"标题层": 14.00, "要点层": 4.00, "详情层": 34.00, "关键图片": 17.00},
    "B": {"标题层": 14.00, "要点层": 14.00, "详情层": 17.00, "关键图片": 17.00},
    "C": {"标题层": 14.00, "要点层": 4.00, "详情层": 17.00, "关键图片": 38.00},
    "D": {"标题层": 4.00, "要点层": 14.00, "详情层": 34.00, "关键图片": 17.00},
    "E": {"标题层": 4.00, "要点层": 14.00, "详情层": 17.00, "关键图片": 38.00},
    "F": {"标题层": 4.00, "要点层": 4.00, "详情层": 34.00, "关键图片": 38.00},
    "G": {"标题层": 14.00, "要点层": 14.00, "详情层": 3.00, "关键图片": 38.00},
    "H": {"标题层": 4.00, "要点层": 4.00, "详情层": 3.00, "关键图片": 17.00},
    "Z": {"标题层": 2.13, "要点层": 3.22, "详情层": 14.72, "关键图片": 4.79},
}

# ---------- 中文变量/标签映射 ----------
OUTCOME_LABELS = {
    "正确率": "正确率",
    "主观_总分": "主观体验总分",
    "扫视log": "扫视次数(log)",
    "路径log": "总扫视路径(log)",
    "AOI注视": "AOI总注视时长",
    "瞳孔": "平均瞳孔直径(mm)",
    "眨眼log": "眨眼次数(log)",
}
