# -*- coding: utf-8 -*-
"""数据读取与派生变量。

主数据表:244 条观看记录 × 52 列(见 README「数据来源」)。
派生变量:
  - 总扫视路径长度(px) = 扫视次数 × 平均眼跳绝对距离_px(论文式 2-19)
  - AOI总注视时长 / AOI注视次数 = 四层之和
  - log 变换:log(1 + x)
"""
import numpy as np
import pandas as pd

import config


def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_master() -> pd.DataFrame:
    """读取总表并生成派生列。返回 DataFrame。"""
    df = _clean_cols(pd.read_csv(config.DATA_MASTER, encoding="utf-8-sig"))
    df["subj"] = df["受试编号"].astype(str)
    df["design"] = df["设计代号"].astype(str).str.strip()
    df["group"] = df["组别"].astype(str).str.strip()
    df["content"] = df["design"].map(config.DESIGN_CONTENT)

    # 数值列统一转 float
    for c in [
        "正确率", "答对题数", "主观_总分",
        "扫视次数", "眨眼次数", "平均瞳孔直径_mm", "最小瞳孔直径_mm", "最大瞳孔直径_mm",
        "平均眼跳绝对距离_px",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 派生:总扫视路径长度
    df["路径"] = df["扫视次数"] * df["平均眼跳绝对距离_px"]

    # 派生:AOI 总和(四层)
    dur_cols = [f"总注视时长_{l}" for l in config.LAYERS]
    fix_cols = [f"注视次数_{l}" for l in config.LAYERS]
    df["AOI注视"] = df[dur_cols].astype(float).sum(axis=1)
    df["AOI注视次数"] = df[fix_cols].astype(float).sum(axis=1)

    # log 变换
    df["扫视log"] = np.log1p(df["扫视次数"])
    df["路径log"] = np.log1p(df["路径"])
    df["眨眼log"] = np.log1p(df["眨眼次数"])

    # 最小瞳孔直径 = 0 视为无效(缺失),置 NaN(论文 3.1,9 条)
    df["最小瞳孔直径_clean"] = df["最小瞳孔直径_mm"].replace(0.0, np.nan)

    return df


def load_material() -> pd.DataFrame:
    """读取 154 块科普牌区域占比统计表(权威 sheet「科普牌区域占比统计-总表」)。

    该工作簿另含 动物类/植物类/自然现象类 三个分类 sheet(合计同样 154 行),
    是总表的拆分,避免重复计入故仅读总表。
    """
    df = _clean_cols(pd.read_excel(config.DATA_MATERIAL, sheet_name="科普牌区域占比统计-总表"))
    df["类别"] = df["类型"].astype(str)
    # 面积比转为 0-100 数值
    pct_cols = ["主要图片占比（%）", "标题层占比（%）", "要点层占比（%）", "详情层占比（%）"]
    for c in pct_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_design_params() -> pd.DataFrame:
    """设计参数(硬编码于 config,返回长表便于对照)。"""
    rows = []
    for d, layers in config.DESIGN_AREA_PCT.items():
        for l in config.LAYERS:
            rows.append({"design": d, "layer": l, "area_pct": layers[l]})
    return pd.DataFrame(rows)
