# -*- coding: utf-8 -*-
"""3.1 数据质量:样本结构、缺失/重复/负值检查、最小瞳孔 0 值处理。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from lib.load_data import load_master


def main():
    df = load_master()
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    n = len(df)
    n_subj = df["subj"].nunique()
    rows.append(dict(指标="观看记录总数", 值=n, 备注="虚构示例数据"))
    rows.append(dict(指标="被试人数", 值=n_subj, 备注="虚构编号"))

    print("=" * 66)
    print("3.1 数据质量")
    print("=" * 66)
    print(f"观看记录 {n} 条,被试 {n_subj} 人")

    # 组别 / 设计计数
    print("\n【组别计数】")
    gc = df.groupby("group")["subj"].nunique()
    for g in ["组1", "组2", "组3"]:
        print(f"  {g}: 被试 {gc[g]}(期望 21/20/20), 记录 {len(df[df['group']==g])}")

    print("\n【设计计数】(期望 A=41, B–H=20/21, Z=61)")
    dc = df.groupby("design").size()
    for d in config.DESIGNS_ALL:
        print(f"  {d}({config.DESIGN_CONTENT[d]}): {dc.get(d, 0)}")

    # 缺失值
    na_counts = df.isna().sum()
    na_cols = na_counts[na_counts > 0]
    print(f"\n【缺失值】列缺失: {len(na_cols)} 列")
    if len(na_cols):
        print(na_cols.to_string())

    # 完全重复行
    dup = df.duplicated(subset=[c for c in df.columns if c != "subj"]).sum()
    print(f"\n【完全重复行】{dup} 条")

    # 负时长/负次数检查
    neg_cols = []
    for c in ["扫视次数", "眨眼次数", "平均瞳孔直径_mm", "总扫视时间_s"]:
        if (df[c] < 0).any():
            neg_cols.append((c, int((df[c] < 0).sum())))
    print(f"\n【负值检查】{neg_cols if neg_cols else '无负值'}")

    # 最小瞳孔 0 值
    zero_pupil = int((df["最小瞳孔直径_mm"] == 0).sum())
    print(f"\n【最小瞳孔直径 = 0】{zero_pupil} 条(按原分析规则置 NaN)")
    rows.append(dict(指标="最小瞳孔直径=0(置NaN)", 值=zero_pupil, 备注="模拟数据"))

    rows.extend([
        dict(指标="组1被试", 值=int(gc["组1"]), 备注="期望 21"),
        dict(指标="组2被试", 值=int(gc["组2"]), 备注="期望 20"),
        dict(指标="组3被试", 值=int(gc["组3"]), 备注="期望 20"),
        dict(指标="完全重复行", 值=dup, 备注=""),
    ])

    out = pd.DataFrame(rows)
    out.to_csv(config.TABLE_DIR / "表3-1_数据质量.csv", index=False, encoding="utf-8-sig")
    print(f"\n已保存 → {config.TABLE_DIR / '表3-1_数据质量.csv'}")


if __name__ == "__main__":
    main()
