# -*- coding: utf-8 -*-
"""2.1 现有科普牌收集与版面编码:154 牌区域占比 → Jenks 二分类 / 四分位三分类。

标题层、要点层、图片占比:剔除极端异常值(|z|>3)后做 Jenks 自然断点二分类;
详情层:四分位数三分类。输出各层级断点与各类典型值。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from lib.load_data import load_material
from lib.stats_helpers import jenks, quartiles


def remove_extreme(values):
    """剔除极端异常值(|z|>3)。返回 (保留值, 剔除条数)。"""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    z = (v - v.mean()) / v.std(ddof=1)
    keep = v[np.abs(z) <= 3]
    return keep, len(v) - len(keep)


def class_means(values, breaks):
    """按断点分箱后求各类均值(低→高)。"""
    v = np.asarray(values, dtype=float)
    edges = [-np.inf] + list(breaks) + [np.inf]
    means = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        seg = v[(v >= lo) & (v < hi)]
        means.append(float(seg.mean()) if len(seg) else np.nan)
    # 最后一类包含右端点
    seg = v[v >= breaks[-1]]
    if len(seg):
        means[-1] = float(seg.mean())
    return means


def main():
    df = load_material()
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)

    pct_map = {
        "标题层": "标题层占比（%）",
        "要点层": "要点层占比（%）",
        "图片占比": "主要图片占比（%）",
        "详情层": "详情层占比（%）",
    }

    rows = []
    print("=" * 70)
    print("2.1 材料编码:154 牌区域占比 → 断点 / 典型值")
    print("=" * 70)

    # 标题/要点/图片:Jenks 二分类
    for layer, col in [("标题层", pct_map["标题层"]), ("要点层", pct_map["要点层"]),
                       ("图片占比", pct_map["图片占比"])]:
        keep, n_out = remove_extreme(df[col])
        brk = jenks(keep, k=2)
        means = class_means(keep, brk)
        print(f"\n【{layer}】原始 n={df[col].notna().sum()}, 剔除极端值 {n_out} 条, "
              f"保留 {len(keep)} 条")
        print(f"  Jenks 断点: {[f'{b:.2f}' for b in brk]}")
        print(f"  各类均值(低/高): {[f'{m:.2f}' for m in means]}")
        rows.append(dict(层级=layer, 方法="Jenks二分类", 断点1=round(brk[0], 2), 断点2="",
                         低类均值=round(means[0], 2), 高类均值=round(means[1], 2),
                         剔除异常=n_out))

    # 详情:四分位三分类
    keep, n_out = remove_extreme(df[pct_map["详情层"]])
    q = quartiles(keep)
    means = class_means(keep, q)
    print(f"\n【详情层】剔除极端值 {n_out} 条, 保留 {len(keep)} 条")
    print(f"  四分位断点(Q1/Q3): {[f'{b:.2f}' for b in q]}")
    print(f"  各类均值(低/中/高): {[f'{m:.2f}' for m in means]}")
    rows.append(dict(层级="详情层", 方法="四分位三分类", 断点1=round(q[0], 2),
                     断点2=round(q[1], 2), 低类均值=round(means[0], 2),
                     高类均值=round(means[2], 2), 剔除异常=n_out))

    out = pd.DataFrame(rows)
    out.to_csv(config.TABLE_DIR / "表2-1_材料编码断点.csv", index=False, encoding="utf-8-sig")
    print(f"\n已保存 → {config.TABLE_DIR / '表2-1_材料编码断点.csv'}")


if __name__ == "__main__":
    main()
