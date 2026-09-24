# -*- coding: utf-8 -*-
"""3.4/3.5 混合效应模型:LMM(6 结局) + 二项 GLMM(正确率)。

- LMM:Y ~ design + group + (1|subj),REML;ICC、总体设计 Wald(7df)、区组内 Wald。
- GLMM:C ~ Binomial(2, p),20 点 Gauss-Hermite;σ_b、总体 Wald、区组 Wald、7 个预定两两 OR + Holm。
输出 表 3-4/3-5/3-6 与 图 3-3。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from lib.load_data import load_master
from lib import mixed_models as mm
from lib.stats_helpers import holm, clustered_bootstrap_stat
from lib.plotting import design_color, style_axes, save_fig

# 6 个 LMM 结局(列名, 是否 log 变换, 标签)
LMM_OUTCOMES = [
    ("主观_总分", False, "主观体验总分"),
    ("扫视次数", True, "扫视次数(log)"),
    ("路径", True, "总扫视路径(log)"),
    ("AOI注视", False, "AOI总注视时长"),
    ("平均瞳孔直径_mm", False, "平均瞳孔直径"),
    ("眨眼次数", True, "眨眼次数(log)"),
]

BLOCKS = [("造山运动", ["B"]), ("红耳鹎", ["C", "D", "E"]), ("红果仔", ["F", "G", "H"])]

PAIRS = [("B", "A"), ("D", "C"), ("E", "C"), ("E", "D"), ("G", "F"), ("H", "F"), ("H", "G")]


def _outcome(df, col, logt):
    y = pd.to_numeric(df[col], errors="coerce")
    return np.log1p(y) if logt else y


def main():
    df = load_master()
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # ================= LMM =================
    print("=" * 78)
    print("LMM 结果(ICC | 总体 Wald 7df | 区组 Wald)")
    print("=" * 78)
    wald_rows, var_rows, block_rows = [], [], []
    for col, logt, label in LMM_OUTCOMES:
        y = _outcome(df, col, logt)
        fit = mm.fit_lmm(df, y)
        cols = fit["cols"]

        C_overall = mm.overall_design_C(len(cols), cols)
        W_overall = mm.wald(fit["fe"], fit["cov"], C_overall)

        block_W = {}
        for bname, bdesigns in BLOCKS:
            W = mm.wald(fit["fe"], fit["cov"], mm.block_C(cols, bdesigns))
            block_W[bname] = W

        # 残差偏度(条件残差,调整偏度 bias=False,匹配论文表 3-5)
        skew = float(pd.Series(mm.conditional_resid(df, fit)).skew())

        icc = fit["icc"]
        sd_subj = float(np.sqrt(fit["cov_re"]))
        print(f"\n{label}: 被试SD={sd_subj:.3f} ICC={icc:.3f} "
              f"总体Wald={W_overall['W']:.2f}(df={W_overall['df']},p={W_overall['p']:.3f}) "
              f"残差偏度={skew:.2f}")
        for bname, W in block_W.items():
            print(f"    区组 {bname}: χ²={W['W']:.2f}(df={W['df']},p={W['p']:.3f})")

        wald_rows.append(dict(结局=label, n=len(y), 收敛="是",
                              Wald=round(W_overall["W"], 2), df=W_overall["df"],
                              p=round(W_overall["p"], 4), ICC=round(icc, 3)))
        var_rows.append(dict(结局=label, n=len(y), 收敛="是",
                             被试SD=round(sd_subj, 3), ICC=round(icc, 3),
                             残差偏度=round(skew, 2)))
        for bname, W in block_W.items():
            block_rows.append(dict(结局=label, 区组=bname,
                                   Wald=round(W["W"], 2), df=W["df"],
                                   p=round(W["p"], 3)))

    # ================= GLMM =================
    print("\n" + "=" * 78)
    print("GLMM 正确率(答对题数 ~ Binomial(2,p),20 点 GH)")
    print("=" * 78)
    y_acc = pd.to_numeric(df["答对题数"], errors="coerce")
    gfit = mm.fit_glmm(df, y_acc, trials=2)
    cols = gfit["cols"]

    W_overall_g = mm.wald(gfit["beta"], gfit["cov"], mm.overall_design_C(len(cols), cols))
    print(f"σ_b={gfit['sigma']:.3f}  总体Wald={W_overall_g['W']:.2f}(df={W_overall_g['df']},p={W_overall_g['p']:.4f})")
    for bname, bdesigns in BLOCKS:
        W = mm.wald(gfit["beta"], gfit["cov"], mm.block_C(cols, bdesigns))
        print(f"  区组 {bname}: χ²={W['W']:.2f}(df={W['df']},p={W['p']:.4f})")

    # 7 个预定两两比较 OR + Holm
    pair_rows = []
    pvals = []
    for a, b in PAIRS:
        r = mm.glmm_pairwise_or(gfit, a, b)
        pvals.append(r["p"])
        pair_rows.append(r)
    holm_p = holm(pvals)
    for r, hp in zip(pair_rows, holm_p):
        r["Holm_p"] = round(float(hp), 4)

    print("\n两两比较 OR(前者/后者):")
    for r in pair_rows:
        print(f"  {r['a']}−{r['b']}: OR={r['OR']:.2f} [{r['ci_low']:.2f},{r['ci_high']:.2f}] "
              f"p={r['p']:.3f} Holm={r['Holm_p']:.4f}")

    wald_rows.append(dict(结局="正确率", n=len(y_acc), 收敛="是",
                          Wald=round(W_overall_g["W"], 2), df=W_overall_g["df"],
                          p=round(W_overall_g["p"], 4), ICC="—"))
    for bname, bdesigns in BLOCKS:
        W = mm.wald(gfit["beta"], gfit["cov"], mm.block_C(cols, bdesigns))
        block_rows.append(dict(结局="正确率", 区组=bname,
                               Wald=round(W["W"], 2), df=W["df"],
                               p=round(W["p"], 3)))

    # ---- 保存(结局顺序与论文一致:正确率在前) ----
    ORDER = ["正确率", "主观体验总分", "扫视次数(log)", "总扫视路径(log)",
             "AOI总注视时长", "平均瞳孔直径", "眨眼次数(log)"]
    BLOCK_ORDER = ["造山运动", "红耳鹎", "红果仔"]
    wald_df = pd.DataFrame(wald_rows).set_index("结局").reindex(ORDER).reset_index()
    wald_df.to_csv(config.TABLE_DIR / "表3-4_混合模型Wald_ICC.csv",
                   index=False, encoding="utf-8-sig")
    # 方差成分仅对 6 个 LMM 结局(正确率为 GLMM,无 ICC/被试SD/残差偏度)
    var_df = pd.DataFrame(var_rows).set_index("结局").reindex(ORDER[1:]).reset_index()
    var_df.to_csv(config.TABLE_DIR / "表3-4_方差成分.csv",
                  index=False, encoding="utf-8-sig")
    block_df = pd.DataFrame(block_rows)
    block_df["结局"] = pd.Categorical(block_df["结局"], categories=ORDER, ordered=True)
    block_df["区组"] = pd.Categorical(block_df["区组"], categories=BLOCK_ORDER, ordered=True)
    block_df = block_df.sort_values(["结局", "区组"]).reset_index(drop=True)
    block_df.to_csv(config.TABLE_DIR / "表3-5_区组Wald.csv",
                    index=False, encoding="utf-8-sig")
    pair_df = pd.DataFrame(pair_rows)
    pair_df = pair_df.rename(columns={"a": "前者", "b": "后者"})
    pair_df.to_csv(config.TABLE_DIR / "表3-6_两两比较OR.csv", index=False, encoding="utf-8-sig")

    # ---- 图 3-3:各内容区组内的知识正确率 ----
    ah = df[df["design"].isin(config.DESIGNS)].copy()
    blocks_plot = [("造山运动", ["A", "B"]), ("红耳鹎", ["C", "D", "E"]),
                   ("红果仔", ["F", "G", "H"])]
    fig, axes = __import__("matplotlib.pyplot", fromlist=["subplots"]).subplots(
        1, 3, figsize=(12, 4), sharey=True)
    for ax, (bname, ds) in zip(axes, blocks_plot):
        for d in ds:
            sub = ah[ah["design"] == d]
            stat = clustered_bootstrap_stat(sub, lambda s: s["正确率"].mean(), "正确率", B=2000)
            m, lo, hi = stat["mean"], stat["ci_low"], stat["ci_high"]
            x = ds.index(d)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="o", color=design_color(d),
                        markersize=7, capsize=4, elinewidth=1.4, zorder=3)
        ax.set_xticks(range(len(ds))); ax.set_xticklabels(ds)
        style_axes(ax, title=f"{bname}", xlabel="设计", ylabel="正确率")
    fig.suptitle("各内容—顺序区组内的知识正确率(点+95% Bootstrap CI)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, "图3-3_区组知识正确率.png")

    print("\n已保存 → 表3-4_混合模型Wald_ICC / 表3-4_方差成分 / 表3-5_区组Wald / 表3-6_两两比较OR + 图3-3")


if __name__ == "__main__":
    main()
