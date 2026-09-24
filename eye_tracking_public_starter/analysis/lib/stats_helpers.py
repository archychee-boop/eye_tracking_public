# -*- coding: utf-8 -*-
"""统计辅助:重复测量相关、Cronbach α、多重比较校正、聚类 Bootstrap、Jenks 自然断点。"""
import numpy as np
import pandas as pd
from scipy import stats


# ---------- 重复测量相关 / Cronbach ----------
def rm_corr(data: pd.DataFrame, x: str, y: str, subject: str = "subj"):
    """Bakdash & Marusich 重复测量相关(论文式 2-9/2-10)。"""
    d = data[[x, y, subject]].dropna()
    xc = d[x].astype(float) - d.groupby(subject)[x].transform("mean")
    yc = d[y].astype(float) - d.groupby(subject)[y].transform("mean")
    r = float(np.corrcoef(xc, yc)[0, 1])
    dof = len(d) - d[subject].nunique() - 1
    t = r * np.sqrt(dof / (1 - r * r))
    return dict(r=r, df=int(dof), p=float(2 * stats.t.sf(abs(t), dof)))


def cronbach_alpha(data: pd.DataFrame) -> float:
    d = data.dropna().astype(float)
    k = d.shape[1]
    return float(k / (k - 1) * (1 - d.var(axis=0, ddof=1).sum()
                                  / d.sum(axis=1).var(ddof=1)))


# ---------- 多重比较校正 ----------
def fdr_bh(pvals):
    """Benjamini-Hochberg FDR(线性插值版,与 R p.adjust(method='BH') 一致)。"""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    rank = np.arange(1, n + 1)
    q = p[order] * n / rank
    # 保证单调不减(反向取累积最小)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n)
    out[order] = q
    return out


def holm(pvals):
    """Holm-Bonferroni 逐步法:调整 p_(i)=max_{j≤i}(n−j+1)·p_(j)(前向累积最大)。"""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    m = np.arange(n, 0, -1)  # n, n-1, ..., 1
    adj = np.empty(n)
    adj[order] = np.maximum.accumulate(p[order] * m)
    return np.clip(adj, 0, 1)


# ---------- 聚类 Bootstrap ----------
def clustered_bootstrap_stat(data: pd.DataFrame, stat_fn, value_col: str,
                             cluster_col: str = "subj", B: int = 2000,
                             seed: int = 2026):
    """按被试整块重抽样,计算 value_col 均值的 2.5%/97.5% 分位 CI。

    本复现包中该函数仅用于「均值」统计量(stat_fn 仅用于点估计)。
    返回 dict(mean, ci_low, ci_high, draws)。numpy 拼接避免 pandas 循环开销。
    """
    rng = np.random.default_rng(seed)
    clusters = np.unique(data[cluster_col].values)
    nc = len(clusters)
    vals = [data.loc[data[cluster_col] == c, value_col].to_numpy(dtype=float)
            for c in clusters]
    vals = [v[np.isfinite(v)] for v in vals]
    draws = np.empty(B)
    for b in range(B):
        picks = rng.integers(0, nc, size=nc)
        sample = np.concatenate([vals[i] for i in picks])
        draws[b] = np.mean(sample) if sample.size else np.nan
    return dict(mean=stat_fn(data), ci_low=float(np.percentile(draws, 2.5)),
                ci_high=float(np.percentile(draws, 97.5)), draws=draws)


def clustered_bootstrap_rank(data: pd.DataFrame, value_col: str, group_col: str,
                             cluster_col: str = "subj", B: int = 2000,
                             seed: int = 2026):
    """聚类 Bootstrap:每个设计 Ḡ_d 的 95%CI、P(排名=1)、P(排名≤3)。"""
    rng = np.random.default_rng(seed)
    clusters = data[cluster_col].unique()
    nc = len(clusters)
    groups = sorted(data[group_col].unique())
    gbar_draws = {g: np.empty(B) for g in groups}
    for b in range(B):
        idx = rng.integers(0, nc, size=nc)
        sampled = pd.concat([data[data[cluster_col] == clusters[i]] for i in idx],
                            ignore_index=True)
        gbar = sampled.groupby(group_col)[value_col].mean()
        for g in groups:
            gbar_draws[g][b] = gbar.get(g, np.nan)
    ranks = np.empty((B, len(groups)))
    for b in range(B):
        vals = np.array([gbar_draws[g][b] for g in groups])
        # 排名 1 = 最大值(降序)
        ranks[b] = len(groups) - stats.rankdata(vals, method="min") + 1
    result = {}
    for gi, g in enumerate(groups):
        result[g] = dict(
            mean=float(gbar_draws[g].mean()),
            ci_low=float(np.percentile(gbar_draws[g], 2.5)),
            ci_high=float(np.percentile(gbar_draws[g], 97.5)),
            p_rank1=float(np.mean(ranks[:, gi] == 1)),
            p_rank_le3=float(np.mean(ranks[:, gi] <= 3)),
        )
    return result


# ---------- Jenks 自然断点(Fisher 动态规划) ----------
def jenks(values, k: int):
    """对一维数值做自然断点分类(k 类),返回断点列表(升序)。"""
    v = np.sort(np.asarray(values, dtype=float))
    n = len(v)
    # 累积和
    cumsum = np.concatenate([[0.0], np.cumsum(v)])
    cumsum2 = np.concatenate([[0.0], np.cumsum(v * v)])

    # SSE[i][j]:i..j(含) 的组内离差平方和
    def sse(i, j):
        s = cumsum[j + 1] - cumsum[i]
        s2 = cumsum2[j + 1] - cumsum2[i]
        m = j - i + 1
        return s2 - s * s / m

    # DP[k][j]:前 j+1 个元素分 k 类的最小 SSE
    INF = np.inf
    dp = np.full((k, n), INF)
    split = np.zeros((k, n), dtype=int)
    for j in range(n):
        dp[0, j] = sse(0, j)
    for kk in range(1, k):
        for j in range(kk, n):
            best = INF
            best_i = kk
            for i in range(kk, j + 1):
                cost = dp[kk - 1, i - 1] + sse(i, j)
                if cost < best:
                    best = cost
                    best_i = i
            dp[kk, j] = best
            split[kk, j] = best_i
    # 回溯得到断点(类边界值)
    breaks = []
    j = n - 1
    for kk in range(k - 1, 0, -1):
        i = split[kk, j]
        breaks.append(v[i])
        j = i - 1
    breaks = sorted(breaks)
    return breaks


def quartiles(values):
    """四分位三分类断点(Q1, Q3)。"""
    return [float(np.quantile(values, 0.25)), float(np.quantile(values, 0.75))]


# ---------- Spearman 排名相关 ----------
def spearman_rank(x, y):
    rho, p = stats.spearmanr(x, y)
    return dict(rho=float(rho), p=float(p))
