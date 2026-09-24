# -*- coding: utf-8 -*-
"""混合效应模型:LMM(statsmodels MixedLM)、二项 GLMM(20 点 Gauss-Hermite)、Wald 检验。

设计因素 9 水平(A–H + Z,参照 A),组别 3 水平(参照 组1),随机截距(1|被试)。
- 总体设计 Wald 为 7 df:仅检验 B–H 7 个虚拟变量(排除锚点 Z)。
- 区组内 Wald:造山 A/B、红耳鹎 C/D/E、红果仔 F/G/H。
- GLMM 采用论文式 2-13 指定的 20 点 Gauss-Hermite 求积,自行实现边际似然。
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, roots_hermite
from statsmodels.regression.mixed_linear_model import MixedLM

import config

N_GH = 20


# ---------- 设计矩阵 ----------
def build_exog(df: pd.DataFrame) -> pd.DataFrame:
    """design(9 水平,参照 A) + group(3 水平,参照 组1),drop_first。"""
    d = df.copy()
    d["design"] = pd.Categorical(d["design"], categories=config.DESIGNS_ALL, ordered=True)
    d["group"] = pd.Categorical(d["group"], categories=["组1", "组2", "组3"], ordered=True)
    exog = pd.get_dummies(d[["design", "group"]], drop_first=True).astype(float)
    exog.insert(0, "Intercept", 1.0)
    return exog


def design_col(cols, design: str):
    """返回 design 虚拟变量列索引;参照 A 无虚拟变量(系数恒为 0),返回 None。"""
    if design == "A":
        return None
    return cols.index(f"design_{design}")


# ---------- Wald 检验 ----------
def wald(beta, cov, C):
    """Wald χ² 检验。C 为对比矩阵(每行一个线性约束)。"""
    beta = np.asarray(beta, float)
    cov = np.asarray(cov, float)
    C = np.asarray(C, float)
    d = C @ beta
    V = C @ cov @ C.T
    W = float(d @ np.linalg.solve(V, d))
    df = int(np.linalg.matrix_rank(C))
    return dict(W=W, df=df, p=float(stats.chi2.sf(W, df)))


def overall_design_C(ncol: int, cols):
    """总体设计 Wald 对比(7 df:B–H 虚拟变量=0)。"""
    C = np.zeros((7, ncol))
    for j, d in enumerate("BCDEFGH"):
        C[j, design_col(cols, d)] = 1.0
    return C


def block_C(cols, designs):
    """区组内设计差异对比。designs 例如 ['C','D','E']。"""
    idx = [design_col(cols, d) for d in designs]
    R = np.zeros((len(idx), len(cols)))
    R[np.arange(len(idx)), idx] = 1.0
    return R if len(idx) == 1 else R[1:] - R[0]


def pairwise_C(cols, a, b):
    """设计 a 相对 b 的对比向量(β_a − β_b;参照 A 系数为 0)。"""
    C = np.zeros(len(cols))
    ia, ib = design_col(cols, a), design_col(cols, b)
    if ia is not None:
        C[ia] += 1.0
    if ib is not None:
        C[ib] -= 1.0
    return C


# ---------- 线性混合模型 ----------
def fit_lmm(df: pd.DataFrame, y: pd.Series):
    """随机截距 LMM(REML)。返回 fe/cov/cov_re/scale/icc 及列名。"""
    exog = build_exog(df)
    model = MixedLM(y.values, exog, groups=df["subj"])
    res = model.fit(reml=True)
    fe = res.fe_params.values
    cov = res.cov_params().values[: len(fe), : len(fe)]
    cov_re = np.asarray(res.cov_re).ravel()[0]
    scale = res.scale
    icc = cov_re / (cov_re + scale)
    return dict(fe=fe, cov=cov, cov_re=cov_re, scale=scale, icc=icc,
                cols=exog.columns.tolist(), exog=exog, res=res)


def conditional_resid(df, fit):
    """条件残差 = y − Xβ − b_i(statsmodels 的 res.resid 即为此定义)。

    MixedLM.resid 返回的已是扣除随机截距 BLUP 后的条件残差,
    其偏度即论文表 3-5 报告的「残差偏度」。
    """
    return fit["res"].resid


# ---------- 二项 GLMM(Gauss-Hermite) ----------
def _glmm_nll(theta, X, y, ididx, n_subj, trials=2):
    beta = theta[:-1]
    sigma = np.exp(theta[-1])
    node_b = np.sqrt(2.0) * sigma * roots_hermite(N_GH)[0]
    logw = np.log(roots_hermite(N_GH)[1])
    total = 0.0
    for gi in range(n_subj):
        idx = np.where(ididx == gi)[0]
        Xg, yg = X[idx], y[idx]
        logs = np.empty(N_GH)
        for k in range(N_GH):
            p = expit(Xg @ beta + node_b[k])
            logs[k] = logw[k] + np.sum(yg * np.log(p) + (trials - yg) * np.log(1.0 - p))
        total += logsumexp(logs) - 0.5 * np.log(np.pi)
    return -total


def fit_glmm(df: pd.DataFrame, y: pd.Series, trials: int = 2):
    """二项 GLMM,20 点 Gauss-Hermite。y = 答对题数(0..trials)。"""
    exog = build_exog(df)
    X = exog.values
    cols = exog.columns.tolist()
    subj = df["subj"].values
    uniq = np.unique(subj)
    idmap = {s: i for i, s in enumerate(uniq)}
    ididx = np.array([idmap[s] for s in subj])
    yv = y.values.astype(float)

    theta0 = np.zeros(len(cols) + 1)  # 最后一项 = log(sigma)
    res = minimize(_glmm_nll, theta0, args=(X, yv, ididx, len(uniq), trials),
                   method="BFGS", options={"maxiter": 5000, "gtol": 1e-6})
    theta = res.x
    beta = theta[:-1]
    sigma = float(np.exp(theta[-1]))

    # 数值 Hessian(观测信息矩阵)-> 参数协方差
    k = len(theta)
    eps = 1e-4
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            ei, ej = np.zeros(k), np.zeros(k)
            ei[i] = ej[j] = eps
            fpp = _glmm_nll(theta + ei + ej, X, yv, ididx, len(uniq), trials)
            fpm = _glmm_nll(theta + ei - ej, X, yv, ididx, len(uniq), trials)
            fmp = _glmm_nll(theta - ei + ej, X, yv, ididx, len(uniq), trials)
            fmm = _glmm_nll(theta - ei - ej, X, yv, ididx, len(uniq), trials)
            H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4 * eps * eps)
    cov_all = np.linalg.inv(H)
    cov = cov_all[: len(beta), : len(beta)]
    return dict(beta=beta, cov=cov, sigma=sigma, cols=cols, exog=exog, converged=bool(res.success))


def glmm_pairwise_or(fit, a, b):
    """设计 a 相对 b 的优势比 OR=exp(β_a−β_b) 及 95%CI。"""
    C = pairwise_C(fit["cols"], a, b)
    diff = float(C @ fit["beta"])
    se = float(np.sqrt(C @ fit["cov"] @ C.T))
    return dict(a=a, b=b, diff=diff, se=se,
                OR=float(np.exp(diff)),
                ci_low=float(np.exp(diff - 1.96 * se)),
                ci_high=float(np.exp(diff + 1.96 * se)),
                p=float(2 * stats.norm.sf(abs(diff / se))))
