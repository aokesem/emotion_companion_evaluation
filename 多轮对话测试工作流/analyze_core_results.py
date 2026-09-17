# -*- coding: utf-8 -*-
"""Generate the focused preview tables for the final ElderComp-Bench analysis."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

from analyze_scores_by_concern import CONCERN_ORDER, load_concern_lookup, load_dialog_scores


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUTS_DIR = ROOT / "outputs"
DEFAULT_SIM_USER_JSONL = ROOT.parent / "用户画像数据" / "模拟用户信息_500.jsonl"
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUTS_DIR / "core_result_preview"

# Confirmed formal model set. Models below the current sample threshold remain excluded.
FORMAL_MODELS = [
    "deepseek_v4_flash",
    "deepseek_v4_pro",
    "doubao-seed-2-0-lite-260428",
    "doubao-seed-2-0-mini-260428",
    "doubao-seed-character-260628",
    "gemini-3.7-flash",
    "glm-5-2-260617",
    "gpt-5.6-terra",
    "gpt-5.6-sol",
    "moonshot-v1-8k",
    "Nemotron_3_Ultra",
]

SEQUENTIAL_PAIRS = [
    ("状态识别与情境理解", "识别层评分", "理解层评分"),
    ("情境理解与回应行动", "理解层评分", "行动层评分"),
    ("回应行动与用户体验", "行动层评分", "主观平均分"),
]

CORE_CONCERN_METRICS = ["识别层评分", "理解层评分", "行动层评分", "主观平均分"]
SCORE_LIMITS = {
    "被理解感": (0, 3),
    "情绪缓解感": (0, 3),
    "个性化贴合度": (0, 3),
    "交流舒适度": (0, 3),
    "识别层评分": (0, 2),
    "理解层评分": (0, 2),
    "行动层评分": (0, 2),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成顺序关联与烦恼分层的核心结果表")
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR, help="各模型结果根目录")
    parser.add_argument("--sim-user-jsonl", type=Path, default=DEFAULT_SIM_USER_JSONL, help="模拟用户信息 JSONL")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="CSV 输出目录")
    parser.add_argument("--bootstrap", type=int, default=5000, help="按用户 ID 聚类的 bootstrap 次数")
    parser.add_argument("--seed", type=int, default=20260914, help="bootstrap 随机种子")
    args = parser.parse_args()
    if args.bootstrap < 100:
        raise ValueError("--bootstrap 至少为 100")
    for name in ("outputs_dir", "sim_user_jsonl", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    return args


def cluster_bootstrap_spearman(
    data: pd.DataFrame,
    left: str,
    right: str,
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    pair = data[["标准化ID", left, right]].dropna()
    groups = [group[[left, right]].to_numpy(dtype=float) for _, group in pair.groupby("标准化ID", sort=False)]
    if len(groups) < 2:
        return np.nan, np.nan

    estimates: list[float] = []
    for _ in range(repetitions):
        selected = rng.integers(0, len(groups), size=len(groups))
        sample = np.concatenate([groups[index] for index in selected], axis=0)
        if np.unique(sample[:, 0]).size < 2 or np.unique(sample[:, 1]).size < 2:
            continue
        estimate = float(spearmanr(sample[:, 0], sample[:, 1]).statistic)
        if np.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        return np.nan, np.nan
    return tuple(float(value) for value in np.percentile(estimates, [2.5, 97.5]))


def model_adjusted_association(data: pd.DataFrame, left: str, right: str) -> tuple[float, float, float, float]:
    pair = data[["模型名称", "标准化ID", left, right]].dropna().copy()
    pair["x"] = (pair[left] - pair[left].mean()) / pair[left].std(ddof=0)
    pair["y"] = (pair[right] - pair[right].mean()) / pair[right].std(ddof=0)
    if pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return np.nan, np.nan, np.nan, np.nan

    fit = smf.ols("y ~ x + C(模型名称)", data=pair).fit(
        cov_type="cluster", cov_kwds={"groups": pair["标准化ID"]}
    )
    interval = fit.conf_int().loc["x"]
    return float(fit.params["x"]), float(fit.pvalues["x"]), float(interval.iloc[0]), float(interval.iloc[1])


def summarize_sequential_associations(data: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, float | str]] = []
    for label, left, right in SEQUENTIAL_PAIRS:
        pair = data[["标准化ID", left, right]].dropna()
        rho, p_value = spearmanr(pair[left], pair[right])
        ci_low, ci_high = cluster_bootstrap_spearman(data, left, right, args.bootstrap, rng)
        beta, beta_p, beta_low, beta_high = model_adjusted_association(data, left, right)
        rows.append(
            {
                "关系": label,
                "前序指标": left,
                "后续指标": right,
                "Spearman rho": float(rho),
                "Spearman p值": float(p_value),
                "rho 95%CI下限": ci_low,
                "rho 95%CI上限": ci_high,
                "控制模型差异后的标准化关联系数": beta,
                "控制模型差异后的p值": beta_p,
                "标准化关联系数95%CI下限": beta_low,
                "标准化关联系数95%CI上限": beta_high,
            }
        )
    return pd.DataFrame(rows).round(4)


def summarize_concern_strata(data: pd.DataFrame) -> pd.DataFrame:
    grouped = data.groupby(["模型名称", "烦恼类别"], observed=True)[CORE_CONCERN_METRICS].mean()
    complete_index = pd.MultiIndex.from_product(
        [FORMAL_MODELS, CONCERN_ORDER], names=["模型名称", "烦恼类别"]
    )
    return grouped.reindex(complete_index).reset_index().round(4)


def summarize_models_with_concern(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby("模型名称", observed=True)[["识别层评分", "理解层评分", "行动层评分"]]
        .mean()
        .reindex(FORMAL_MODELS)
        .reset_index()
        .round(4)
    )


def summarize_no_concern_recognition_understanding(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for model_name in FORMAL_MODELS:
        model_data = data[data["模型名称"] == model_name]
        row: dict[str, float | str] = {"模型名称": model_name}
        for metric, prefix in [("识别层评分", "识别层"), ("理解层评分", "理解层")]:
            values = pd.to_numeric(model_data[metric], errors="coerce").dropna()
            row[f"{prefix}平均分"] = values.mean() if not values.empty else np.nan
            for score in (0, 1, 2):
                row[f"{prefix}{score}分比例"] = (values == score).mean() if not values.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows).round(4)


def split_invalid_scores(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    invalid = pd.Series(False, index=data.index)
    reasons = pd.Series("", index=data.index, dtype="object")
    for metric, (lower, upper) in SCORE_LIMITS.items():
        values = pd.to_numeric(data[metric], errors="coerce")
        out_of_range = values.notna() & ~values.between(lower, upper)
        invalid |= out_of_range
        reasons.loc[out_of_range] += f"{metric} 超出 {lower}–{upper}；"
    audit_columns = ["模型名称", "ID", "sample_pick_order", "烦恼类别"] + list(SCORE_LIMITS)
    invalid_rows = data.loc[invalid, audit_columns].copy()
    invalid_rows["原因"] = reasons.loc[invalid].str.rstrip("；")
    return data.loc[~invalid].copy(), invalid_rows


def write_invalid_audit(path: Path, invalid_rows: pd.DataFrame) -> None:
    try:
        invalid_rows.to_csv(path, index=False, encoding="utf-8-sig")
    except PermissionError:
        print(f"警告：无法覆盖正在使用的审计文件，保留原文件：{path}")


def plot_concern_heatmap(summary: pd.DataFrame, output_dir: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="white", font="Microsoft YaHei")

    panel_config = [
        ("识别层评分", 2, "Blues"),
        ("理解层评分", 2, "Oranges"),
        ("行动层评分", 2, "Greens"),
        ("主观平均分", 3, "Purples"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(17, 13), constrained_layout=True)
    for axis, (metric, maximum, palette) in zip(axes.flat, panel_config):
        values = summary.pivot(index="模型名称", columns="烦恼类别", values=metric).reindex(
            index=FORMAL_MODELS, columns=CONCERN_ORDER
        )
        sns.heatmap(
            values,
            ax=axis,
            cmap=palette,
            vmin=0,
            vmax=maximum,
            annot=True,
            fmt=".2f",
            linewidths=0.7,
            linecolor="white",
            cbar_kws={"label": f"平均分（0–{maximum}）", "shrink": 0.82},
        )
        axis.set_title(metric, fontsize=14, pad=10)
        axis.set_xlabel("烦恼类别")
        axis.set_ylabel("模型")
        axis.tick_params(axis="x", rotation=25)
        axis.tick_params(axis="y", rotation=0)

    figure.suptitle("不同生活烦恼类别下的模型分层表现", fontsize=18)
    figure.savefig(output_dir / "烦恼类别分层热力图.png", dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(output_dir / "烦恼类别分层热力图.pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def write_metadata(
    path: Path,
    data: pd.DataFrame,
    skipped: list[dict[str, object]],
    invalid_rows: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    metadata = {
        "生成时间": datetime.now().isoformat(timespec="seconds"),
        "用途": "正式分析前的趋势预览；正式结论应在全部模型完成500次测试后重算。",
        "固定纳入模型": FORMAL_MODELS,
        "顺序关系解释": "统计关联，不表示状态识别、情境理解、回应行动和用户体验之间的因果影响。",
        "顺序关系主统计": "Spearman rho；95% CI 按用户ID聚类 bootstrap。",
        "稳健性统计": "控制模型固定效应的标准化 OLS 关联，标准误按用户ID聚类。",
        "烦恼分层指标": CORE_CONCERN_METRICS,
        "顺序关系补充分组": "排除“无明显生活烦恼”后，仅保留存在明确生活烦恼的对话。",
        "当前纳入的有效对话数": int(len(data)),
        "当前唯一用户ID数": int(data["标准化ID"].nunique()),
        "跳过记录数": len(skipped),
        "因超出量表范围而剔除的记录数": int(len(invalid_rows)),
        "bootstrap次数": args.bootstrap,
        "随机种子": args.seed,
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    concern_lookup = load_concern_lookup(args.sim_user_jsonl)
    raw_data, skipped = load_dialog_scores(FORMAL_MODELS, args.outputs_dir, concern_lookup)
    data, invalid_rows = split_invalid_scores(raw_data)
    if data.empty:
        raise ValueError("剔除超出量表范围的记录后，没有可分析的评分")

    sequential = summarize_sequential_associations(data, args)
    concern_data = data[data["烦恼类别"] != "无明显生活烦恼"].copy()
    if concern_data.empty:
        raise ValueError("排除无明显生活烦恼后，没有可分析的评分")
    sequential_with_concern = summarize_sequential_associations(concern_data, args)
    model_scores_with_concern = summarize_models_with_concern(concern_data)
    no_concern_data = data[data["烦恼类别"] == "无明显生活烦恼"].copy()
    if no_concern_data.empty:
        raise ValueError("没有可用于无明显生活烦恼分析的评分")
    no_concern_scores = summarize_no_concern_recognition_understanding(no_concern_data)
    concern_strata = summarize_concern_strata(data)
    sequential.to_csv(args.output_dir / "顺序关系分析.csv", index=False, encoding="utf-8-sig")
    sequential_with_concern.to_csv(
        args.output_dir / "顺序关系分析_排除无明显生活烦恼.csv", index=False, encoding="utf-8-sig"
    )
    model_scores_with_concern.to_csv(
        args.output_dir / "模型三层得分_排除无明显生活烦恼.csv", index=False, encoding="utf-8-sig"
    )
    no_concern_scores.to_csv(
        args.output_dir / "无明显生活烦恼_识别理解得分及分布.csv", index=False, encoding="utf-8-sig"
    )
    concern_strata.to_csv(args.output_dir / "烦恼类别分层表现.csv", index=False, encoding="utf-8-sig")
    write_invalid_audit(args.output_dir / "异常评分记录.csv", invalid_rows)
    plot_concern_heatmap(concern_strata, args.output_dir)
    write_metadata(args.output_dir / "分析说明.json", data, skipped, invalid_rows, args)

    print(f"纳入模型: {len(FORMAL_MODELS)} 个")
    print(f"结果目录: {args.output_dir}")
    print("\n顺序关系分析：")
    print(sequential.to_string(index=False))


if __name__ == "__main__":
    main()
