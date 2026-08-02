# -*- coding: utf-8 -*-
"""Analyze associations and internal consistency of dialog evaluation scores."""
from __future__ import annotations

import argparse
import json
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import kendalltau, spearmanr
from sklearn.decomposition import FactorAnalysis
from statsmodels.stats.multitest import multipletests

from extract_eval_scores import extract_score_row, read_summary_rows


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUTS_DIR = ROOT / "outputs"
DEFAULT_ANALYSIS_DIR = DEFAULT_OUTPUTS_DIR / "correlation_analysis"

SUBJECTIVE_COLUMNS = ["被理解感", "情绪缓解感", "个性化贴合度", "交流舒适度"]
EVALUATOR_COLUMNS = ["识别层评分", "理解层评分", "行动层评分"]
ANALYSIS_COLUMNS = SUBJECTIVE_COLUMNS + EVALUATOR_COLUMNS + ["主观评分均值"]

INTERNAL_NAMES = {
    "被理解感": "understood",
    "情绪缓解感": "relief",
    "个性化贴合度": "personalization",
    "交流舒适度": "comfort",
    "识别层评分": "recognition",
    "理解层评分": "understanding",
    "行动层评分": "action",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析主观评分与评估AI评分的相关性和内部一致性")
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR, help="模型结果根目录")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ANALYSIS_DIR, help="统计结果输出目录")
    parser.add_argument("--bootstrap", type=int, default=2000, help="按用户ID聚类的bootstrap次数")
    parser.add_argument("--seed", type=int, default=20260731, help="bootstrap随机种子")
    parser.add_argument("--min-model-n", type=int, default=30, help="分模型相关性所需的最小样本量")
    args = parser.parse_args()
    if args.bootstrap <= 0:
        raise ValueError("--bootstrap 必须为正整数")
    if args.min_model_n <= 2:
        raise ValueError("--min-model-n 必须大于2")
    args.outputs_dir = args.outputs_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    return args


def to_float(value: Any) -> float:
    if value is None or value == "":
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def normalized_id(value: Any) -> str:
    text = str(value).strip().lstrip("0")
    return text or "0"


def load_analysis_data(outputs_dir: Path) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for model_dir in sorted(outputs_dir.iterdir(), key=lambda path: path.name.casefold()):
        if not model_dir.is_dir() or model_dir.name.casefold() in {"manual", "correlation_analysis"}:
            continue
        summary_path = model_dir / "dialog_eval_summary.csv"
        if not summary_path.is_file():
            skipped.append({"模型名称": model_dir.name, "原因": "缺少 dialog_eval_summary.csv"})
            continue

        rows = read_summary_rows(summary_path)
        for row_no, row in enumerate(rows, start=1):
            try:
                output, scores = extract_score_row(row, row_no)
            except ValueError as exc:
                raise ValueError(f"处理 {summary_path} 失败: {exc}") from exc

            dimensions = [
                scores["understood"],
                scores["relief"],
                scores["personalization"],
                scores["comfort"],
            ]
            subjective_avg = float(np.mean(dimensions)) if all(value is not None for value in dimensions) else np.nan
            row_id = output.get("ID", "")
            pick_order = output.get("sample_pick_order", "")
            records.append(
                {
                    "模型名称": model_dir.name,
                    "ID": row_id,
                    "聚类ID": normalized_id(row_id),
                    "sample_pick_order": pd.to_numeric(pick_order, errors="coerce"),
                    "被理解感": scores["understood"],
                    "情绪缓解感": scores["relief"],
                    "个性化贴合度": scores["personalization"],
                    "交流舒适度": scores["comfort"],
                    "识别层评分": scores["recognition"],
                    "理解层评分": scores["understanding"],
                    "行动层评分": scores["action"],
                    "主观评分均值": subjective_avg,
                }
            )

    if not records:
        raise ValueError(f"没有在 {outputs_dir} 的模型子目录中找到可分析的评分")
    data = pd.DataFrame(records)
    for column in ANALYSIS_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["sample_pick_order"] = pd.to_numeric(data["sample_pick_order"], errors="coerce").astype("Int64")
    return data, skipped


def validate_score_ranges(data: pd.DataFrame) -> list[str]:
    messages: list[str] = []
    for column in SUBJECTIVE_COLUMNS:
        invalid = data[column].notna() & ~data[column].between(0, 3)
        if invalid.any():
            messages.append(f"{column} 有 {int(invalid.sum())} 条超出0-3范围")
    for column in EVALUATOR_COLUMNS:
        invalid = data[column].notna() & ~data[column].between(0, 2)
        if invalid.any():
            messages.append(f"{column} 有 {int(invalid.sum())} 条超出0-2范围")
    return messages


def descriptive_statistics(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in ANALYSIS_COLUMNS:
        values = data[column].dropna()
        rows.append(
            {
                "指标": column,
                "有效N": len(values),
                "缺失N": int(data[column].isna().sum()),
                "均值": values.mean(),
                "标准差": values.std(ddof=1),
                "中位数": values.median(),
                "第一四分位数": values.quantile(0.25),
                "第三四分位数": values.quantile(0.75),
                "最小值": values.min(),
                "最大值": values.max(),
            }
        )
    return pd.DataFrame(rows).round(4)


def score_distributions(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in SUBJECTIVE_COLUMNS + EVALUATOR_COLUMNS:
        valid = data[column].dropna()
        for value, count in valid.value_counts().sort_index().items():
            rows.append(
                {
                    "指标": column,
                    "分值": value,
                    "数量": int(count),
                    "占有效样本比例": count / len(valid) if len(valid) else np.nan,
                }
            )
    return pd.DataFrame(rows).round(6)


def correlation_matrix(data: pd.DataFrame, columns: list[str], method: str) -> pd.DataFrame:
    matrix = pd.DataFrame(np.nan, index=columns, columns=columns, dtype=float)
    for left in columns:
        for right in columns:
            pair = data[[left, right]].dropna()
            if left == right:
                value = 1.0 if len(pair) else np.nan
            elif len(pair) < 3 or pair[left].nunique() < 2 or pair[right].nunique() < 2:
                value = np.nan
            elif method == "spearman":
                value = float(spearmanr(pair[left], pair[right]).statistic)
            elif method == "kendall":
                value = float(kendalltau(pair[left], pair[right], variant="b").statistic)
            else:
                raise ValueError(f"未知相关方法: {method}")
            matrix.loc[left, right] = value
    return matrix


def bh_adjust(p_values: list[float]) -> list[float]:
    result = [np.nan] * len(p_values)
    valid_indices = [index for index, value in enumerate(p_values) if np.isfinite(value)]
    if valid_indices:
        adjusted = multipletests([p_values[index] for index in valid_indices], method="fdr_bh")[1]
        for index, value in zip(valid_indices, adjusted):
            result[index] = float(value)
    return result


def pairwise_internal_table(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for left, right in combinations(columns, 2):
        pair = data[[left, right]].dropna()
        if len(pair) >= 3 and pair[left].nunique() >= 2 and pair[right].nunique() >= 2:
            spearman = spearmanr(pair[left], pair[right])
            kendall = kendalltau(pair[left], pair[right], variant="b")
            rho, rho_p = float(spearman.statistic), float(spearman.pvalue)
            tau, tau_p = float(kendall.statistic), float(kendall.pvalue)
        else:
            rho = rho_p = tau = tau_p = np.nan
        rows.append(
            {
                "指标1": left,
                "指标2": right,
                "N": len(pair),
                "Spearman_rho": rho,
                "Spearman_p": rho_p,
                "Kendall_tau_b": tau,
                "Kendall_p": tau_p,
            }
        )
    table = pd.DataFrame(rows)
    table["Spearman_p_BH"] = bh_adjust(table["Spearman_p"].tolist())
    table["Kendall_p_BH"] = bh_adjust(table["Kendall_p"].tolist())
    return table.round(6)


def cluster_bootstrap_spearman(
    data: pd.DataFrame,
    left: str,
    right: str,
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    pair = data[["聚类ID", left, right]].dropna()
    groups = [group[[left, right]].to_numpy(dtype=float) for _, group in pair.groupby("聚类ID", sort=False)]
    if len(groups) < 2:
        return np.nan, np.nan, 0

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
        return np.nan, np.nan, 0
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return float(lower), float(upper), len(estimates)


def primary_associations(
    data: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for evaluator_column in EVALUATOR_COLUMNS:
        pair = data[["聚类ID", evaluator_column, "主观评分均值"]].dropna()
        spearman = spearmanr(pair[evaluator_column], pair["主观评分均值"])
        kendall = kendalltau(pair[evaluator_column], pair["主观评分均值"], variant="b")
        lower, upper, successful = cluster_bootstrap_spearman(
            data,
            evaluator_column,
            "主观评分均值",
            repetitions,
            rng,
        )
        rows.append(
            {
                "评估端指标": evaluator_column,
                "结果指标": "主观评分均值",
                "N": len(pair),
                "用户ID簇数": pair["聚类ID"].nunique(),
                "Spearman_rho": float(spearman.statistic),
                "Spearman_p": float(spearman.pvalue),
                "Spearman_95CI下限": lower,
                "Spearman_95CI上限": upper,
                "有效bootstrap次数": successful,
                "Kendall_tau_b": float(kendall.statistic),
                "Kendall_p": float(kendall.pvalue),
            }
        )
    table = pd.DataFrame(rows)
    table["Spearman_p_BH"] = bh_adjust(table["Spearman_p"].tolist())
    table["Kendall_p_BH"] = bh_adjust(table["Kendall_p"].tolist())
    return table.round(6)


def per_model_associations(data: pd.DataFrame, min_model_n: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, model_data in data.groupby("模型名称", sort=True):
        for evaluator_column in EVALUATOR_COLUMNS:
            pair = model_data[[evaluator_column, "主观评分均值"]].dropna()
            if len(pair) < min_model_n or pair[evaluator_column].nunique() < 2 or pair["主观评分均值"].nunique() < 2:
                continue
            spearman = spearmanr(pair[evaluator_column], pair["主观评分均值"])
            kendall = kendalltau(pair[evaluator_column], pair["主观评分均值"], variant="b")
            rows.append(
                {
                    "模型名称": model_name,
                    "评估端指标": evaluator_column,
                    "N": len(pair),
                    "Spearman_rho": float(spearman.statistic),
                    "Spearman_p": float(spearman.pvalue),
                    "Kendall_tau_b": float(kendall.statistic),
                    "Kendall_p": float(kendall.pvalue),
                }
            )
    table = pd.DataFrame(rows)
    if not table.empty:
        table["Spearman_p_BH"] = bh_adjust(table["Spearman_p"].tolist())
        table["Kendall_p_BH"] = bh_adjust(table["Kendall_p"].tolist())
    return table.round(6)


def cronbach_alpha(values: pd.DataFrame) -> float:
    complete = values.dropna()
    item_count = complete.shape[1]
    total_variance = complete.sum(axis=1).var(ddof=1)
    if item_count < 2 or total_variance <= 0:
        return np.nan
    return float(item_count / (item_count - 1) * (1 - complete.var(ddof=1).sum() / total_variance))


def mcdonald_omega(values: pd.DataFrame) -> tuple[float, list[float], list[float], int]:
    complete = values.dropna()
    if len(complete) < 10:
        return np.nan, [], [], len(complete)
    standardized = (complete - complete.mean()) / complete.std(ddof=0)
    model = FactorAnalysis(n_components=1, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(standardized)
    loadings = model.components_[0].astype(float)
    if loadings.sum() < 0:
        loadings *= -1
    uniqueness = model.noise_variance_.astype(float)
    numerator = float(loadings.sum() ** 2)
    denominator = numerator + float(uniqueness.sum())
    omega = numerator / denominator if denominator > 0 else np.nan
    return float(omega), loadings.tolist(), uniqueness.tolist(), len(complete)


def reliability_results(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    values = data[SUBJECTIVE_COLUMNS]
    complete_n = len(values.dropna())
    alpha = cronbach_alpha(values)
    omega, loadings, uniqueness, omega_n = mcdonald_omega(values)
    table = pd.DataFrame(
        [
            {"量表": "用户AI主观评分四分项", "指标": "Cronbach_alpha", "估计值": alpha, "完整样本N": complete_n},
            {"量表": "用户AI主观评分四分项", "指标": "McDonald_omega", "估计值": omega, "完整样本N": omega_n},
        ]
    ).round(6)
    details = {
        "omega估计方法": "对标准化四分项拟合单因子最大似然FactorAnalysis后计算omega total",
        "因子载荷": dict(zip(SUBJECTIVE_COLUMNS, [round(value, 6) for value in loadings])),
        "独特性方差": dict(zip(SUBJECTIVE_COLUMNS, [round(value, 6) for value in uniqueness])),
    }
    return table, details


def regression_results(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = ["模型名称", "聚类ID", "主观评分均值", *EVALUATOR_COLUMNS]
    frame = data[columns].dropna().rename(columns=INTERNAL_NAMES).copy()
    formula = "Q('主观评分均值') ~ recognition + understanding + action + C(Q('模型名称'))"
    model = smf.ols(formula, data=frame).fit(cov_type="cluster", cov_kwds={"groups": frame["聚类ID"]})

    standardized = frame.copy()
    standardized["subjective_z"] = (standardized["主观评分均值"] - standardized["主观评分均值"].mean()) / standardized["主观评分均值"].std(ddof=0)
    for column in ["recognition", "understanding", "action"]:
        standardized[f"{column}_z"] = (standardized[column] - standardized[column].mean()) / standardized[column].std(ddof=0)
    standardized_formula = "subjective_z ~ recognition_z + understanding_z + action_z + C(Q('模型名称'))"
    standardized_model = smf.ols(standardized_formula, data=standardized).fit(
        cov_type="cluster", cov_kwds={"groups": standardized["聚类ID"]}
    )

    rows: list[dict[str, Any]] = []
    labels = {"recognition": "识别层评分", "understanding": "理解层评分", "action": "行动层评分"}
    for internal, label in labels.items():
        ci = model.conf_int().loc[internal]
        standardized_name = f"{internal}_z"
        standardized_ci = standardized_model.conf_int().loc[standardized_name]
        rows.append(
            {
                "预测指标": label,
                "非标准化系数": model.params[internal],
                "聚类稳健标准误": model.bse[internal],
                "95CI下限": ci.iloc[0],
                "95CI上限": ci.iloc[1],
                "p值": model.pvalues[internal],
                "标准化系数": standardized_model.params[standardized_name],
                "标准化95CI下限": standardized_ci.iloc[0],
                "标准化95CI上限": standardized_ci.iloc[1],
                "标准化p值": standardized_model.pvalues[standardized_name],
            }
        )
    table = pd.DataFrame(rows)
    table["p值_BH"] = bh_adjust(table["p值"].tolist())
    table["标准化p值_BH"] = bh_adjust(table["标准化p值"].tolist())
    details = {
        "模型": "OLS，因变量为主观评分均值，三个评估端指标同时进入，控制模型固定效应",
        "标准误": "按用户ID聚类的稳健标准误",
        "N": int(model.nobs),
        "用户ID簇数": int(frame["聚类ID"].nunique()),
        "模型数": int(frame["模型名称"].nunique()),
        "R_squared": round(float(model.rsquared), 6),
        "adjusted_R_squared": round(float(model.rsquared_adj), 6),
    }
    return table.round(6), details


def save_heatmap(matrix: pd.DataFrame, output_path: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="white", font="Microsoft YaHei")
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "Spearman ρ", "shrink": 0.8},
        ax=ax,
    )
    ax.set_title("评分指标 Spearman 相关矩阵", pad=14, fontsize=14)
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data, skipped = load_analysis_data(args.outputs_dir)
    warnings_found = validate_score_ranges(data)

    model_counts = data.groupby("模型名称", as_index=False).agg(
        已测对话数=("ID", "size"),
        唯一用户ID数=("聚类ID", "nunique"),
    )
    descriptives = descriptive_statistics(data)
    distributions = score_distributions(data)
    spearman_matrix = correlation_matrix(data, ANALYSIS_COLUMNS, "spearman")
    kendall_matrix = correlation_matrix(data, ANALYSIS_COLUMNS, "kendall")
    primary = primary_associations(data, args.bootstrap, args.seed)
    subjective_internal = pairwise_internal_table(data, SUBJECTIVE_COLUMNS)
    evaluator_internal = pairwise_internal_table(data, EVALUATOR_COLUMNS)
    per_model = per_model_associations(data, args.min_model_n)
    reliability, reliability_details = reliability_results(data)
    regression, regression_details = regression_results(data)

    data.to_csv(args.output_dir / "analysis_dataset.csv", index=False, encoding="utf-8-sig")
    model_counts.to_csv(args.output_dir / "model_sample_counts.csv", index=False, encoding="utf-8-sig")
    descriptives.to_csv(args.output_dir / "descriptive_statistics.csv", index=False, encoding="utf-8-sig")
    distributions.to_csv(args.output_dir / "score_distributions.csv", index=False, encoding="utf-8-sig")
    spearman_matrix.round(6).to_csv(args.output_dir / "spearman_correlation_matrix.csv", encoding="utf-8-sig")
    kendall_matrix.round(6).to_csv(args.output_dir / "kendall_tau_b_matrix.csv", encoding="utf-8-sig")
    primary.to_csv(args.output_dir / "primary_associations.csv", index=False, encoding="utf-8-sig")
    subjective_internal.to_csv(args.output_dir / "subjective_internal_correlations.csv", index=False, encoding="utf-8-sig")
    evaluator_internal.to_csv(args.output_dir / "evaluator_internal_correlations.csv", index=False, encoding="utf-8-sig")
    per_model.to_csv(args.output_dir / "per_model_primary_correlations.csv", index=False, encoding="utf-8-sig")
    reliability.to_csv(args.output_dir / "subjective_scale_reliability.csv", index=False, encoding="utf-8-sig")
    regression.to_csv(args.output_dir / "model_fixed_effects_regression.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(skipped, columns=["模型名称", "原因"]).to_csv(
        args.output_dir / "skipped_model_directories.csv", index=False, encoding="utf-8-sig"
    )
    save_heatmap(spearman_matrix, args.output_dir / "spearman_correlation_heatmap.png")

    metadata = {
        "分析单位": "每次成功的模型-抽样位置对话",
        "总对话数": len(data),
        "唯一用户ID数": int(data["聚类ID"].nunique()),
        "模型数": int(data["模型名称"].nunique()),
        "主分析": "Spearman rho",
        "稳健性分析": "Kendall tau-b",
        "多重比较": "每个检验族内使用Benjamini-Hochberg FDR校正",
        "bootstrap": {"次数": args.bootstrap, "随机种子": args.seed, "聚类单位": "用户ID"},
        "分模型最小样本量": args.min_model_n,
        "评分范围警告": warnings_found,
        "跳过目录": skipped,
        "内部一致性": reliability_details,
        "回归": regression_details,
    }
    write_json(args.output_dir / "analysis_metadata.json", metadata)

    print(f"分析数据: {len(data)} 条，{data['模型名称'].nunique()} 个模型，{data['聚类ID'].nunique()} 个唯一用户ID")
    print(f"输出目录: {args.output_dir}")
    print("\n评估端三层评分与主观评分均值：")
    print(primary[["评估端指标", "N", "Spearman_rho", "Spearman_95CI下限", "Spearman_95CI上限", "Spearman_p_BH", "Kendall_tau_b"]].to_string(index=False))
    print("\n用户AI主观评分内部一致性：")
    print(reliability.to_string(index=False))
    print("\n控制模型固定效应后的回归：")
    print(regression[["预测指标", "标准化系数", "标准化95CI下限", "标准化95CI上限", "标准化p值_BH"]].to_string(index=False))
    if warnings_found:
        print("\n评分范围警告：")
        for message in warnings_found:
            print(f"- {message}")


if __name__ == "__main__":
    main()
