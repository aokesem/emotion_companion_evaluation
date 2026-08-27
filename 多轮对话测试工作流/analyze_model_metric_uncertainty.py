# -*- coding: utf-8 -*-
"""Compute model-level metric means, SDs, and cluster-bootstrap confidence intervals."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analyze_scores_by_concern import DEFAULT_MODEL_SUMMARY, DEFAULT_OUTPUTS_DIR, load_model_names, normalized_id
from extract_eval_scores import extract_score_row, read_summary_rows


DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUTS_DIR / "model_metric_uncertainty"

SUBJECTIVE_METRICS = ["被理解感", "情绪缓解感", "个性化贴合度", "交流舒适度", "主观平均分"]
EVALUATOR_METRICS = ["识别层评分", "理解层评分", "行动层评分", "评估端平均分"]
ALL_METRICS = SUBJECTIVE_METRICS + EVALUATOR_METRICS

SCORE_KEYS = {
    "被理解感": "understood",
    "情绪缓解感": "relief",
    "个性化贴合度": "personalization",
    "交流舒适度": "comfort",
    "识别层评分": "recognition",
    "理解层评分": "understanding",
    "行动层评分": "action",
}

PLOT_COLORS = {
    "被理解感": "#256D85",
    "情绪缓解感": "#3E8E7E",
    "个性化贴合度": "#D28C3C",
    "交流舒适度": "#B85555",
    "主观平均分": "#252525",
    "识别层评分": "#2878B5",
    "理解层评分": "#E07A38",
    "行动层评分": "#4C956C",
    "评估端平均分": "#252525",
}

MARKERS = {
    "被理解感": "o",
    "情绪缓解感": "s",
    "个性化贴合度": "^",
    "交流舒适度": "D",
    "主观平均分": "P",
    "识别层评分": "o",
    "理解层评分": "s",
    "行动层评分": "^",
    "评估端平均分": "P",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="计算各模型七项指标及两端平均分的均值、标准差和置信区间")
    parser.add_argument(
        "--all-model-summary",
        type=Path,
        default=DEFAULT_MODEL_SUMMARY,
        help="用于确定模型名单的 all_model_score_summary CSV",
    )
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR, help="各模型结果目录的根目录")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="统计表和图表输出目录")
    parser.add_argument("--bootstrap-samples", type=int, default=5000, help="聚类 bootstrap 重抽样次数")
    parser.add_argument("--confidence-level", type=float, default=0.95, help="置信区间水平")
    parser.add_argument("--seed", type=int, default=20260814, help="bootstrap 随机种子")
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples 至少为 100")
    if not 0 < args.confidence_level < 1:
        raise ValueError("--confidence-level 必须在 0 和 1 之间")
    for name in ["all_model_summary", "outputs_dir", "output_dir"]:
        setattr(args, name, getattr(args, name).resolve())
    return args


def complete_mean(values: list[float | None]) -> float:
    if not values or any(value is None for value in values):
        return np.nan
    return float(np.mean(values))


def load_dialog_metrics(
    model_names: list[str], outputs_dir: Path
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for model_name in model_names:
        source = outputs_dir / model_name / "dialog_eval_summary.csv"
        if not source.is_file():
            skipped.append({"模型名称": model_name, "行号": "", "原因": "缺少 dialog_eval_summary.csv"})
            continue
        try:
            rows = read_summary_rows(source)
        except Exception as exc:  # noqa: BLE001
            skipped.append({"模型名称": model_name, "行号": "", "原因": f"读取失败: {exc}"})
            continue

        for row_no, row in enumerate(rows, start=1):
            try:
                output, scores = extract_score_row(row, row_no)
            except Exception as exc:  # noqa: BLE001
                skipped.append({"模型名称": model_name, "行号": row_no, "原因": f"评分解析失败: {exc}"})
                continue

            row_id = normalized_id(output.get("ID", ""))
            if row_id == "0":
                skipped.append({"模型名称": model_name, "行号": row_no, "原因": "缺少有效 ID"})
                continue
            subjective_values = [
                scores["understood"],
                scores["relief"],
                scores["personalization"],
                scores["comfort"],
            ]
            evaluator_values = [scores["recognition"], scores["understanding"], scores["action"]]
            record: dict[str, Any] = {
                "模型名称": model_name,
                "ID": output.get("ID", ""),
                "标准化ID": row_id,
                "sample_pick_order": pd.to_numeric(output.get("sample_pick_order", ""), errors="coerce"),
                "主观平均分": complete_mean(subjective_values),
                "评估端平均分": complete_mean(evaluator_values),
            }
            for label, score_key in SCORE_KEYS.items():
                record[label] = scores[score_key]
            records.append(record)

    if not records:
        raise ValueError("没有可用于统计的有效对话评分")
    data = pd.DataFrame(records)
    data["sample_pick_order"] = pd.to_numeric(data["sample_pick_order"], errors="coerce").astype("Int64")
    return data, skipped


def cluster_bootstrap_ci(
    frame: pd.DataFrame,
    metric: str,
    bootstrap_samples: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    valid = frame[["标准化ID", metric]].copy()
    valid[metric] = pd.to_numeric(valid[metric], errors="coerce")
    valid = valid.dropna(subset=[metric])
    grouped = valid.groupby("标准化ID", sort=False)[metric].agg(["sum", "count"])
    cluster_count = len(grouped)
    if cluster_count < 2:
        return np.nan, np.nan, cluster_count

    sums = grouped["sum"].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    bootstrap_means = np.empty(bootstrap_samples, dtype=float)
    for start in range(0, bootstrap_samples, 500):
        batch_size = min(500, bootstrap_samples - start)
        sampled = rng.integers(0, cluster_count, size=(batch_size, cluster_count))
        bootstrap_means[start : start + batch_size] = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)

    alpha = 1 - confidence_level
    lower, upper = np.quantile(bootstrap_means, [alpha / 2, 1 - alpha / 2])
    return float(lower), float(upper), cluster_count


def calculate_statistics(
    data: pd.DataFrame,
    model_names: list[str],
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seed_sequence = np.random.SeedSequence(seed)
    child_seeds = iter(seed_sequence.spawn(len(model_names) * len(ALL_METRICS)))
    for model_name in model_names:
        model_data = data[data["模型名称"] == model_name]
        for metric in ALL_METRICS:
            values = pd.to_numeric(model_data[metric], errors="coerce").dropna()
            rng = np.random.default_rng(next(child_seeds))
            ci_lower, ci_upper, cluster_count = cluster_bootstrap_ci(
                model_data,
                metric,
                bootstrap_samples,
                confidence_level,
                rng,
            )
            rows.append(
                {
                    "模型名称": model_name,
                    "评分端": "用户端" if metric in SUBJECTIVE_METRICS else "评估端",
                    "指标": metric,
                    "均值": values.mean(),
                    "标准差": values.std(ddof=1),
                    "置信区间下限": ci_lower,
                    "置信区间上限": ci_upper,
                    "有效样本数": values.count(),
                    "有效用户ID数": cluster_count,
                }
            )
    return pd.DataFrame(rows).round(6)


def setup_plot_style() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", font="Microsoft YaHei")


def plot_metric_intervals(
    statistics: pd.DataFrame,
    model_names: list[str],
    metrics: list[str],
    score_upper: float,
    title: str,
    filename: str,
    output_dir: Path,
) -> None:
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(17, 8.5), constrained_layout=True)
    x = np.arange(len(model_names), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(metrics))

    for offset, metric in zip(offsets, metrics):
        subset = statistics[statistics["指标"] == metric].set_index("模型名称").reindex(model_names)
        means = subset["均值"].to_numpy(dtype=float)
        lower = subset["置信区间下限"].to_numpy(dtype=float)
        upper = subset["置信区间上限"].to_numpy(dtype=float)
        yerr = np.vstack([means - lower, upper - means])
        ax.errorbar(
            x + offset,
            means,
            yerr=yerr,
            fmt=MARKERS[metric],
            markersize=7.5 if "平均分" not in metric else 9,
            color=PLOT_COLORS[metric],
            ecolor=PLOT_COLORS[metric],
            elinewidth=1.3,
            capsize=3,
            capthick=1.1,
            linestyle="none",
            label=metric,
            zorder=3,
        )

    ax.set_xlim(-0.65, len(model_names) - 0.35)
    ax.set_ylim(0, score_upper)
    ax.set_xticks(x, model_names, rotation=22, ha="right")
    ax.set_ylabel("平均分（误差线为 95% 聚类 bootstrap CI）")
    ax.set_xlabel("")
    ax.set_title(title, fontsize=16, pad=14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=len(metrics), frameon=False)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.savefig(output_dir / f"{filename}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{filename}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_metadata(
    path: Path,
    args: argparse.Namespace,
    model_names: list[str],
    data: pd.DataFrame,
    skipped: list[dict[str, Any]],
) -> None:
    metadata = {
        "生成时间": datetime.now().isoformat(timespec="seconds"),
        "模型名单来源": str(args.all_model_summary),
        "模型名单": model_names,
        "逐条评分来源": str(args.outputs_dir / "<模型名称>" / "dialog_eval_summary.csv"),
        "有效对话记录数": len(data),
        "唯一用户ID数": int(data["标准化ID"].nunique()),
        "标准差": "逐条有效对话分数的样本标准差（ddof=1）",
        "置信区间": f"按用户ID聚类的 percentile bootstrap，{args.confidence_level:.1%} CI",
        "bootstrap次数": args.bootstrap_samples,
        "随机种子": args.seed,
        "主观平均分": "仅在单条对话四项主观指标均完整时取算术平均",
        "评估端平均分": "仅在单条对话识别、理解、行动三项均完整时取算术平均",
        "跳过记录数": len(skipped),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_names = load_model_names(args.all_model_summary)
    data, skipped = load_dialog_metrics(model_names, args.outputs_dir)
    statistics = calculate_statistics(
        data,
        model_names,
        args.bootstrap_samples,
        args.confidence_level,
        args.seed,
    )

    data.to_csv(args.output_dir / "dialog_metric_scores.csv", index=False, encoding="utf-8-sig")
    statistics.to_csv(args.output_dir / "model_metric_statistics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(skipped, columns=["模型名称", "行号", "原因"]).to_csv(
        args.output_dir / "skipped_rows.csv", index=False, encoding="utf-8-sig"
    )
    plot_metric_intervals(
        statistics,
        model_names,
        SUBJECTIVE_METRICS,
        3,
        "各模型用户端指标表现",
        "subjective_metrics_with_ci",
        args.output_dir,
    )
    plot_metric_intervals(
        statistics,
        model_names,
        EVALUATOR_METRICS,
        2,
        "各模型评估端指标表现",
        "evaluator_metrics_with_ci",
        args.output_dir,
    )
    write_metadata(args.output_dir / "analysis_metadata.json", args, model_names, data, skipped)

    print(f"模型数量: {len(model_names)}")
    print(f"有效对话记录: {len(data)} 条，唯一用户ID: {data['标准化ID'].nunique()} 个")
    print(f"跳过记录: {len(skipped)} 条")
    print(f"输出目录: {args.output_dir}")
    print("\n各模型两端平均分：")
    display = statistics[statistics["指标"].isin(["主观平均分", "评估端平均分"])][
        ["模型名称", "指标", "均值", "标准差", "置信区间下限", "置信区间上限", "有效样本数"]
    ]
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
