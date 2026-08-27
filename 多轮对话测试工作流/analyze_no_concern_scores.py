# -*- coding: utf-8 -*-
"""Analyze seven metrics for dialogs whose concern category is no obvious concern."""
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

from analyze_scores_by_concern import (
    ALL_MODELS_ROW,
    DEFAULT_MODEL_SUMMARY,
    DEFAULT_OUTPUTS_DIR,
    DEFAULT_SIM_USER_JSONL,
    load_concern_lookup,
    load_dialog_scores,
    load_model_names,
)


TARGET_CONCERN = "无明显生活烦恼"
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUTS_DIR / "no_concern_analysis"

SUBJECTIVE_METRICS = ["被理解感", "情绪缓解感", "个性化贴合度", "交流舒适度"]
OBJECTIVE_METRICS = ["识别层评分", "理解层评分", "行动层评分"]
ALL_METRICS = SUBJECTIVE_METRICS + OBJECTIVE_METRICS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计无明显生活烦恼样本的七项指标和识别层分布")
    parser.add_argument(
        "--all-model-summary",
        type=Path,
        default=DEFAULT_MODEL_SUMMARY,
        help="模型评分总表；未传 --models 时从其中读取模型名单",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="逗号分隔的模型目录名；提供后覆盖模型评分总表中的名单",
    )
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR, help="各模型结果目录的根目录")
    parser.add_argument("--sim-user-jsonl", type=Path, default=DEFAULT_SIM_USER_JSONL, help="模拟用户信息 JSONL")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="分析结果输出目录")
    args = parser.parse_args()
    for name in ["all_model_summary", "outputs_dir", "sim_user_jsonl", "output_dir"]:
        setattr(args, name, getattr(args, name).resolve())
    return args


def parse_model_names(args: argparse.Namespace) -> list[str]:
    if not args.models.strip():
        return load_model_names(args.all_model_summary)
    names = [name.strip() for name in args.models.split(",") if name.strip()]
    names = list(dict.fromkeys(names))
    if not names:
        raise ValueError("--models 未包含有效模型名称")
    return names


def summarize_metrics(data: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name in model_names:
        model_data = data[data["模型名称"] == model_name]
        row: dict[str, Any] = {
            "模型名称": model_name,
            "对话数量": len(model_data),
            "唯一ID数": model_data["标准化ID"].nunique(),
        }
        for metric in ALL_METRICS:
            values = pd.to_numeric(model_data[metric], errors="coerce")
            row[f"{metric}_平均分"] = values.mean()
            row[f"{metric}_标准差"] = values.std()
            row[f"{metric}_有效N"] = values.count()
        rows.append(row)

    summary = pd.DataFrame(rows)
    overall: dict[str, Any] = {
        "模型名称": ALL_MODELS_ROW,
        "对话数量": len(data),
        "唯一ID数": data["标准化ID"].nunique(),
    }
    for metric in ALL_METRICS:
        mean_column = f"{metric}_平均分"
        valid_means = pd.to_numeric(summary[mean_column], errors="coerce").dropna()
        overall[mean_column] = valid_means.mean()
        overall[f"{metric}_标准差"] = valid_means.std()
        overall[f"{metric}_有效N"] = pd.to_numeric(data[metric], errors="coerce").count()
    return pd.concat([summary, pd.DataFrame([overall])], ignore_index=True).round(6)


def summarize_recognition_distribution(data: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name in model_names:
        values = pd.to_numeric(
            data.loc[data["模型名称"] == model_name, "识别层评分"], errors="coerce"
        ).dropna()
        counts = values.value_counts()
        total = len(values)
        row: dict[str, Any] = {
            "模型名称": model_name,
            "识别层平均分": values.mean(),
            "有效N": total,
        }
        for score in [0, 1, 2]:
            count = int(counts.get(float(score), 0))
            row[f"{score}分数量"] = count
            row[f"{score}分比例"] = count / total if total else np.nan
        rows.append(row)

    summary = pd.DataFrame(rows)
    overall: dict[str, Any] = {
        "模型名称": ALL_MODELS_ROW,
        "识别层平均分": summary["识别层平均分"].mean(),
        "有效N": int(summary["有效N"].sum()),
    }
    for score in [0, 1, 2]:
        overall[f"{score}分数量"] = int(summary[f"{score}分数量"].sum())
        overall[f"{score}分比例"] = summary[f"{score}分比例"].mean()
    return pd.concat([summary, pd.DataFrame([overall])], ignore_index=True).round(6)


def setup_plot_style() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="white", font="Microsoft YaHei")


def heatmap_annotations(values: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    annotations = pd.DataFrame("", index=values.index, columns=values.columns)
    for row in values.index:
        for column in values.columns:
            value = values.loc[row, column]
            count = counts.loc[row, column]
            annotations.loc[row, column] = "—" if pd.isna(value) else f"{value:.2f}\nn={int(count)}"
    return annotations


def plot_seven_metrics(summary: pd.DataFrame, output_dir: Path) -> None:
    setup_plot_style()
    labels = summary["模型名称"].tolist()
    height = max(6.8, 0.64 * len(labels) + 2.0)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(18, height),
        gridspec_kw={"width_ratios": [4, 3]},
        constrained_layout=True,
    )
    panels = [
        (axes[0], SUBJECTIVE_METRICS, 3, "YlGnBu", "(a) 主观评分（0–3）"),
        (axes[1], OBJECTIVE_METRICS, 2, "YlOrRd", "(b) 评估端评分（0–2）"),
    ]
    for ax, metrics, upper, cmap, title in panels:
        values = summary.set_index("模型名称")[[f"{metric}_平均分" for metric in metrics]].copy()
        counts = summary.set_index("模型名称")[[f"{metric}_有效N" for metric in metrics]].copy()
        values.columns = metrics
        counts.columns = metrics
        annotations = heatmap_annotations(values, counts)
        sns.heatmap(
            values,
            mask=values.isna(),
            annot=annotations,
            fmt="",
            cmap=cmap,
            vmin=0,
            vmax=upper,
            linewidths=0.8,
            linecolor="white",
            cbar_kws={"shrink": 0.78},
            ax=ax,
        )
        ax.set_title(title, fontsize=14, pad=12)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)
        ax.axhline(len(values) - 1, color="#333333", linewidth=1.8)
        ax.set_facecolor("#eeeeee")
    fig.suptitle("无明显生活烦恼样本的七项指标表现", fontsize=16)
    fig.savefig(output_dir / "no_concern_seven_metric_heatmap.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / "no_concern_seven_metric_heatmap.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_recognition(distribution: pd.DataFrame, output_dir: Path) -> None:
    setup_plot_style()
    labels = distribution["模型名称"].tolist()
    y = np.arange(len(labels))
    height = max(6.8, 0.64 * len(labels) + 2.0)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(17, height),
        gridspec_kw={"width_ratios": [1.0, 1.45]},
        constrained_layout=True,
    )

    means = distribution["识别层平均分"].to_numpy(dtype=float)
    axes[0].barh(y, means, color="#3B82A0", height=0.62)
    axes[0].set_xlim(0, 2)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("识别层平均分")
    axes[0].set_title("(a) 平均得分", fontsize=14, pad=12)
    axes[0].grid(axis="x", color="#dddddd", linewidth=0.8)
    axes[0].set_axisbelow(True)
    for index, (mean_value, count) in enumerate(zip(means, distribution["有效N"])):
        if np.isfinite(mean_value):
            axes[0].text(min(mean_value + 0.03, 1.93), index, f"{mean_value:.2f}  n={int(count)}", va="center")

    colors = ["#C94F45", "#E7B85C", "#4D9078"]
    left = np.zeros(len(distribution))
    for score, color in zip([0, 1, 2], colors):
        proportions = distribution[f"{score}分比例"].fillna(0).to_numpy(dtype=float) * 100
        axes[1].barh(y, proportions, left=left, color=color, height=0.62, label=f"{score} 分")
        for index, (start, width) in enumerate(zip(left, proportions)):
            if width >= 7:
                axes[1].text(start + width / 2, index, f"{width:.0f}%", ha="center", va="center", fontsize=9)
        left += proportions
    axes[1].set_xlim(0, 100)
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("评分占比（%）")
    axes[1].set_title("(b) 0/1/2 分分布", fontsize=14, pad=12)
    axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    axes[1].grid(axis="x", color="#eeeeee", linewidth=0.8)
    axes[1].set_axisbelow(True)
    for ax in axes:
        ax.axhline(len(labels) - 1.5, color="#333333", linewidth=1.5)

    fig.suptitle("无明显生活烦恼样本的识别层表现", fontsize=16)
    fig.savefig(output_dir / "no_concern_recognition_scores.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / "no_concern_recognition_scores.pdf", bbox_inches="tight", facecolor="white")
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
        "目标烦恼类别": TARGET_CONCERN,
        "模型名单": model_names,
        "模型名单来源": "--models" if args.models.strip() else str(args.all_model_summary),
        "评分明细来源": str(args.outputs_dir / "<模型名称>" / "dialog_eval_summary.csv"),
        "烦恼类别来源": str(args.sim_user_jsonl),
        "匹配对话数": len(data),
        "唯一用户ID数": int(data["标准化ID"].nunique()),
        "所有模型行": "各模型等权平均；识别层分布为各模型比例的等权平均，不是合并全部对话后的比例",
        "跳过记录数": len(skipped),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_names = parse_model_names(args)
    concern_lookup = load_concern_lookup(args.sim_user_jsonl)
    all_data, skipped = load_dialog_scores(model_names, args.outputs_dir, concern_lookup)
    data = all_data[all_data["烦恼类别"] == TARGET_CONCERN].copy()
    if data.empty:
        raise ValueError(f"没有匹配到烦恼类别为“{TARGET_CONCERN}”的评分记录")

    metric_summary = summarize_metrics(data, model_names)
    recognition_distribution = summarize_recognition_distribution(data, model_names)

    data.to_csv(args.output_dir / "no_concern_dialog_scores.csv", index=False, encoding="utf-8-sig")
    metric_summary.to_csv(args.output_dir / "no_concern_metric_summary.csv", index=False, encoding="utf-8-sig")
    recognition_distribution.to_csv(
        args.output_dir / "recognition_score_distribution.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(skipped, columns=["模型名称", "行号", "原因"]).to_csv(
        args.output_dir / "skipped_rows.csv", index=False, encoding="utf-8-sig"
    )
    plot_seven_metrics(metric_summary, args.output_dir)
    plot_recognition(recognition_distribution, args.output_dir)
    write_metadata(args.output_dir / "analysis_metadata.json", args, model_names, data, skipped)

    print(f"目标类别: {TARGET_CONCERN}")
    print(f"模型数量: {len(model_names)}")
    print(f"匹配对话: {len(data)} 条，唯一用户ID: {data['标准化ID'].nunique()} 个")
    print(f"跳过记录: {len(skipped)} 条")
    print(f"输出目录: {args.output_dir}")
    print("\n识别层统计:")
    print(
        recognition_distribution[
            ["模型名称", "识别层平均分", "0分比例", "1分比例", "2分比例", "有效N"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
