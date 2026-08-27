# -*- coding: utf-8 -*-
"""Summarize subjective/objective scores by model and concern category."""
from __future__ import annotations

import argparse
import json
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from extract_eval_scores import extract_score_row, read_summary_rows


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUTS_DIR = ROOT / "outputs"
DEFAULT_MODEL_SUMMARY = DEFAULT_OUTPUTS_DIR / "all_model_score_summary copy.csv"
DEFAULT_SIM_USER_JSONL = ROOT.parent / "用户画像数据" / "模拟用户信息_500.jsonl"
DEFAULT_ANALYSIS_DIR = DEFAULT_OUTPUTS_DIR / "concern_type_analysis"

CONCERN_ORDER = [
    "无明显生活烦恼",
    "孤独/人际交往缺失",
    "对身体健康的担忧",
    "家庭关系或陪伴困扰",
    "无法适应新事物或自我价值感下降",
]

CONCERN_SHORT_LABELS = {
    "无明显生活烦恼": "无明显生活\n烦恼",
    "孤独/人际交往缺失": "孤独/人际\n交往缺失",
    "对身体健康的担忧": "身体健康\n担忧",
    "家庭关系或陪伴困扰": "家庭关系/\n陪伴困扰",
    "无法适应新事物或自我价值感下降": "适应新事物/\n价值感下降",
}

CONCERN_ALIASES = {
    "感到孤独/缺失人际交往": "孤独/人际交往缺失",
    "孤独/缺失人际交往": "孤独/人际交往缺失",
}

ALL_MODELS_ROW = "所有模型（模型等权）"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按模型和烦恼类别统计主观、客观平均分并生成热力图")
    parser.add_argument("--all-model-summary", type=Path, default=DEFAULT_MODEL_SUMMARY, help="全部模型评分总表，用作模型名单")
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR, help="各模型结果目录的根目录")
    parser.add_argument("--sim-user-jsonl", type=Path, default=DEFAULT_SIM_USER_JSONL, help="模拟用户信息JSONL")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ANALYSIS_DIR, help="统计结果与图表输出目录")
    parser.add_argument("--min-cell-n", type=int, default=5, help="模型等权汇总纳入单元格的最低有效样本数")
    args = parser.parse_args()
    if args.min_cell_n <= 0:
        raise ValueError("--min-cell-n 必须为正整数")
    for name in ["all_model_summary", "outputs_dir", "sim_user_jsonl", "output_dir"]:
        setattr(args, name, getattr(args, name).resolve())
    return args


def normalized_id(value: Any) -> str:
    text = str(value).strip().lstrip("0")
    return text or "0"


def normalize_concern(value: Any) -> str:
    text = str(value).strip()
    return CONCERN_ALIASES.get(text, text)


def mean_if_complete(values: list[float | None]) -> float:
    return float(np.mean(values)) if values and all(value is not None for value in values) else np.nan


def load_model_names(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"全部模型评分总表不存在: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if "模型名称" not in frame.columns:
        raise KeyError(f"全部模型评分总表缺少列: 模型名称 ({path})")
    names = [str(value).strip() for value in frame["模型名称"] if str(value).strip()]
    if not names:
        raise ValueError(f"全部模型评分总表中没有模型: {path}")
    return list(dict.fromkeys(names))


def load_concern_lookup(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"模拟用户信息不存在: {path}")
    lookup: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            row_id = normalized_id(obj.get("ID", ""))
            concern_obj = obj.get("生活烦恼", {})
            concern = normalize_concern(concern_obj.get("烦恼类别", "") if isinstance(concern_obj, dict) else "")
            if not row_id or row_id == "0" or not concern:
                raise ValueError(f"模拟用户信息第 {line_no} 行缺少有效ID或烦恼类别")
            if concern not in CONCERN_ORDER:
                raise ValueError(f"模拟用户信息第 {line_no} 行包含未知烦恼类别: {concern}")
            previous = lookup.get(row_id)
            if previous is not None and previous != concern:
                raise ValueError(f"同一ID存在不同烦恼类别: ID={row_id}, {previous} / {concern}")
            lookup[row_id] = concern
    return lookup


def load_dialog_scores(
    model_names: list[str],
    outputs_dir: Path,
    concern_lookup: dict[str, str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for model_name in model_names:
        summary_path = outputs_dir / model_name / "dialog_eval_summary.csv"
        if not summary_path.is_file():
            skipped.append({"模型名称": model_name, "行号": "", "原因": "缺少 dialog_eval_summary.csv"})
            continue
        try:
            rows = read_summary_rows(summary_path)
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
            concern = concern_lookup.get(row_id)
            if concern is None:
                skipped.append({"模型名称": model_name, "行号": row_no, "原因": f"ID={row_id} 未匹配到烦恼类别"})
                continue

            subjective = mean_if_complete(
                [scores["understood"], scores["relief"], scores["personalization"], scores["comfort"]]
            )
            objective = mean_if_complete(
                [scores["recognition"], scores["understanding"], scores["action"]]
            )
            records.append(
                {
                    "模型名称": model_name,
                    "ID": output.get("ID", ""),
                    "标准化ID": row_id,
                    "sample_pick_order": pd.to_numeric(output.get("sample_pick_order", ""), errors="coerce"),
                    "烦恼类别": concern,
                    "被理解感": scores["understood"],
                    "情绪缓解感": scores["relief"],
                    "个性化贴合度": scores["personalization"],
                    "交流舒适度": scores["comfort"],
                    "识别层评分": scores["recognition"],
                    "理解层评分": scores["understanding"],
                    "行动层评分": scores["action"],
                    "主观平均分": subjective,
                    "客观平均分": objective,
                }
            )

    if not records:
        raise ValueError("没有可用于烦恼类别分析的成功评分记录")
    data = pd.DataFrame(records)
    data["sample_pick_order"] = pd.to_numeric(data["sample_pick_order"], errors="coerce").astype("Int64")
    return data, skipped


def summarize_model_concerns(data: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    observed = (
        data.groupby(["模型名称", "烦恼类别"], observed=True)
        .agg(
            对话数量=("ID", "size"),
            唯一ID数=("标准化ID", "nunique"),
            主观有效N=("主观平均分", "count"),
            主观平均分=("主观平均分", "mean"),
            主观标准差=("主观平均分", "std"),
            客观有效N=("客观平均分", "count"),
            客观平均分=("客观平均分", "mean"),
            客观标准差=("客观平均分", "std"),
        )
        .reset_index()
    )

    complete_index = pd.MultiIndex.from_product([model_names, CONCERN_ORDER], names=["模型名称", "烦恼类别"])
    summary = observed.set_index(["模型名称", "烦恼类别"]).reindex(complete_index).reset_index()
    for column in ["对话数量", "唯一ID数", "主观有效N", "客观有效N"]:
        summary[column] = summary[column].fillna(0).astype(int)

    model_subjective = data.groupby("模型名称")["主观平均分"].mean()
    model_objective = data.groupby("模型名称")["客观平均分"].mean()
    summary["主观相对模型总体偏差"] = summary["主观平均分"] - summary["模型名称"].map(model_subjective)
    summary["客观相对模型总体偏差"] = summary["客观平均分"] - summary["模型名称"].map(model_objective)
    return summary.round(6)


def summarize_all_models(model_summary: pd.DataFrame, data: pd.DataFrame, min_cell_n: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for concern in CONCERN_ORDER:
        concern_cells = model_summary[model_summary["烦恼类别"] == concern]
        conversation_rows = data[data["烦恼类别"] == concern]
        subjective_cells = concern_cells[
            concern_cells["主观平均分"].notna() & (concern_cells["主观有效N"] >= min_cell_n)
        ]
        objective_cells = concern_cells[
            concern_cells["客观平均分"].notna() & (concern_cells["客观有效N"] >= min_cell_n)
        ]
        rows.append(
            {
                "烦恼类别": concern,
                "总对话数量": len(conversation_rows),
                "唯一ID数": conversation_rows["标准化ID"].nunique(),
                "主观对话加权平均分": conversation_rows["主观平均分"].mean(),
                "客观对话加权平均分": conversation_rows["客观平均分"].mean(),
                "主观模型等权平均分": subjective_cells["主观平均分"].mean(),
                "主观纳入模型数": len(subjective_cells),
                "客观模型等权平均分": objective_cells["客观平均分"].mean(),
                "客观纳入模型数": len(objective_cells),
                "模型等权最低单元格N": min_cell_n,
            }
        )
    return pd.DataFrame(rows).round(6)


def build_heatmap_tables(
    model_summary: pd.DataFrame,
    all_models: pd.DataFrame,
    model_names: list[str],
    metric: str,
    count_column: str,
    all_models_metric: str,
    all_models_count: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = model_summary.pivot(index="模型名称", columns="烦恼类别", values=metric).reindex(
        index=model_names, columns=CONCERN_ORDER
    )
    counts = model_summary.pivot(index="模型名称", columns="烦恼类别", values=count_column).reindex(
        index=model_names, columns=CONCERN_ORDER
    )
    all_value_row = all_models.set_index("烦恼类别")[all_models_metric].reindex(CONCERN_ORDER)
    all_count_row = all_models.set_index("烦恼类别")[all_models_count].reindex(CONCERN_ORDER)
    values.loc[ALL_MODELS_ROW] = all_value_row
    counts.loc[ALL_MODELS_ROW] = all_count_row
    values.columns = [CONCERN_SHORT_LABELS[column] for column in values.columns]
    counts.columns = values.columns
    return values, counts


def annotation_matrix(values: pd.DataFrame, counts: pd.DataFrame, min_cell_n: int) -> pd.DataFrame:
    annotations = pd.DataFrame("", index=values.index, columns=values.columns)
    for row in values.index:
        for column in values.columns:
            value = values.loc[row, column]
            count = counts.loc[row, column]
            if pd.isna(value):
                annotations.loc[row, column] = "—"
            elif row == ALL_MODELS_ROW:
                annotations.loc[row, column] = f"{value:.2f}\nm={int(count)}"
            else:
                suffix = "*" if count < min_cell_n else ""
                annotations.loc[row, column] = f"{value:.2f}\nn={int(count)}{suffix}"
    return annotations


def plot_absolute_heatmaps(
    subjective_values: pd.DataFrame,
    subjective_counts: pd.DataFrame,
    objective_values: pd.DataFrame,
    objective_counts: pd.DataFrame,
    min_cell_n: int,
    output_dir: Path,
) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="white", font="Microsoft YaHei")
    height = max(9.5, 0.56 * len(subjective_values) + 2.5)
    fig, axes = plt.subplots(1, 2, figsize=(20, height), constrained_layout=True)
    panels = [
        (axes[0], subjective_values, subjective_counts, 0, 3, "YlGnBu", "(a) 主观平均分（0–3）"),
        (axes[1], objective_values, objective_counts, 0, 2, "YlOrRd", "(b) 客观平均分（0–2）"),
    ]
    for ax, values, counts, lower, upper, cmap, title in panels:
        annotations = annotation_matrix(values, counts, min_cell_n)
        sns.heatmap(
            values,
            mask=values.isna(),
            annot=annotations,
            fmt="",
            cmap=cmap,
            vmin=lower,
            vmax=upper,
            linewidths=0.8,
            linecolor="white",
            cbar_kws={"shrink": 0.76},
            ax=ax,
        )
        ax.set_title(title, fontsize=14, pad=12)
        ax.set_xlabel("烦恼类别")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)
        ax.axhline(len(values) - 1, color="#333333", linewidth=1.8)
        ax.set_facecolor("#eeeeee")
    fig.suptitle(
        f"不同模型在各烦恼类别下的阶段性得分（* 单元格有效样本量 < {min_cell_n}）",
        fontsize=16,
    )
    fig.savefig(output_dir / "concern_score_heatmaps.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / "concern_score_heatmaps.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_centered_heatmaps(model_summary: pd.DataFrame, model_names: list[str], min_cell_n: int, output_dir: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="white", font="Microsoft YaHei")
    specs = [
        ("主观相对模型总体偏差", "主观有效N", "(a) 主观得分相对模型总体均分", "主观得分偏差"),
        ("客观相对模型总体偏差", "客观有效N", "(b) 客观得分相对模型总体均分", "客观得分偏差"),
    ]
    matrices: list[tuple[pd.DataFrame, pd.DataFrame, str, str]] = []
    maximum = 0.0
    for metric, count_column, title, color_label in specs:
        values = model_summary.pivot(index="模型名称", columns="烦恼类别", values=metric).reindex(
            index=model_names, columns=CONCERN_ORDER
        )
        counts = model_summary.pivot(index="模型名称", columns="烦恼类别", values=count_column).reindex(
            index=model_names, columns=CONCERN_ORDER
        )
        values.columns = [CONCERN_SHORT_LABELS[column] for column in values.columns]
        counts.columns = values.columns
        finite = np.abs(values.to_numpy(dtype=float))
        if np.isfinite(finite).any():
            maximum = max(maximum, float(np.nanmax(finite)))
        matrices.append((values, counts, title, color_label))
    scale = max(0.25, np.ceil(maximum * 10) / 10)

    height = max(9, 0.56 * len(model_names) + 2.2)
    fig, axes = plt.subplots(1, 2, figsize=(20, height), constrained_layout=True)
    for ax, (values, counts, title, color_label) in zip(axes, matrices):
        annotations = annotation_matrix(values, counts, min_cell_n)
        sns.heatmap(
            values,
            mask=values.isna(),
            annot=annotations,
            fmt="",
            cmap="vlag",
            center=0,
            vmin=-scale,
            vmax=scale,
            linewidths=0.8,
            linecolor="white",
            cbar_kws={"label": color_label, "shrink": 0.76},
            ax=ax,
        )
        ax.set_title(title, fontsize=14, pad=12)
        ax.set_xlabel("烦恼类别")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)
        ax.set_facecolor("#eeeeee")
    fig.suptitle(
        f"各模型相对自身总体均分的烦恼类别偏差（* 单元格有效样本量 < {min_cell_n}）",
        fontsize=16,
    )
    fig.savefig(output_dir / "concern_score_centered_heatmaps.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / "concern_score_centered_heatmaps.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_metadata(
    path: Path,
    args: argparse.Namespace,
    model_names: list[str],
    concern_lookup: dict[str, str],
    data: pd.DataFrame,
    skipped: list[dict[str, Any]],
) -> None:
    newest_summary = max(
        (
            (args.outputs_dir / model / "dialog_eval_summary.csv").stat().st_mtime
            for model in model_names
            if (args.outputs_dir / model / "dialog_eval_summary.csv").is_file()
        ),
        default=0,
    )
    metadata = {
        "生成时间": datetime.now().isoformat(timespec="seconds"),
        "模型名单来源": str(args.all_model_summary),
        "评分明细来源": "模型目录中的 dialog_eval_summary.csv",
        "烦恼类别来源": str(args.sim_user_jsonl),
        "模型数": len(model_names),
        "成功匹配对话数": len(data),
        "唯一用户ID数": int(data["标准化ID"].nunique()),
        "模拟用户ID数": len(concern_lookup),
        "模型等权最低单元格N": args.min_cell_n,
        "主观平均分": "四项主观指标完整时取算术平均，范围0–3",
        "客观平均分": "识别、理解、行动三层评分完整时取算术平均，范围0–2",
        "所有模型汇总": "模型等权均值仅纳入该类别有效N达到min-cell-n的模型单元格",
        "跳过记录数": len(skipped),
        "总表可能过时": bool(newest_summary > args.all_model_summary.stat().st_mtime),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_names = load_model_names(args.all_model_summary)
    concern_lookup = load_concern_lookup(args.sim_user_jsonl)
    data, skipped = load_dialog_scores(model_names, args.outputs_dir, concern_lookup)
    model_summary = summarize_model_concerns(data, model_names)
    all_models = summarize_all_models(model_summary, data, args.min_cell_n)

    subjective_values, subjective_counts = build_heatmap_tables(
        model_summary,
        all_models,
        model_names,
        "主观平均分",
        "主观有效N",
        "主观模型等权平均分",
        "主观纳入模型数",
    )
    objective_values, objective_counts = build_heatmap_tables(
        model_summary,
        all_models,
        model_names,
        "客观平均分",
        "客观有效N",
        "客观模型等权平均分",
        "客观纳入模型数",
    )

    data.to_csv(args.output_dir / "dialog_scores_with_concern.csv", index=False, encoding="utf-8-sig")
    model_summary.to_csv(args.output_dir / "model_concern_score_summary.csv", index=False, encoding="utf-8-sig")
    all_models.to_csv(args.output_dir / "all_models_concern_score_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(skipped, columns=["模型名称", "行号", "原因"]).to_csv(
        args.output_dir / "skipped_rows.csv", index=False, encoding="utf-8-sig"
    )
    plot_absolute_heatmaps(
        subjective_values,
        subjective_counts,
        objective_values,
        objective_counts,
        args.min_cell_n,
        args.output_dir,
    )
    plot_centered_heatmaps(model_summary, model_names, args.min_cell_n, args.output_dir)
    write_metadata(args.output_dir / "analysis_metadata.json", args, model_names, concern_lookup, data, skipped)

    print(f"模型名单: {len(model_names)} 个")
    print(f"成功匹配对话: {len(data)} 条，唯一用户ID: {data['标准化ID'].nunique()} 个")
    print(f"跳过记录: {len(skipped)} 条")
    print(f"输出目录: {args.output_dir}")
    print("\n所有模型在各烦恼类别下的阶段性得分：")
    columns = [
        "烦恼类别",
        "总对话数量",
        "主观对话加权平均分",
        "主观模型等权平均分",
        "客观对话加权平均分",
        "客观模型等权平均分",
    ]
    display = all_models[columns].copy()
    display["烦恼类别"] = display["烦恼类别"].map(lambda value: "\n".join(textwrap.wrap(value, width=16)))
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
