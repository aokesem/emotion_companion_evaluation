# -*- coding: utf-8 -*-
"""Extract compact score table from dialog_eval_summary.csv."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "outputs"

SUMMARY_FIELDS = [
    "run_id",
    "ID",
    "sample_pick_order",
    "turns",
    "simulated_user_rating_json",
    "tested_agent_summary_json",
    "evaluator_rating_json",
]

OUTPUT_FIELDS = [
    "ID",
    "sample_pick_order",
    "被理解感",
    "情绪缓解感",
    "个性化贴合度",
    "交流舒适度",
    "用户AI主观平均分",
    "识别层评分",
    "理解层评分",
    "行动层评分",
    "评估AI平均分",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 dialog_eval_summary.csv 提取精简评分 CSV")
    parser.add_argument("--output-dir", type=str, default="", help="输出实验目录；相对路径默认位于 outputs/ 下")
    parser.add_argument("--input", type=Path, default=None, help="输入 summary CSV；默认由 output-dir 派生")
    parser.add_argument("--output", type=Path, default=None, help="输出精简评分 CSV；默认由 output-dir 派生")
    parser.add_argument("--score-summary-output", type=Path, default=None, help="输出总体评分 JSON；默认由 output-dir 派生")
    args = parser.parse_args()

    output_dir = resolve_output_dir(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    args.input = resolve_path(args.input) if args.input is not None else output_dir / "dialog_eval_summary.csv"
    args.output = resolve_path(args.output) if args.output is not None else output_dir / "dialog_eval_scores.csv"
    args.score_summary_output = (
        resolve_path(args.score_summary_output)
        if args.score_summary_output is not None
        else output_dir / "dialog_eval_score_summary.json"
    )
    return args

def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def resolve_output_dir(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "outputs":
        return (ROOT / path).resolve()
    return (ROOT / "outputs" / path).resolve()


def parse_json(value: str, row_no: int, field: str) -> dict[str, Any]:
    try:
        obj = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"第 {row_no} 行字段 {field} 不是合法 JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"第 {row_no} 行字段 {field} 的 JSON 不是对象")
    return obj


def number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def first_number(obj: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = number(obj.get(key))
        if value is not None:
            return value
    return None


def average(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return round(mean(valid), 2)


def format_score(value: float | None) -> str:
    if value is None:
        return ""
    if value.is_integer():
        return str(int(value))
    return f"{value:.2f}"


def read_summary_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return []

    first = rows[0]
    has_header = "ID" in first and "simulated_user_rating_json" in first
    if has_header:
        return [dict(zip(first, row)) for row in rows[1:] if any(cell.strip() for cell in row)]

    return [dict(zip(SUMMARY_FIELDS, row)) for row in rows if any(cell.strip() for cell in row)]


def extract_score_row(row: dict[str, str], row_no: int) -> tuple[dict[str, str], dict[str, float | None]]:
    user_rating = parse_json(row.get("simulated_user_rating_json", ""), row_no, "simulated_user_rating_json")
    evaluator_rating = parse_json(row.get("evaluator_rating_json", ""), row_no, "evaluator_rating_json")

    understood = first_number(user_rating, ["被理解感"])
    relief = first_number(user_rating, ["情绪缓解感", "情绪慰藉感"])
    personalization = first_number(user_rating, ["个性化贴合度"])
    comfort = first_number(user_rating, ["交流舒适度", "交流舒适度_被尊重感"])
    user_avg = first_number(user_rating, ["主观总分"])
    if user_avg is None:
        user_avg = average([understood, relief, personalization, comfort])

    recognition = first_number(evaluator_rating, ["识别层评分"])
    understanding = first_number(evaluator_rating, ["理解层评分"])
    action = first_number(evaluator_rating, ["行动层评分"])
    evaluator_avg = first_number(evaluator_rating, ["情绪识别总分"])
    if evaluator_avg is None:
        evaluator_avg = average([recognition, understanding, action])

    scores = {
        "user_avg": user_avg,
        "evaluator_avg": evaluator_avg,
        "understood": understood,
        "relief": relief,
        "personalization": personalization,
        "comfort": comfort,
        "recognition": recognition,
        "understanding": understanding,
        "action": action,
    }
    output = {
        "ID": row.get("ID", ""),
        "sample_pick_order": row.get("sample_pick_order", ""),
        "被理解感": format_score(understood),
        "情绪缓解感": format_score(relief),
        "个性化贴合度": format_score(personalization),
        "交流舒适度": format_score(comfort),
        "用户AI主观平均分": format_score(user_avg),
        "识别层评分": format_score(recognition),
        "理解层评分": format_score(understanding),
        "行动层评分": format_score(action),
        "评估AI平均分": format_score(evaluator_avg),
    }
    return output, scores


def score_summary(score_rows: list[dict[str, float | None]]) -> dict[str, float | int | None]:
    summary: dict[str, float | int | None] = {"样本数": len(score_rows)}
    metrics = [
        ("用户AI主观平均分均值", "user_avg"),
        ("评估AI平均分均值", "evaluator_avg"),
        ("被理解感均值", "understood"),
        ("情绪缓解感均值", "relief"),
        ("个性化贴合度均值", "personalization"),
        ("交流舒适度均值", "comfort"),
        ("识别层评分均值", "recognition"),
        ("理解层评分均值", "understanding"),
        ("行动层评分均值", "action"),
    ]
    for label, key in metrics:
        summary[label] = average([row[key] for row in score_rows])
    return summary


def print_summary(summary: dict[str, float | int | None]) -> None:
    for label, value in summary.items():
        display = format_score(value) if isinstance(value, float) else value
        print(f"{label}: {display}")

def main() -> None:
    args = parse_args()
    rows = read_summary_rows(args.input)
    output_rows: list[dict[str, str]] = []
    score_rows: list[dict[str, float | None]] = []

    for index, row in enumerate(rows, start=1):
        output, scores = extract_score_row(row, index)
        output_rows.append(output)
        score_rows.append(scores)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    summary = score_summary(score_rows)
    args.score_summary_output.parent.mkdir(parents=True, exist_ok=True)
    with args.score_summary_output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"输入: {args.input}")
    print(f"输出: {args.output}")
    print(f"总体评分: {args.score_summary_output}")
    print_summary(summary)

if __name__ == "__main__":
    main()
