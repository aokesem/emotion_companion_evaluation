# -*- coding: utf-8 -*-
"""比较加权总体与500次有放回抽样的重点用户画像分布。"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POPULATION = ROOT / "整理后数据" / "用户画像_分析样本.dta"
DEFAULT_SAMPLE = ROOT / "用户画像数据" / "用户画像_抽样500.dta"
DEFAULT_OUTPUT = ROOT / "用户画像数据" / "用户画像分布对比.csv"
WEIGHT_COL = "INDV_weight_ad2"
AGE_COL = "xrage"
MISSING_CODES = {97, 98, 99, 997, 998, 999}


Classifier = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class VariableSpec:
    column: str
    meaning: str
    rule: str
    categories: tuple[str, ...]
    classifier: Classifier


def numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        raise KeyError(f"数据缺少列: {column}")
    values = pd.to_numeric(df[column], errors="coerce")
    return values.mask(values.isin(MISSING_CODES))


def classify_codes(
    column: str,
    mapping: dict[int, str],
    *,
    fallback: str | None = None,
) -> Classifier:
    def classify(df: pd.DataFrame) -> pd.Series:
        values = numeric(df, column)
        if fallback is not None:
            values = values.fillna(numeric(df, fallback))
        return values.map(mapping).astype("string")

    return classify


def classify_ranges(
    column: str,
    ranges: list[tuple[float, float, str]],
    *,
    apply_missing_codes: bool = True,
) -> Classifier:
    def classify(df: pd.DataFrame) -> pd.Series:
        values = numeric(df, column) if apply_missing_codes else pd.to_numeric(df[column], errors="coerce")
        result = pd.Series(pd.NA, index=df.index, dtype="string")
        for lower, upper, label in ranges:
            result.loc[values.between(lower, upper, inclusive="both")] = label
        return result

    return classify


SPECS = (
    VariableSpec(
        "ba001",
        "访员记录性别",
        "原始编码：1=男；2=女",
        ("男", "女"),
        classify_codes("ba001", {1: "男", 2: "女"}),
    ),
    VariableSpec(
        "xrage",
        "年龄",
        "分档：45-59岁；60-69岁；70岁及以上",
        ("45-59岁", "60-69岁", "70岁及以上"),
        classify_ranges(
            "xrage",
            [(45, 59, "45-59岁"), (60, 69, "60-69岁"), (70, np.inf, "70岁及以上")],
            apply_missing_codes=False,
        ),
    ),
    VariableSpec(
        "ba007",
        "现居住地地址类型",
        "1=家庭住房；2/3/4=养老院、医院或其他",
        ("家庭住房", "机构或其他"),
        classify_codes("ba007", {1: "家庭住房", 2: "机构或其他", 3: "机构或其他", 4: "机构或其他"}),
    ),
    VariableSpec(
        "ba008",
        "城乡类型",
        "1=城/镇中心；2/3/4=城乡结合部、镇乡、农村或特殊地区",
        ("城/镇中心", "城乡结合部等"),
        classify_codes("ba008", {1: "城/镇中心", 2: "城乡结合部等", 3: "城乡结合部等", 4: "城乡结合部等"}),
    ),
    VariableSpec(
        "ba010/zredu",
        "目前最高学历",
        "ba010缺失时使用zredu；1-2=低学历；3-7=中等学历；8-11=高学历",
        ("低学历", "中等学历", "高学历"),
        classify_codes(
            "ba010",
            {
                1: "低学历", 2: "低学历",
                3: "中等学历", 4: "中等学历", 5: "中等学历", 6: "中等学历", 7: "中等学历",
                8: "高学历", 9: "高学历", 10: "高学历", 11: "高学历",
            },
            fallback="zredu",
        ),
    ),
    VariableSpec(
        "ba011",
        "婚姻状况",
        "按原始编码1-6分类",
        ("已婚且配偶同住", "已婚暂分居", "分居", "离婚", "丧偶", "未婚"),
        classify_codes(
            "ba011",
            {1: "已婚且配偶同住", 2: "已婚暂分居", 3: "分居", 4: "离婚", 5: "丧偶", 6: "未婚"},
        ),
    ),
    VariableSpec(
        "dc026",
        "生活满意度",
        "原始编码1-5（分数越高满意度越低）",
        ("非常满意", "很满意", "有点满意", "不太满意", "完全不满意"),
        classify_codes("dc026", {1: "非常满意", 2: "很满意", 3: "有点满意", 4: "不太满意", 5: "完全不满意"}),
    ),
    VariableSpec(
        "dc024",
        "感到孤独（过去一周）",
        "原始编码1-4；997/999及缺失不计入有效比例",
        ("几乎没有", "少量", "有时/中等", "大多数时候"),
        classify_codes("dc024", {1: "几乎没有", 2: "少量", 3: "有时/中等", 4: "大多数时候"}),
    ),
    VariableSpec(
        "da001",
        "自评健康状况",
        "原始编码1-5；997及缺失不计入有效比例",
        ("很好", "好", "一般", "差", "很差"),
        classify_codes("da001", {1: "很好", 2: "好", 3: "一般", 4: "差", 5: "很差"}),
    ),
    VariableSpec(
        "da038_s9",
        "上月是否以上活动都没有",
        "多选题编码：0=有其他活动；9=以上活动都没有",
        ("有其他活动", "以上都没有"),
        classify_codes("da038_s9", {0: "有其他活动", 9: "以上都没有"}),
    ),
    VariableSpec(
        "da040",
        "是否使用互联网",
        "原始编码：1=是；2=否",
        ("使用互联网", "不使用互联网"),
        classify_codes("da040", {1: "使用互联网", 2: "不使用互联网"}),
    ),
    VariableSpec(
        "xchildnum",
        "子女总数",
        "按数值分为0个、1个、2个、3个及以上",
        ("0个", "1个", "2个", "3个及以上"),
        classify_ranges(
            "xchildnum",
            [(0, 0, "0个"), (1, 1, "1个"), (2, 2, "2个"), (3, np.inf, "3个及以上")],
        ),
    ),
    VariableSpec(
        "fh001",
        "是否已办理退休",
        "原始编码：1=是；2=否",
        ("已退休", "未退休"),
        classify_codes("fh001", {1: "已退休", 2: "未退休"}),
    ),
    VariableSpec(
        "ff003",
        "上月是否在找工作",
        "原始编码：1=是；2=否",
        ("在找工作", "未找工作"),
        classify_codes("ff003", {1: "在找工作", 2: "未找工作"}),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="比较原始加权总体与抽样500数据的用户画像分布")
    parser.add_argument("--population", type=Path, default=DEFAULT_POPULATION, help="分析样本DTA")
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE, help="抽样500 DTA")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="一变量一行的汇总CSV")
    parser.add_argument("--detail-output", type=Path, default=None, help="逐类别明细CSV")
    parser.add_argument("--min-age", type=float, default=45.0, help="总体年龄下限")
    return parser.parse_args()


def validate_inputs(population: pd.DataFrame, sample: pd.DataFrame) -> None:
    required = {WEIGHT_COL, AGE_COL, "zredu"}
    for spec in SPECS:
        required.update(spec.column.split("/"))
    for name, frame in (("总体数据", population), ("抽样数据", sample)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise KeyError(f"{name}缺少列: {', '.join(missing)}")


def format_distribution(categories: tuple[str, ...], percentages: dict[str, float]) -> str:
    return "，".join(f"{category}{percentages[category]:.1f}%" for category in categories)


def build_tables(
    population: pd.DataFrame,
    sample: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weights = pd.to_numeric(population[WEIGHT_COL], errors="coerce")
    if weights.isna().any() or (weights <= 0).any():
        raise ValueError("年龄筛选后的总体数据包含无效权重")

    compact_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []

    for spec in SPECS:
        population_labels = spec.classifier(population)
        sample_labels = spec.classifier(sample)
        population_valid = population_labels.notna()
        sample_valid = sample_labels.notna()

        population_weight_total = float(weights.loc[population_valid].sum())
        population_percentages: dict[str, float] = {}
        sample_percentages: dict[str, float] = {}

        for category in spec.categories:
            population_weight = float(weights.loc[population_labels == category].sum())
            population_pct = 100.0 * population_weight / population_weight_total if population_weight_total else np.nan
            sample_count = int((sample_labels == category).sum())
            sample_pct = 100.0 * sample_count / int(sample_valid.sum()) if sample_valid.any() else np.nan
            population_percentages[category] = population_pct
            sample_percentages[category] = sample_pct

            detail_rows.append(
                {
                    "列名": spec.column,
                    "含义": spec.meaning,
                    "类别": category,
                    "原始数据有效N": int(population_valid.sum()),
                    "原始数据加权比例(%)": round(population_pct, 2),
                    "抽样500有效N": int(sample_valid.sum()),
                    "抽样500类别数量": sample_count,
                    "抽样500比例(%)": round(sample_pct, 2),
                    "比例差(百分点)": round(sample_pct - population_pct, 2),
                }
            )

        differences = [abs(sample_percentages[c] - population_percentages[c]) for c in spec.categories]
        compact_rows.append(
            {
                "列名": spec.column,
                "含义": spec.meaning,
                "分类规则": spec.rule,
                "原始数据加权统计": format_distribution(spec.categories, population_percentages),
                "抽样500统计": format_distribution(spec.categories, sample_percentages),
                "原始数据有效N": int(population_valid.sum()),
                "原始数据缺失率(%)": round(100.0 * (~population_valid).mean(), 2),
                "抽样500有效N": int(sample_valid.sum()),
                "抽样500缺失率(%)": round(100.0 * (~sample_valid).mean(), 2),
                "最大绝对差异(百分点)": round(max(differences), 2),
            }
        )

    return pd.DataFrame(compact_rows), pd.DataFrame(detail_rows)


def main() -> None:
    args = parse_args()
    population = pd.read_stata(args.population, convert_categoricals=False)
    sample = pd.read_stata(args.sample, convert_categoricals=False)
    validate_inputs(population, sample)

    ages = pd.to_numeric(population[AGE_COL], errors="coerce")
    population = population.loc[ages >= args.min_age].copy()
    if population.empty:
        raise ValueError(f"总体中没有年龄大于等于 {args.min_age:g} 岁的记录")

    compact, detail = build_tables(population, sample)
    detail_output = args.detail_output or args.output.with_name(f"{args.output.stem}_明细{args.output.suffix}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    detail_output.parent.mkdir(parents=True, exist_ok=True)
    compact.to_csv(args.output, index=False, encoding="utf-8-sig")
    detail.to_csv(detail_output, index=False, encoding="utf-8-sig")

    print(f"原始合格样本: {len(population)} 条（{AGE_COL} >= {args.min_age:g}，按 {WEIGHT_COL} 加权）")
    print(f"抽样数据: {len(sample)} 条（按抽样位置直接统计，重复ID保留）")
    print(f"汇总表: {args.output}")
    print(f"明细表: {detail_output}")


if __name__ == "__main__":
    main()
