# -*- coding: utf-8 -*-
"""从分析样本中抽取真实整行记录，并尽量贴近控制列的加权分布。"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "CHARLS2020r 用户画像数据"
DATA_PATH = DATA_DIR / "用户画像_分析样本.dta"
RULES_PATH = ROOT / "用户画像权重分层.csv"

WEIGHT_COL = "INDV_weight_ad2"
PAIR_RE = re.compile(r"(.+?)(\d+(?:\.\d+)?)%$")


@dataclass
class ControlSpec:
    name: str
    labels: list[str]
    targets: np.ndarray
    codes: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="抽取用户画像样本")
    parser.add_argument("--n", type=int, default=1000, help="抽样条数")
    parser.add_argument("--min-valid", type=int, default=300, help="每个控制列的最低有效样本数")
    parser.add_argument("--seed", type=int, default=20260523, help="随机种子")
    parser.add_argument(
        "--coverage-weight",
        type=float,
        default=5.0,
        help="控制列有效样本缺口的优先级权重",
    )
    parser.add_argument(
        "--dist-weight",
        type=float,
        default=1.0,
        help="控制列分布贴合的优先级权重",
    )
    return parser.parse_args()


def parse_target_stats(text: str) -> list[tuple[str, float]]:
    pairs: list[tuple[str, float]] = []
    for part in str(text).split("，"):
        token = part.strip()
        if not token:
            continue
        match = PAIR_RE.match(token)
        if not match:
            raise ValueError(f"无法解析加权统计: {text}")
        label = match.group(1)
        pct = float(match.group(2)) / 100.0
        pairs.append((label, pct))
    if not pairs:
        raise ValueError(f"加权统计为空: {text}")
    return pairs


def map_control_labels(df: pd.DataFrame, col: str) -> pd.Series:
    labels = pd.Series(pd.NA, index=df.index, dtype="object")

    if col == "ba001":
        return df["ba001"].map({1: "男", 2: "女"}).astype("object")

    if col == "xrage":
        labels.loc[df["xrage"].between(45, 59, inclusive="both")] = "45–59岁"
        labels.loc[df["xrage"].between(60, 69, inclusive="both")] = "60–69岁"
        labels.loc[df["xrage"] >= 70] = "70岁及以上"
        return labels

    if col == "ba007":
        labels.loc[df["ba007"] == 1] = "家庭住房"
        labels.loc[df["ba007"].isin([2, 3, 4])] = "机构或其他"
        return labels

    if col == "ba008":
        labels.loc[df["ba008"] == 1] = "城/镇中心"
        labels.loc[df["ba008"].isin([2, 3, 4])] = "城乡结合部等"
        return labels

    if col == "ba010":
        edu = df["ba010"].fillna(df["zredu"])
        labels.loc[edu.between(1, 2)] = "低学历"
        labels.loc[edu.between(3, 7)] = "中学历"
        labels.loc[edu.between(8, 11)] = "高学历"
        return labels

    if col == "ba011":
        labels.loc[df["ba011"] == 1] = "已婚且配偶同住"
        labels.loc[df["ba011"].isin([2, 3])] = "分居"
        labels.loc[df["ba011"].isin([4, 5])] = "离婚或丧偶"
        labels.loc[df["ba011"] == 6] = "未婚"
        return labels

    if col == "dc026":
        labels.loc[df["dc026"].isin([1, 2])] = "满意"
        labels.loc[df["dc026"] == 3] = "有点满意"
        labels.loc[df["dc026"].isin([4, 5])] = "不满意"
        return labels

    if col == "dc024":
        labels.loc[df["dc024"] == 1] = "几乎不孤独"
        labels.loc[df["dc024"].isin([2, 3])] = "少量或有时孤独"
        labels.loc[df["dc024"] == 4] = "经常孤独"
        return labels

    if col == "da001":
        labels.loc[df["da001"].isin([1, 2])] = "好或很好"
        labels.loc[df["da001"] == 3] = "一般"
        labels.loc[df["da001"].isin([4, 5])] = "差或很差"
        return labels

    if col == "da038_s9":
        labels.loc[df["da038_s9"] == 0] = "有其他活动"
        labels.loc[df["da038_s9"] == 9] = "以上都没有"
        return labels

    if col == "da040":
        return df["da040"].map({1: "使用互联网", 2: "不使用"}).astype("object")

    if col == "xchildnum":
        labels.loc[df["xchildnum"] == 0] = "0个"
        labels.loc[df["xchildnum"] == 1] = "1个"
        labels.loc[df["xchildnum"] == 2] = "2个"
        labels.loc[df["xchildnum"] >= 3] = "3个及以上"
        return labels

    if col == "fh001":
        return df["fh001"].map({1: "已退休", 2: "未退休"}).astype("object")

    raise KeyError(f"未实现控制列映射: {col}")


def build_control_specs(df: pd.DataFrame, rules: pd.DataFrame) -> list[ControlSpec]:
    specs: list[ControlSpec] = []
    for _, row in rules.iterrows():
        pairs = parse_target_stats(row["加权统计"])
        labels = [label for label, _ in pairs]
        targets = np.array([pct for _, pct in pairs], dtype=float)
        label_series = map_control_labels(df, row["列名"])
        codes = pd.Categorical(label_series, categories=labels).codes.astype(np.int16)
        specs.append(ControlSpec(name=row["列名"], labels=labels, targets=targets, codes=codes))
    return specs


def distribution_improvements(counts: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if counts.sum() == 0:
        base_loss = 2.0
    else:
        base_loss = np.abs(counts / counts.sum() - targets).sum()

    improvements = np.zeros(len(targets), dtype=float)
    for idx in range(len(targets)):
        next_counts = counts.copy()
        next_counts[idx] += 1
        next_loss = np.abs(next_counts / next_counts.sum() - targets).sum()
        improvements[idx] = base_loss - next_loss
    return improvements


def pick_one(
    remaining: np.ndarray,
    specs: list[ControlSpec],
    counts_by_col: list[np.ndarray],
    valid_counts: np.ndarray,
    weights: np.ndarray,
    min_valid: int,
    coverage_weight: float,
    dist_weight: float,
    rng: np.random.Generator,
) -> int:
    scores = np.zeros(len(remaining), dtype=float)

    for idx, spec in enumerate(specs):
        codes = spec.codes[remaining]
        valid_mask = codes >= 0

        deficit_ratio = max(0.0, (min_valid - valid_counts[idx]) / min_valid)
        if deficit_ratio > 0:
            scores[valid_mask] += coverage_weight * deficit_ratio

        if valid_mask.any():
            improvements = distribution_improvements(counts_by_col[idx], spec.targets)
            scores[valid_mask] += dist_weight * improvements[codes[valid_mask]]

    best_mask = scores >= (scores.max() - 1e-12)
    candidates = remaining[best_mask]
    candidate_weights = weights[candidates]
    if np.isfinite(candidate_weights).all() and candidate_weights.sum() > 0:
        probs = candidate_weights / candidate_weights.sum()
    else:
        probs = np.repeat(1.0 / len(candidates), len(candidates))
    return int(rng.choice(candidates, p=probs))


def extract_sample(
    df: pd.DataFrame,
    specs: list[ControlSpec],
    sample_size: int,
    min_valid: int,
    seed: int,
    coverage_weight: float,
    dist_weight: float,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    if sample_size > len(df):
        raise ValueError(f"抽样条数 {sample_size} 大于样本总数 {len(df)}")

    rng = np.random.default_rng(seed)
    weights = df[WEIGHT_COL].to_numpy(dtype=float)
    selected = np.zeros(len(df), dtype=bool)
    picked: list[int] = []
    counts_by_col = [np.zeros(len(spec.labels), dtype=int) for spec in specs]
    valid_counts = np.zeros(len(specs), dtype=int)

    for step in range(sample_size):
        remaining = np.flatnonzero(~selected)
        chosen = pick_one(
            remaining=remaining,
            specs=specs,
            counts_by_col=counts_by_col,
            valid_counts=valid_counts,
            weights=weights,
            min_valid=min_valid,
            coverage_weight=coverage_weight,
            dist_weight=dist_weight,
            rng=rng,
        )
        selected[chosen] = True
        picked.append(chosen)

        for idx, spec in enumerate(specs):
            code = int(spec.codes[chosen])
            if code >= 0:
                counts_by_col[idx][code] += 1
                valid_counts[idx] += 1

        if (step + 1) % 100 == 0 or step + 1 == sample_size:
            below = int((valid_counts < min_valid).sum())
            print(f"已抽取 {step + 1}/{sample_size}，尚未达到最低有效数的控制列: {below}")

    return np.array(picked, dtype=int), counts_by_col, valid_counts


def build_report(
    specs: list[ControlSpec],
    counts_by_col: list[np.ndarray],
    valid_counts: np.ndarray,
    min_valid: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx, spec in enumerate(specs):
        n_valid = int(valid_counts[idx])
        sample_props = (
            counts_by_col[idx] / n_valid if n_valid > 0 else np.zeros(len(spec.labels), dtype=float)
        )
        abs_diff = np.abs(sample_props - spec.targets)
        for label, target, count, prop, diff in zip(
            spec.labels,
            spec.targets,
            counts_by_col[idx],
            sample_props,
            abs_diff,
            strict=True,
        ):
            rows.append(
                {
                    "列名": spec.name,
                    "类别": label,
                    "目标比例": round(float(target) * 100, 1),
                    "抽样比例": round(float(prop) * 100, 1),
                    "绝对偏差百分点": round(float(diff) * 100, 1),
                    "抽样人数": int(count),
                    "该列有效样本数": n_valid,
                    "达到最低有效数": "是" if n_valid >= min_valid else "否",
                    "列总偏差百分点": round(float(abs_diff.sum()) * 100, 1),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    df = pd.read_stata(DATA_PATH, convert_categoricals=False)
    rules = pd.read_csv(RULES_PATH, encoding="utf-8-sig")
    specs = build_control_specs(df, rules)

    print(f"输入样本: {DATA_PATH.name}，共 {len(df)} 条")
    print(f"抽样条数: {args.n}，最低有效样本阈值: {args.min_valid}，随机种子: {args.seed}")

    picked, counts_by_col, valid_counts = extract_sample(
        df=df,
        specs=specs,
        sample_size=args.n,
        min_valid=args.min_valid,
        seed=args.seed,
        coverage_weight=args.coverage_weight,
        dist_weight=args.dist_weight,
    )

    sampled = df.iloc[picked].copy()
    sampled.insert(0, "sample_pick_order", np.arange(1, len(sampled) + 1))

    stem = f"用户画像_抽样{args.n}"
    out_dta = DATA_DIR / f"{stem}.dta"
    out_csv = DATA_DIR / f"{stem}.csv"
    report_path = DATA_DIR / f"{stem}_分布对比.csv"

    sampled.to_stata(out_dta, write_index=False, version=118)
    sampled.to_csv(out_csv, index=False, encoding="utf-8-sig")

    report = build_report(specs, counts_by_col, valid_counts, args.min_valid)
    report.to_csv(report_path, index=False, encoding="utf-8-sig")

    print(f"\n已写入样本: {out_dta}")
    print(f"已写入样本副本: {out_csv}")
    print(f"已写入分布报告: {report_path}\n")

    summary = (
        report.groupby("列名", as_index=False)
        .agg(
            该列有效样本数=("该列有效样本数", "max"),
            达到最低有效数=("达到最低有效数", "max"),
            列总偏差百分点=("列总偏差百分点", "max"),
        )
        .sort_values("列名")
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
