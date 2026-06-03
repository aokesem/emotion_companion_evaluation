# -*- coding: utf-8 -*-
"""从分析样本中按个体权重概率有放回抽取用户画像样本。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "整理后数据" / "用户画像_分析样本.dta"
OUTPUT_DIR = ROOT / "用户画像数据"
WEIGHT_COL = "INDV_weight_ad2"
AGE_COL = "xrage"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按权重概率有放回抽取用户画像样本")
    parser.add_argument("--n", type=int, default=500, help="抽样条数")
    parser.add_argument("--seed", type=int, default=20260602, help="随机种子")
    parser.add_argument("--input", type=Path, default=INPUT_PATH, help="输入 dta 文件路径")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="输出目录")
    parser.add_argument("--weight-col", type=str, default=WEIGHT_COL, help="权重列名")
    parser.add_argument("--age-col", type=str, default=AGE_COL, help="年龄列名")
    parser.add_argument("--min-age", type=float, default=45.0, help="抽样年龄下限")
    return parser.parse_args()


def normalized_probabilities(weights: pd.Series) -> np.ndarray:
    probs = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(probs) & (probs > 0)
    if not valid.any():
        raise ValueError("没有可用于抽样的正权重")
    probs = np.where(valid, probs, 0.0)
    return probs / probs.sum()


def main() -> None:
    args = parse_args()
    if args.n <= 0:
        raise ValueError("--n 必须为正整数")
    if not args.input.exists():
        raise FileNotFoundError(f"输入文件不存在: {args.input}")

    df = pd.read_stata(args.input, convert_categoricals=False)
    if args.weight_col not in df.columns:
        raise KeyError(f"输入数据缺少权重列: {args.weight_col}")
    if args.age_col not in df.columns:
        raise KeyError(f"输入数据缺少年龄列: {args.age_col}")

    input_count = len(df)
    ages = pd.to_numeric(df[args.age_col], errors="coerce")
    df = df.loc[ages >= args.min_age].copy()
    if df.empty:
        raise ValueError(f"年龄大于等于 {args.min_age:g} 岁的样本为空")

    probs = normalized_probabilities(df[args.weight_col])
    rng = np.random.default_rng(args.seed)
    picked = rng.choice(df.index.to_numpy(), size=args.n, replace=True, p=probs)

    sampled = df.loc[picked].copy()
    sampled.insert(0, "sample_pick_order", np.arange(1, args.n + 1))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"用户画像_抽样{args.n}"
    out_dta = args.output_dir / f"{stem}.dta"
    out_csv = args.output_dir / f"{stem}.csv"

    sampled.to_stata(out_dta, write_index=False, version=118)
    sampled.to_csv(out_csv, index=False, encoding="utf-8-sig")

    duplicate_ids = int(sampled["ID"].duplicated().sum()) if "ID" in sampled.columns else "无法统计"
    unique_ids = int(sampled["ID"].nunique()) if "ID" in sampled.columns else "无法统计"

    print(f"输入样本: {args.input}，共 {input_count} 条")
    print(f"年龄筛选: {args.age_col} >= {args.min_age:g}，保留 {len(df)} 条")
    print(f"抽样条数: {args.n}，随机种子: {args.seed}")
    print(f"唯一 ID 数: {unique_ids}，重复 ID 行数: {duplicate_ids}")
    print(f"已写入: {out_dta}")
    print(f"已写入: {out_csv}")


if __name__ == "__main__":
    main()
