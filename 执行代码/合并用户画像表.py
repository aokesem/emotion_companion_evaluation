# -*- coding: utf-8 -*-
"""六张瘦表合并为个体宽表（方案 A：权重健在骨架 + left join）。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "CHARLS2020r 用户画像数据"
OUT_PATH = DATA_DIR / "用户画像_合并.dta"

MERGE_ORDER = [
    ("Weights.dta", "ID"),
    ("Demographic_Background.dta", "ID"),
    ("Health_Status_and_Functioning.dta", "ID"),
    ("Work_Retirement.dta", "ID"),
    ("Family_Information.dta", "householdID"),
]


def _drop_overlap(df_right: pd.DataFrame, base_cols: set[str], key: str) -> pd.DataFrame:
    """去掉与左表重复的键列（保留 householdID 仅一份时用）。"""
    drop = [c for c in df_right.columns if c in base_cols and c != key]
    if drop:
        df_right = df_right.drop(columns=drop)
    return df_right


def main() -> None:
    sample = pd.read_stata(DATA_DIR / "Sample_Infor.dta", convert_categoricals=False)
    n0 = len(sample)
    base = sample[(sample["crosssection"] == 1) & (sample["died"] == 0)].copy()
    print(f"Sample_Infor: {n0} -> 骨架 crosssection=1 & died=0: {len(base)}")

    for fname, key in MERGE_ORDER:
        path = DATA_DIR / fname
        right = pd.read_stata(path, convert_categoricals=False)
        n_right = len(right)
        if key == "ID":
            dup = right["ID"].duplicated().sum()
            if dup:
                raise ValueError(f"{fname}: ID 重复 {dup} 行")
            right = _drop_overlap(right, set(base.columns), key)
            base = base.merge(right, on="ID", how="left", validate="one_to_one")
        else:
            dup = right["householdID"].duplicated().sum()
            if dup:
                raise ValueError(f"{fname}: householdID 重复 {dup} 行")
            right = _drop_overlap(right, set(base.columns), key)
            base = base.merge(right, on="householdID", how="left", validate="m:1")
        print(f"  + {fname} ({n_right} 行) -> 合并后 {len(base)} 行, {base.shape[1]} 列")

    base.to_stata(OUT_PATH, write_index=False, version=118)
    print(f"\n已写出: {OUT_PATH}")
    print(f"最终: {len(base)} 行 × {base.shape[1]} 列")


if __name__ == "__main__":
    main()
