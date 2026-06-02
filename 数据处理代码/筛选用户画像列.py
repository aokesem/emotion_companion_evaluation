# -*- coding: utf-8 -*-
"""各 .dta 仅保留 用户画像.csv 登记列 + 合并用键（ID / householdID）。"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "用户画像.csv"
DATA_DIR = ROOT / "CHARLS2020r 用户画像数据"

INDIVIDUAL_KEYS = ["ID", "householdID"]
FAMILY_KEYS = ["householdID"]

# 个体表：登记列 + ID + householdID；Family 仅 householdID + 登记列
FILE_KEYS = {
    "Sample_Infor.dta": INDIVIDUAL_KEYS,
    "Weights.dta": ["ID"],  # 登记列仅权重；ID 即可并表
    "Demographic_Background.dta": INDIVIDUAL_KEYS,
    "Health_Status_and_Functioning.dta": INDIVIDUAL_KEYS,
    "Work_Retirement.dta": INDIVIDUAL_KEYS,
    "Family_Information.dta": FAMILY_KEYS,
}


def main() -> None:
    catalog = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    by_source: dict[str, list[str]] = {}
    for src, grp in catalog.groupby("来源数据集"):
        by_source[src] = grp["列名"].astype(str).tolist()

    for fname, keys in FILE_KEYS.items():
        path = DATA_DIR / fname
        if not path.exists():
            raise FileNotFoundError(path)

        portrait_cols = by_source.get(fname, [])
        keep: list[str] = []
        seen: set[str] = set()
        for c in keys + portrait_cols:
            if c not in seen:
                keep.append(c)
                seen.add(c)

        df_full = pd.read_stata(path, convert_categoricals=False)
        missing = [c for c in keep if c not in df_full.columns]
        if missing:
            raise KeyError(f"{fname} 缺少列: {missing}")

        df = df_full[keep].copy()
        n_before, p_before = df_full.shape[1], path.stat().st_size
        df.to_stata(path, write_index=False, version=118)
        n_after, p_after = df.shape[1], path.stat().st_size

        print(
            f"{fname}: {df_full.shape[0]} 行, "
            f"列 {n_before} -> {n_after}, "
            f"体积 {p_before // 1024} KB -> {p_after // 1024} KB"
        )
        print(f"  保留: {', '.join(keep)}")

    print("\n完成。已就地覆盖 CHARLS2020r 用户画像数据 文件夹中的 6 个 .dta。")


if __name__ == "__main__":
    main()
