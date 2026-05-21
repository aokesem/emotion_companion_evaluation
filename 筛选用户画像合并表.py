# -*- coding: utf-8 -*-
"""在 用户画像_合并.dta 上筛分析样本并输出新 .dta。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "CHARLS2020r 用户画像数据"
IN_PATH = DATA_DIR / "用户画像_合并.dta"
OUT_PATH = DATA_DIR / "用户画像_分析样本.dta"


def main() -> None:
    df = pd.read_stata(IN_PATH, convert_categoricals=False)
    n0 = len(df)

    mask = (
        (df["crosssection"] == 1)
        & (df["died"] == 0)
        & (df["INDV_weight_ad2"].notna())
        & (df["INDV_weight_ad2"] > 0)
        & (df["proxy_2"] != 1)
        & (df["proxy_5"] != 1)
    )
    out = df.loc[mask].copy()

    print(f"输入: {IN_PATH.name}  {n0} 行")
    print(f"  crosssection!=1: {(df['crosssection']!=1).sum()}")
    print(f"  died!=0: {(df['died']!=0).sum()}")
    print(f"  权重缺失或<=0: {(df['INDV_weight_ad2'].isna() | (df['INDV_weight_ad2']<=0)).sum()}")
    print(f"  proxy_2=1: {(df['proxy_2']==1).sum()}")
    print(f"  proxy_5=1: {(df['proxy_5']==1).sum()}")
    print(f"输出: {OUT_PATH.name}  {len(out)} 行 × {out.shape[1]} 列")

    out.to_stata(OUT_PATH, write_index=False, version=118)
    print("完成。")


if __name__ == "__main__":
    main()
