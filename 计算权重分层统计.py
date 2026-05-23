# -*- coding: utf-8 -*-
"""按 用户画像权重分层.csv 对分析样本做加权边际统计，写回 CSV。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "CHARLS2020r 用户画像数据" / "用户画像_分析样本.dta"
RULES_PATH = ROOT / "用户画像权重分层.csv"


def weighted_pct(sub: pd.DataFrame, label: str) -> float:
    w = sub["INDV_weight_ad2"].sum()
    if w <= 0:
        return 0.0
    return 100.0 * sub["INDV_weight_ad2"].sum() / w


def fmt_parts(parts: list[tuple[str, float]]) -> str:
    """[(标签, 占比), ...] -> 男45.2%，女54.8%"""
    return "，".join(f"{lab}{pct:.1f}%" for lab, pct in parts)


def stat_ba001(df: pd.DataFrame) -> str:
    sub = df[df["ba001"].isin([1, 2])]
    wtot = sub["INDV_weight_ad2"].sum()
    parts = []
    for code, lab in [(1, "男"), (2, "女")]:
        m = sub["ba001"] == code
        pct = 100 * sub.loc[m, "INDV_weight_ad2"].sum() / wtot
        parts.append((lab, pct))
    return fmt_parts(parts)


def stat_xrage(df: pd.DataFrame) -> str:
    sub = df[df["xrage"].notna()].copy()
    sub["age_grp"] = pd.cut(
        sub["xrage"],
        bins=[44, 59, 69, np.inf],
        labels=["45–59岁", "60–69岁", "70岁及以上"],
    )
    wtot = sub["INDV_weight_ad2"].sum()
    parts = []
    for lab in ["45–59岁", "60–69岁", "70岁及以上"]:
        m = sub["age_grp"] == lab
        pct = 100 * sub.loc[m, "INDV_weight_ad2"].sum() / wtot
        parts.append((lab, pct))
    return fmt_parts(parts)


def stat_ba007(df: pd.DataFrame) -> str:
    sub = df[df["ba007"].isin([1, 2, 3, 4])]
    wtot = sub["INDV_weight_ad2"].sum()
    home = sub["ba007"] == 1
    inst = sub["ba007"].isin([2, 3, 4])
    return fmt_parts(
        [
            ("家庭住房", 100 * sub.loc[home, "INDV_weight_ad2"].sum() / wtot),
            ("机构或其他", 100 * sub.loc[inst, "INDV_weight_ad2"].sum() / wtot),
        ]
    )


def stat_ba008(df: pd.DataFrame) -> str:
    sub = df[df["ba008"].isin([1, 2, 3, 4])]
    wtot = sub["INDV_weight_ad2"].sum()
    urban = sub["ba008"] == 1
    other = sub["ba008"].isin([2, 3, 4])
    return fmt_parts(
        [
            ("城/镇中心", 100 * sub.loc[urban, "INDV_weight_ad2"].sum() / wtot),
            ("城乡结合部等", 100 * sub.loc[other, "INDV_weight_ad2"].sum() / wtot),
        ]
    )


def stat_ba010(df: pd.DataFrame) -> str:
    sub = df.copy()
    sub["edu"] = sub["ba010"].fillna(sub["zredu"])
    valid = sub["edu"].between(1, 11)
    sub = sub[valid]
    wtot = sub["INDV_weight_ad2"].sum()

    def grp(e):
        if e <= 2:
            return "低学历"
        if e <= 7:
            return "中学历"
        return "高学历"

    sub["edu_grp"] = sub["edu"].map(grp)
    parts = []
    for lab in ["低学历", "中学历", "高学历"]:
        m = sub["edu_grp"] == lab
        pct = 100 * sub.loc[m, "INDV_weight_ad2"].sum() / wtot
        parts.append((lab, pct))
    return fmt_parts(parts)


def stat_binary(df: pd.DataFrame, col: str, lab1: str, lab2: str, v1=1, v2=2) -> str:
    sub = df[df[col].isin([v1, v2])]
    wtot = sub["INDV_weight_ad2"].sum()
    return fmt_parts(
        [
            (lab1, 100 * sub.loc[sub[col] == v1, "INDV_weight_ad2"].sum() / wtot),
            (lab2, 100 * sub.loc[sub[col] == v2, "INDV_weight_ad2"].sum() / wtot),
        ]
    )


def stat_ordinal(df: pd.DataFrame, col: str, labels: dict[int, str]) -> str:
    """仅统计 labels 中的有效编码；997/999/NaN 等不进入分母。"""
    codes = sorted(labels.keys())
    sub = df[df[col].isin(codes)]
    wtot = sub["INDV_weight_ad2"].sum()
    parts = []
    for c in codes:
        m = sub[col] == c
        pct = 100 * sub.loc[m, "INDV_weight_ad2"].sum() / wtot
        parts.append((labels[c], pct))
    return fmt_parts(parts)


def stat_da038_s9(df: pd.DataFrame) -> str:
    sub = df[df["da038_s9"].isin([0, 9])]
    wtot = sub["INDV_weight_ad2"].sum()
    return fmt_parts(
        [
            ("有其他活动", 100 * sub.loc[sub["da038_s9"] == 0, "INDV_weight_ad2"].sum() / wtot),
            ("以上都没有", 100 * sub.loc[sub["da038_s9"] == 9, "INDV_weight_ad2"].sum() / wtot),
        ]
    )


def stat_xchildnum(df: pd.DataFrame) -> str:
    sub = df[df["xchildnum"].notna() & (df["xchildnum"] >= 0)]
    wtot = sub["INDV_weight_ad2"].sum()
    # 0 / 1 / 2 / 3+
    g0 = sub["xchildnum"] == 0
    g1 = sub["xchildnum"] == 1
    g2 = sub["xchildnum"] == 2
    g3 = sub["xchildnum"] >= 3
    return fmt_parts(
        [
            ("0个", 100 * sub.loc[g0, "INDV_weight_ad2"].sum() / wtot),
            ("1个", 100 * sub.loc[g1, "INDV_weight_ad2"].sum() / wtot),
            ("2个", 100 * sub.loc[g2, "INDV_weight_ad2"].sum() / wtot),
            ("3个及以上", 100 * sub.loc[g3, "INDV_weight_ad2"].sum() / wtot),
        ]
    )


STAT_FUNCS = {
    "ba001": stat_ba001,
    "xrage": stat_xrage,
    "ba007": stat_ba007,
    "ba008": stat_ba008,
    "ba010": stat_ba010,
    "ba011": lambda d: stat_ordinal(
        d,
        "ba011",
        {
            1: "已婚且配偶同住",
            2: "已婚暂分居",
            3: "分居",
            4: "离婚",
            5: "丧偶",
            6: "未婚",
        },
    ),
    "dc026": lambda d: stat_ordinal(
        d,
        "dc026",
        {1: "非常满意", 2: "很满意", 3: "有点满意", 4: "不太满意", 5: "完全不满意"},
    ),
    "dc024": lambda d: stat_ordinal(
        d,
        "dc024",
        {1: "几乎不孤独", 2: "少量", 3: "有时", 4: "经常"},
    ),
    "da001": lambda d: stat_ordinal(
        d,
        "da001",
        {1: "很好", 2: "好", 3: "一般", 4: "差", 5: "很差"},
    ),
    "da038_s9": stat_da038_s9,
    "da040": lambda d: stat_binary(d, "da040", "使用互联网", "不使用"),
    "xchildnum": stat_xchildnum,
    "fh001": lambda d: stat_binary(d, "fh001", "已退休", "未退休"),
    "ff003": lambda d: stat_binary(d, "ff003", "在找工作", "未找"),
}


BA011_RULE = {
    "列名": "ba011",
    "含义": "婚姻状况",
    "分类规则": "按原始编码：1=已婚且配偶同住；2=已婚暂分居；3=分居；4=离婚；5=丧偶；6=未婚（997/999/缺失不纳入统计）",
}


def normalize_rules(rules: pd.DataFrame) -> pd.DataFrame:
    """将已废弃的 ba012 行替换为 ba011。"""
    if "ba012" not in rules["列名"].values:
        return rules
    rules = rules[rules["列名"] != "ba012"].copy()
    idx = rules.index[rules["列名"] == "ba010"]
    insert_at = int(idx[0]) + 1 if len(idx) else len(rules)
    new_row = pd.DataFrame([{**BA011_RULE, "加权统计": ""}])
    return pd.concat(
        [rules.iloc[:insert_at], new_row, rules.iloc[insert_at:]],
        ignore_index=True,
    )


def main() -> None:
    df = pd.read_stata(DATA_PATH, convert_categoricals=False)
    df = df[df["INDV_weight_ad2"].notna() & (df["INDV_weight_ad2"] > 0)]
    print(f"分析样本（有权重）: {len(df)} 人，权重和 {df['INDV_weight_ad2'].sum():.0f}")

    rules = normalize_rules(pd.read_csv(RULES_PATH, encoding="utf-8-sig"))
    stats = []
    for col in rules["列名"]:
        fn = STAT_FUNCS.get(col)
        if fn is None:
            stats.append("（未实现）")
        else:
            stats.append(fn(df))
            print(f"{col}: {stats[-1]}")

    rules["加权统计"] = stats
    try:
        rules.to_csv(RULES_PATH, index=False, encoding="utf-8-sig")
        print(f"\n已更新: {RULES_PATH}")
    except PermissionError:
        alt = RULES_PATH.with_name(RULES_PATH.stem + "_新.csv")
        rules.to_csv(alt, index=False, encoding="utf-8-sig")
        print(f"\n原文件被占用，已写入: {alt}")
        print("请关闭 Excel/编辑器后，用该文件覆盖原 CSV，或重新运行本脚本。")


if __name__ == "__main__":
    main()
