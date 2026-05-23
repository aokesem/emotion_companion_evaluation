# -*- coding: utf-8 -*-
"""按 用户画像权重分层.csv 对分析样本做加权边际统计，并回写 CSV。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "CHARLS2020r 用户画像数据" / "用户画像_分析样本.dta"
RULES_PATH = ROOT / "用户画像权重分层.csv"
WEIGHT_COL = "INDV_weight_ad2"


def fmt_parts(parts: list[tuple[str, float]]) -> str:
    return "，".join(f"{label}{pct:.1f}%" for label, pct in parts)


def _weighted_parts(sub: pd.DataFrame, groups: list[tuple[str, pd.Series]]) -> str:
    total = sub[WEIGHT_COL].sum()
    parts: list[tuple[str, float]] = []
    for label, mask in groups:
        pct = 0.0 if total <= 0 else 100.0 * sub.loc[mask, WEIGHT_COL].sum() / total
        parts.append((label, pct))
    return fmt_parts(parts)


def stat_ba001(df: pd.DataFrame) -> str:
    sub = df[df["ba001"].isin([1, 2])]
    return _weighted_parts(
        sub,
        [("男", sub["ba001"] == 1), ("女", sub["ba001"] == 2)],
    )


def stat_xrage(df: pd.DataFrame) -> str:
    sub = df[df["xrage"].notna()].copy()
    sub["age_grp"] = pd.cut(
        sub["xrage"],
        bins=[44, 59, 69, np.inf],
        labels=["45–59岁", "60–69岁", "70岁及以上"],
    )
    return _weighted_parts(
        sub,
        [
            ("45–59岁", sub["age_grp"] == "45–59岁"),
            ("60–69岁", sub["age_grp"] == "60–69岁"),
            ("70岁及以上", sub["age_grp"] == "70岁及以上"),
        ],
    )


def stat_ba007(df: pd.DataFrame) -> str:
    sub = df[df["ba007"].isin([1, 2, 3, 4])]
    return _weighted_parts(
        sub,
        [
            ("家庭住房", sub["ba007"] == 1),
            ("机构或其他", sub["ba007"].isin([2, 3, 4])),
        ],
    )


def stat_ba008(df: pd.DataFrame) -> str:
    sub = df[df["ba008"].isin([1, 2, 3, 4])]
    return _weighted_parts(
        sub,
        [
            ("城/镇中心", sub["ba008"] == 1),
            ("城乡结合部等", sub["ba008"].isin([2, 3, 4])),
        ],
    )


def stat_ba010(df: pd.DataFrame) -> str:
    sub = df.copy()
    sub["edu"] = sub["ba010"].fillna(sub["zredu"])
    sub = sub[sub["edu"].between(1, 11)].copy()

    def group_edu(value: float) -> str:
        if value <= 2:
            return "低学历"
        if value <= 7:
            return "中学历"
        return "高学历"

    sub["edu_grp"] = sub["edu"].map(group_edu)
    return _weighted_parts(
        sub,
        [
            ("低学历", sub["edu_grp"] == "低学历"),
            ("中学历", sub["edu_grp"] == "中学历"),
            ("高学历", sub["edu_grp"] == "高学历"),
        ],
    )


def stat_binary(df: pd.DataFrame, col: str, yes_label: str, no_label: str, yes_value=1, no_value=2) -> str:
    sub = df[df[col].isin([yes_value, no_value])]
    return _weighted_parts(
        sub,
        [(yes_label, sub[col] == yes_value), (no_label, sub[col] == no_value)],
    )


def stat_ba011(df: pd.DataFrame) -> str:
    sub = df[df["ba011"].isin([1, 2, 3, 4, 5, 6])]
    return _weighted_parts(
        sub,
        [
            ("已婚且配偶同住", sub["ba011"] == 1),
            ("分居", sub["ba011"].isin([2, 3])),
            ("离婚或丧偶", sub["ba011"].isin([4, 5])),
            ("未婚", sub["ba011"] == 6),
        ],
    )


def stat_dc026(df: pd.DataFrame) -> str:
    sub = df[df["dc026"].isin([1, 2, 3, 4, 5])]
    return _weighted_parts(
        sub,
        [
            ("满意", sub["dc026"].isin([1, 2])),
            ("有点满意", sub["dc026"] == 3),
            ("不满意", sub["dc026"].isin([4, 5])),
        ],
    )


def stat_dc024(df: pd.DataFrame) -> str:
    sub = df[df["dc024"].isin([1, 2, 3, 4])]
    return _weighted_parts(
        sub,
        [
            ("几乎不孤独", sub["dc024"] == 1),
            ("少量或有时孤独", sub["dc024"].isin([2, 3])),
            ("经常孤独", sub["dc024"] == 4),
        ],
    )


def stat_da001(df: pd.DataFrame) -> str:
    sub = df[df["da001"].isin([1, 2, 3, 4, 5])]
    return _weighted_parts(
        sub,
        [
            ("好或很好", sub["da001"].isin([1, 2])),
            ("一般", sub["da001"] == 3),
            ("差或很差", sub["da001"].isin([4, 5])),
        ],
    )


def stat_da038_s9(df: pd.DataFrame) -> str:
    sub = df[df["da038_s9"].isin([0, 9])]
    return _weighted_parts(
        sub,
        [
            ("有其他活动", sub["da038_s9"] == 0),
            ("以上都没有", sub["da038_s9"] == 9),
        ],
    )


def stat_xchildnum(df: pd.DataFrame) -> str:
    sub = df[df["xchildnum"].notna() & (df["xchildnum"] >= 0)]
    return _weighted_parts(
        sub,
        [
            ("0个", sub["xchildnum"] == 0),
            ("1个", sub["xchildnum"] == 1),
            ("2个", sub["xchildnum"] == 2),
            ("3个及以上", sub["xchildnum"] >= 3),
        ],
    )


STAT_FUNCS = {
    "ba001": stat_ba001,
    "xrage": stat_xrage,
    "ba007": stat_ba007,
    "ba008": stat_ba008,
    "ba010": stat_ba010,
    "ba011": stat_ba011,
    "dc026": stat_dc026,
    "dc024": stat_dc024,
    "da001": stat_da001,
    "da038_s9": stat_da038_s9,
    "da040": lambda d: stat_binary(d, "da040", "使用互联网", "不使用"),
    "xchildnum": stat_xchildnum,
    "fh001": lambda d: stat_binary(d, "fh001", "已退休", "未退休"),
}


RULE_TEXT_UPDATES = {
    "ba011": "按合并后编码：1→「已婚且配偶同住」；2/3→「分居」；4/5→「离婚或丧偶」；6→「未婚」（997/999/缺失不纳入统计）",
    "dc026": "按合并后编码：1/2→「满意」；3→「有点满意」；4/5→「不满意」（分数越高满意度越低；997/999/缺失不纳入统计）",
    "dc024": "按合并后编码：1→「几乎不孤独」；2/3→「少量或有时孤独」；4→「经常孤独」（997/999/缺失不纳入统计）",
    "da001": "按合并后编码：1/2→「好或很好」；3→「一般」；4/5→「差或很差」（997=不知道，不纳入统计）",
}


def normalize_rules(rules: pd.DataFrame) -> pd.DataFrame:
    rules = rules[rules["列名"] != "ff003"].copy()
    for col, text in RULE_TEXT_UPDATES.items():
        rules.loc[rules["列名"] == col, "分类规则"] = text
    return rules


def main() -> None:
    df = pd.read_stata(DATA_PATH, convert_categoricals=False)
    df = df[df[WEIGHT_COL].notna() & (df[WEIGHT_COL] > 0)].copy()
    print(f"分析样本（有权重）: {len(df)} 人，权重和 {df[WEIGHT_COL].sum():.0f}")

    rules = normalize_rules(pd.read_csv(RULES_PATH, encoding="utf-8-sig"))
    stats: list[str] = []
    for col in rules["列名"]:
        fn = STAT_FUNCS.get(col)
        if fn is None:
            stats.append("（未实现）")
        else:
            result = fn(df)
            stats.append(result)
            print(f"{col}: {result}")

    rules["加权统计"] = stats
    try:
        rules.to_csv(RULES_PATH, index=False, encoding="utf-8-sig")
        print(f"\n已更新: {RULES_PATH}")
    except PermissionError:
        alt = RULES_PATH.with_name(RULES_PATH.stem + "_新.csv")
        rules.to_csv(alt, index=False, encoding="utf-8-sig")
        print(f"\n原文件被占用，已写入: {alt}")


if __name__ == "__main__":
    main()
