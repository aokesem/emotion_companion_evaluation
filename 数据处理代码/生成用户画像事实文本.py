# -*- coding: utf-8 -*-
"""将抽样后的用户画像数据转换为半结构化事实文本。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "用户画像数据"
SAMPLE_PATH = DATA_DIR / "用户画像_抽样1000.csv"
CATALOG_PATH = DATA_DIR / "用户画像.csv"
GROUP_PATH = DATA_DIR / "用户画像信息分组.csv"
OUT_PATH = DATA_DIR / "用户画像_抽样1000_事实文本.csv"

MISSING_CODES = {97, 98, 99, 997, 998, 999}

YES_NO_MAP = {1: "是", 2: "否"}
GENDER_MAP = {1: "男", 2: "女"}
RESIDENCE_MAP = {1: "家庭住房", 2: "养老院", 3: "医院", 4: "其他"}
URBAN_MAP = {1: "城/镇中心", 2: "城乡结合部/镇乡", 3: "农村", 4: "特殊地区"}
EDU_MAP = {
    1: "未受教育",
    2: "未读完小学",
    3: "私塾",
    4: "小学",
    5: "初中",
    6: "高中",
    7: "职高",
    8: "大专",
    9: "本科",
    10: "硕士",
    11: "博士",
}
MARRIAGE_MAP = {
    1: "已婚且配偶同住",
    2: "已婚暂分居",
    3: "分居",
    4: "离婚",
    5: "丧偶",
    6: "未婚",
}
SATISFACTION_MAP = {1: "非常满意", 2: "很满意", 3: "有点满意", 4: "不太满意", 5: "完全不满意"}
CESD_FREQ_MAP = {
    1: "几乎没有（<1天）",
    2: "少量（1–2天）",
    3: "有时/中等（3–4天）",
    4: "大多数时候（5–7天）",
}
HEALTH_MAP = {1: "很好", 2: "好", 3: "一般", 4: "差", 5: "很差"}
PAIN_MAP = {1: "无", 2: "一点", 3: "有些", 4: "相当", 5: "非常"}
PURPOSE_MAP = {1: "工作需要", 2: "娱乐", 3: "锻炼", 4: "其他"}
FREQ3_MAP = {1: "几乎每天", 2: "几乎每周", 3: "不规律"}
WORK_ABILITY_MAP = {1: "完全不能", 2: "不能长时间", 3: "没问题"}
HELP_FREQ_MAP = {1: "从不", 2: "偶尔", 3: "大多时候", 4: "本节由代答完成"}
WORK_STUDY_MAP = {1: "工作", 2: "上学", 3: "边工边读", 4: "都不"}
CONTACT_FREQ_MAP = {
    1: "几乎每天",
    2: "每周2–3次",
    3: "每周1次",
    4: "每两周",
    5: "每月",
    6: "每三月",
    7: "每半年",
    8: "每年",
    9: "几乎不见/不联系",
    10: "其他",
}

CHRONIC_COLS = [
    "da003_1_",
    "da003_2_",
    "da003_3_",
    "da003_4_",
    "da003_5_",
    "da003_6_",
    "da003_7_",
    "da003_8_",
    "da003_9_",
    "da003_10_",
    "da003_11_",
    "da003_12_",
    "da003_13_",
    "da003_14_",
    "da003_15_",
]

ACTIVITY_LABELS = {
    "da038_s1": "与朋友交往",
    "da038_s2": "打麻将/下棋/打牌/去社区活动室",
    "da038_s3": "帮助不住一起的家人朋友邻居",
    "da038_s4": "参加运动/社交等俱乐部",
    "da038_s5": "参加社区相关组织",
    "da038_s6": "做志愿/慈善或照顾不住一起的病弱者",
    "da038_s7": "参加教育/培训课程",
    "da038_s8": "其他活动",
}

DEVICE_LABELS = {
    "da041_s1": "台式电脑",
    "da041_s2": "笔记本",
    "da041_s3": "平板",
    "da041_s4": "手机",
    "da041_s5": "其他",
}

USE_LABELS = {
    "da042_s1": "聊天",
    "da042_s2": "看新闻",
    "da042_s3": "看视频",
    "da042_s4": "玩游戏",
    "da042_s5": "理财",
    "da042_s6": "其他",
}


def is_missing(value: object) -> bool:
    if pd.isna(value):
        return True
    if isinstance(value, str):
        text = value.strip()
        if text == "" or text.lower() == "nan":
            return True
    if isinstance(value, (int, float)):
        try:
            if int(value) in MISSING_CODES and float(value).is_integer():
                return True
        except OverflowError:
            return False
    return False


def as_int(value: object) -> int | None:
    if is_missing(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def as_number_text(value: object) -> str | None:
    if is_missing(value):
        return None
    num = float(value)
    if num.is_integer():
        return str(int(num))
    return f"{num:.1f}".rstrip("0").rstrip(".")


def map_by_dict(value: object, mapping: dict[int, str]) -> str | None:
    code = as_int(value)
    if code is None:
        return None
    return mapping.get(code)


def load_catalog() -> dict[str, dict[str, str]]:
    df = pd.read_csv(CATALOG_PATH, encoding="utf-8-sig")
    out: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        out[str(row["列名"])] = {
            "label": str(row["含义"]),
            "simple": str(row["简单释义"]),
        }
    return out


def load_groups() -> list[tuple[str, list[str]]]:
    df = pd.read_csv(GROUP_PATH, encoding="utf-8-sig")
    groups: list[tuple[str, list[str]]] = []
    for _, row in df.iterrows():
        cols = [part.strip() for part in str(row["包含列名"]).split(",") if part.strip()]
        groups.append((str(row["信息名称"]), cols))
    return groups


def format_field(label: str, value: str | None) -> str | None:
    if value is None or value == "":
        return None
    return f"{label}：{value}"


def build_basic_parts(row: pd.Series) -> list[str]:
    parts: list[str] = []
    parts.append(format_field("性别", map_by_dict(row.get("ba001"), GENDER_MAP)))

    age = as_number_text(row.get("xrage"))
    if age is not None:
        parts.append(f"年龄：{age}岁")

    parts.append(format_field("现居住地地址类型", map_by_dict(row.get("ba007"), RESIDENCE_MAP)))
    parts.append(format_field("城乡类型", map_by_dict(row.get("ba008"), URBAN_MAP)))
    return [part for part in parts if part]


def build_education_parts(row: pd.Series) -> list[str]:
    edu = row.get("ba010")
    if is_missing(edu):
        edu = row.get("zredu")
    value = map_by_dict(edu, EDU_MAP)
    item = format_field("教育程度", value)
    return [item] if item else []


def build_marriage_parts(row: pd.Series) -> list[str]:
    parts: list[str] = []
    parts.append(format_field("婚姻状况", map_by_dict(row.get("ba011"), MARRIAGE_MAP)))
    parts.append(format_field("是否有人以配偶身份同住", map_by_dict(row.get("ba012"), YES_NO_MAP)))

    months = as_number_text(row.get("ba013"))
    if months is not None:
        parts.append(f"过去一年与配偶同住月数：{months}个月")

    days_alone = as_number_text(row.get("ba018"))
    if days_alone is not None:
        parts.append(f"上半年独自居住天数：{days_alone}天")

    spouse_days = as_number_text(row.get("ba019"))
    if spouse_days is not None:
        parts.append(f"上半年仅与配偶同住天数：{spouse_days}天")

    impact = as_number_text(row.get("ba020"))
    if impact is not None:
        parts.append(f"上半年因不与别人同住而受影响程度：{impact}")
    return [part for part in parts if part]


def build_emotion_parts(row: pd.Series) -> list[str]:
    parts: list[str] = []
    parts.append(format_field("生活满意度", map_by_dict(row.get("dc026"), SATISFACTION_MAP)))
    child_sat = map_by_dict(row.get("dc027"), {**SATISFACTION_MAP, 6: "无子女"})
    parts.append(format_field("对子女满意度", child_sat))
    parts.append(format_field("因小事烦恼（过去一周）", map_by_dict(row.get("dc016"), CESD_FREQ_MAP)))
    parts.append(format_field("感到抑郁（过去一周）", map_by_dict(row.get("dc018"), CESD_FREQ_MAP)))
    parts.append(format_field("对未来抱有希望（过去一周）", map_by_dict(row.get("dc020"), CESD_FREQ_MAP)))
    parts.append(format_field("感到孤独（过去一周）", map_by_dict(row.get("dc024"), CESD_FREQ_MAP)))
    parts.append(format_field("感到高兴（过去一周）", map_by_dict(row.get("dc023"), CESD_FREQ_MAP)))
    parts.append(format_field("睡眠不安（过去一周）", map_by_dict(row.get("dc022"), CESD_FREQ_MAP)))
    return [part for part in parts if part]


def build_health_parts(row: pd.Series) -> list[str]:
    parts: list[str] = []
    parts.append(format_field("自评健康状况", map_by_dict(row.get("da001"), HEALTH_MAP)))

    diseases: list[str] = []
    disease_valid = False
    for col in CHRONIC_COLS:
        code = as_int(row.get(col))
        if code is None:
            continue
        if code in {1, 2}:
            disease_valid = True
        if code == 1:
            diseases.append(CATALOG[col]["label"].replace("医生诊断：", ""))
    if diseases:
        parts.append(f"医生诊断慢性病：{'、'.join(diseases)}")
    elif disease_valid:
        parts.append("医生诊断慢性病：未报告上述疾病")

    parts.append(format_field("身体疼痛程度", map_by_dict(row.get("da027"), PAIN_MAP)))

    sleep_hours = as_number_text(row.get("da030"))
    if sleep_hours is not None:
        parts.append(f"上月平均每天实际睡眠小时数：{sleep_hours}小时")

    work_limit = map_by_dict(row.get("db043"), {1: "完全不能工作", 2: "不能长时间工作", 3: "没问题"})
    parts.append(format_field("是否因健康/残疾不能工作", work_limit))

    house_limit = map_by_dict(row.get("db044"), WORK_ABILITY_MAP)
    parts.append(format_field("是否因健康/残疾不能做家事", house_limit))

    parts.append(format_field("多久接受他人帮助", map_by_dict(row.get("db045"), HELP_FREQ_MAP)))
    return [part for part in parts if part]


def build_activity_parts(row: pd.Series) -> list[str]:
    parts: list[str] = []

    for suffix, label in [("1", "高强度体力活动"), ("2", "中等强度体力活动"), ("3", "轻度体力活动")]:
        active = map_by_dict(row.get(f"da032_{suffix}_"), YES_NO_MAP)
        parts.append(format_field(label, active))
        purpose = map_by_dict(row.get(f"da037_{suffix}_"), PURPOSE_MAP)
        parts.append(format_field(f"{label}目的", purpose))

    chosen_activities: list[str] = []
    for idx, (col, label) in enumerate(ACTIVITY_LABELS.items(), start=1):
        code = as_int(row.get(col))
        if code == idx:
            freq = map_by_dict(row.get(f"da039_{idx}_"), FREQ3_MAP)
            chosen_activities.append(f"{label}（{freq}）" if freq else label)

    none_code = as_int(row.get("da038_s9"))
    if chosen_activities:
        parts.append(f"上月活动：{'、'.join(chosen_activities)}")
    elif none_code == 9:
        parts.append("上月活动：以上都没有")

    return [part for part in parts if part]


def build_online_parts(row: pd.Series) -> list[str]:
    parts: list[str] = []
    internet = map_by_dict(row.get("da040"), YES_NO_MAP)
    parts.append(format_field("是否使用互联网", internet))

    if internet == "是":
        devices = [label for col, label in DEVICE_LABELS.items() if as_int(row.get(col)) and as_int(row.get(col)) > 0]
        if devices:
            parts.append(f"上网设备：{'、'.join(devices)}")

        uses = [label for col, label in USE_LABELS.items() if as_int(row.get(col)) and as_int(row.get(col)) > 0]
        if uses:
            parts.append(f"上网用途：{'、'.join(uses)}")

        parts.append(format_field("是否用手机支付", map_by_dict(row.get("da043"), YES_NO_MAP)))
        parts.append(format_field("是否使用微信", map_by_dict(row.get("da044"), YES_NO_MAP)))
        parts.append(format_field("是否使用微信朋友圈", map_by_dict(row.get("da045"), YES_NO_MAP)))

    return [part for part in parts if part]


def build_child_summary(row: pd.Series, idx: int) -> str | None:
    parts: list[str] = []

    birth_year = as_number_text(row.get(f"ca005_{idx}_"))
    if birth_year is not None:
        parts.append(f"出生年：{birth_year}")

    parts.append(format_field("性别", map_by_dict(row.get(f"ca006_{idx}_"), GENDER_MAP)))
    parts.append(format_field("最高学历", map_by_dict(row.get(f"ca007_{idx}_"), EDU_MAP)))
    parts.append(format_field("是否工作/上学", map_by_dict(row.get(f"ca008_{idx}_"), WORK_STUDY_MAP)))
    parts.append(format_field("婚姻状况", map_by_dict(row.get(f"ca010_{idx}_"), MARRIAGE_MAP)))

    stay = as_number_text(row.get(f"ca014_{idx}_"))
    if stay is not None:
        parts.append(f"过去一年同住月数：{stay}个月")

    parts.append(format_field("见面频率", map_by_dict(row.get(f"ca015_{idx}_"), CONTACT_FREQ_MAP)))
    parts.append(format_field("电话/短信/微信联系频率", map_by_dict(row.get(f"ca016_{idx}_"), CONTACT_FREQ_MAP)))

    clean_parts = [part for part in parts if part]
    if not clean_parts:
        return None
    return f"第{idx}子女：{'；'.join(clean_parts)}"


def build_children_parts(row: pd.Series) -> list[str]:
    parts: list[str] = []

    child_num = as_number_text(row.get("xchildnum"))
    if child_num is not None:
        parts.append(f"子女总数：{child_num}个")

    alive_num = as_number_text(row.get("xchildalivenum"))
    if alive_num is not None:
        parts.append(f"健在子女数：{alive_num}个")

    for idx in (1, 2):
        summary = build_child_summary(row, idx)
        if summary:
            parts.append(summary)
    return parts


def build_work_parts(row: pd.Series) -> list[str]:
    parts: list[str] = []
    parts.append(format_field("是否从事农业/家庭农活", map_by_dict(row.get("fa001"), {1: "有", 2: "无"})))
    retire = map_by_dict(row.get("fh001"), YES_NO_MAP)
    parts.append(format_field("是否已办理退休", retire))

    year = as_number_text(row.get("fh003_1"))
    month = as_number_text(row.get("fh003_2"))
    if year is not None:
        parts.append(f"退休时间：{year}年{month}月" if month is not None else f"退休年份：{year}年")

    work_now = map_by_dict(row.get("xworking"), {0: "否", 1: "是"})
    parts.append(format_field("是否在工作", work_now))

    days = as_number_text(row.get("ff001"))
    if days is not None:
        parts.append(f"过去一年总工作天数：{days}天")

    parts.append(format_field("上月是否在找工作", map_by_dict(row.get("ff003"), YES_NO_MAP)))
    return [part for part in parts if part]


def build_group_text(row: pd.Series, group_name: str) -> str:
    if group_name == "基础信息":
        parts = build_basic_parts(row)
    elif group_name == "教育信息":
        parts = build_education_parts(row)
    elif group_name == "婚姻信息":
        parts = build_marriage_parts(row)
    elif group_name == "情绪信息":
        parts = build_emotion_parts(row)
    elif group_name == "健康信息":
        parts = build_health_parts(row)
    elif group_name == "活动信息":
        parts = build_activity_parts(row)
    elif group_name == "上网信息":
        parts = build_online_parts(row)
    elif group_name == "子女信息":
        parts = build_children_parts(row)
    elif group_name == "工作信息":
        parts = build_work_parts(row)
    else:
        parts = []

    if not parts:
        return ""
    return f"{group_name}：{'；'.join(parts)}"


def main() -> None:
    global CATALOG
    CATALOG = load_catalog()
    groups = [name for name, _ in load_groups()]
    sample = pd.read_csv(SAMPLE_PATH, encoding="utf-8-sig")

    out = sample[["sample_pick_order", "ID"]].copy()
    fact_columns: list[str] = []

    for group_name in groups:
        col_name = f"{group_name}文本"
        out[col_name] = sample.apply(lambda row, name=group_name: build_group_text(row, name), axis=1)
        fact_columns.append(col_name)

    out["画像事实文本"] = out[fact_columns].apply(
        lambda row: "\n".join([text for text in row if isinstance(text, str) and text.strip()]),
        axis=1,
    )

    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"已写入: {OUT_PATH}")
    print(out.head(3).to_string(index=False))


if __name__ == "__main__":
    CATALOG: dict[str, dict[str, str]] = {}
    main()
