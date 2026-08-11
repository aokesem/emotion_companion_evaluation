# -*- coding: utf-8 -*-
"""Cross-score fixed dialog transcripts with official and lab user-AI APIs."""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from run_langgraph_workflow import (
    LabAgentClient,
    OpenAICompatClient,
    ROOT,
    load_config,
    load_dotenv,
    load_sim_user_records,
    normalize_base_url,
    normalized_id,
    resolve_output_dir,
    resolve_path,
    serialize_messages_for_lab,
    transcript_text,
)

CONDITIONS = ("official-native", "official-serialized", "lab-serialized")
RATING_KEYS = ("被理解感", "情绪缓解感", "个性化贴合度", "交流舒适度")
RESULT_NAME = "subjective_rescore_runs.jsonl"
ERROR_NAME = "subjective_rescore_errors.jsonl"
SUMMARY_NAME = "subjective_rescore_summary.csv"


def resolve_input(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / "outputs" / path
    if path.is_dir() or not path.suffix:
        path = path / "dialog_eval_runs.jsonl"
    return path.resolve()


def parse_args() -> argparse.Namespace:
    config = load_config()
    defaults = config["defaults"]
    paths = config["paths"]
    parser = argparse.ArgumentParser(
        description="对固定对话执行 Official/Lab 用户 AI 主观评分交叉实验"
    )
    parser.add_argument(
        "--official-input-dir",
        default="deepseek_v4_pro-official-paired",
        help="Official 对话目录或 dialog_eval_runs.jsonl 路径",
    )
    parser.add_argument(
        "--lab-input-dir",
        default="deepseek_v4_pro",
        help="Lab 对话目录或 dialog_eval_runs.jsonl 路径",
    )
    parser.add_argument(
        "--output-dir",
        default="deepseek_v4_pro-subjective-cross-rescore",
        help="新评分输出目录；相对路径位于 outputs/ 下",
    )
    parser.add_argument(
        "--sim-user-jsonl",
        type=Path,
        default=resolve_path(paths["sim_user_jsonl"]),
        help="模拟用户信息 JSONL",
    )
    parser.add_argument(
        "--conditions",
        default=",".join(CONDITIONS),
        help="逗号分隔的评分条件",
    )
    parser.add_argument("--start-pick-order", type=int, default=None, help="最小 sample_pick_order")
    parser.add_argument("--end-pick-order", type=int, default=None, help="最大 sample_pick_order")
    parser.add_argument("--limit", type=int, default=0, help="最多处理的配对样本数；0 表示全部")
    parser.add_argument(
        "--simulated-user-model",
        default=defaults["simulated_user_model"],
        help="Official 用户 AI 模型名",
    )
    parser.add_argument(
        "--official-base-url",
        default=os.getenv("SIMULATED_USER_BASE_URL") or os.getenv("OPENAI_BASE_URL", ""),
        help="Official 用户 AI Base URL",
    )
    parser.add_argument(
        "--official-api-key",
        default=os.getenv("SIMULATED_USER_API_KEY") or os.getenv("OPENAI_API_KEY", ""),
        help="Official 用户 AI API Key",
    )
    parser.add_argument(
        "--lab-api-url",
        default=os.getenv("LAB_AGENT_API_URL", "https://ithink.isapientia.com/api/app/utv/v1/agent/qa"),
        help="实验室智能体 API 地址",
    )
    parser.add_argument(
        "--lab-token",
        default=os.getenv("LAB_SIMULATED_USER_TOKEN", ""),
        help="实验室用户 AI 工作流 Token",
    )
    parser.add_argument("--temperature", type=float, default=defaults["temperature"])
    parser.add_argument("--max-tokens", type=int, default=defaults["max_tokens"])
    parser.add_argument("--timeout", type=int, default=defaults["timeout"])
    parser.add_argument("--retries", type=int, default=defaults["retries"])
    parser.add_argument("--sleep", type=float, default=defaults["sleep"])
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--regenerate-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug-http", action="store_true")
    args = parser.parse_args()

    args.official_input = resolve_input(args.official_input_dir)
    args.lab_input = resolve_input(args.lab_input_dir)
    args.output_dir = resolve_output_dir(args.output_dir)
    args.result_jsonl = args.output_dir / RESULT_NAME
    args.error_jsonl = args.output_dir / ERROR_NAME
    args.summary_csv = args.output_dir / SUMMARY_NAME
    requested = [item.strip() for item in args.conditions.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(CONDITIONS))
    if unknown:
        raise ValueError(f"未知评分条件: {', '.join(unknown)}；可用条件: {', '.join(CONDITIONS)}")
    if not requested:
        raise ValueError("至少需要一个评分条件")
    args.conditions = requested
    return args


def load_source_records(path: Path, provider: str) -> dict[int, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"对话记录不存在: {path}")
    selected: dict[int, dict[str, Any]] = {}
    duplicates: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_no} 行不是合法 JSON: {exc}") from exc
            record_provider = str(record.get("run_config", {}).get("aux_provider", "")).strip()
            if record_provider != provider:
                continue
            try:
                pick_order = int(float(record["sample_pick_order"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path} 第 {line_no} 行缺少有效 sample_pick_order") from exc
            if pick_order in selected:
                duplicates.append(pick_order)
            selected[pick_order] = record
    if duplicates:
        values = ", ".join(str(value) for value in sorted(set(duplicates)))
        raise ValueError(f"{path} 存在重复 sample_pick_order: {values}")
    if not selected:
        raise ValueError(f"{path} 中没有 aux_provider={provider} 的记录")
    return selected


def pair_sources(
    official: dict[int, dict[str, Any]],
    lab: dict[int, dict[str, Any]],
    start_pick_order: int | None = None,
    end_pick_order: int | None = None,
    limit: int = 0,
) -> list[tuple[int, dict[str, Any], dict[str, Any]]]:
    official_only = sorted(set(official) - set(lab))
    lab_only = sorted(set(lab) - set(official))
    if official_only or lab_only:
        raise ValueError(
            "两种来源的 sample_pick_order 集合不一致。"
            f"仅 Official: {official_only}; 仅 Lab: {lab_only}"
        )
    pairs: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for pick_order in sorted(official):
        if start_pick_order is not None and pick_order < start_pick_order:
            continue
        if end_pick_order is not None and pick_order > end_pick_order:
            continue
        official_record = official[pick_order]
        lab_record = lab[pick_order]
        if normalized_id(official_record.get("ID", "")) != normalized_id(lab_record.get("ID", "")):
            raise ValueError(
                f"sample_pick_order={pick_order} 的 ID 不一致: "
                f"Official={official_record.get('ID')}, Lab={lab_record.get('ID')}"
            )
        pairs.append((pick_order, official_record, lab_record))
        if limit > 0 and len(pairs) >= limit:
            break
    if not pairs:
        raise ValueError("筛选后没有可评分的配对样本")
    return pairs


def render_prompt(path: Path, **values: str) -> str:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"提示词文件为空: {path}")
    for name, value in values.items():
        text = text.replace("{" + name + "}", value)
    return text


def build_rating_messages(
    record: dict[str, Any],
    sim_info: dict[str, Any],
    dialog_prompt_path: Path,
    rating_prompt_path: Path,
) -> list[dict[str, str]]:
    dialog_messages = record.get("dialog_messages")
    if not isinstance(dialog_messages, list) or not dialog_messages:
        raise ValueError("对话记录缺少 dialog_messages")
    opening = next(
        (
            str(message.get("content", "")).strip()
            for message in dialog_messages
            if message.get("speaker") == "simulated_user" and str(message.get("content", "")).strip()
        ),
        "",
    )
    if not opening:
        raise ValueError("对话记录缺少模拟用户开场")
    system_prompt = render_prompt(
        dialog_prompt_path,
        sim_user_info_json=json.dumps(sim_info, ensure_ascii=False, indent=2),
        opening_sentence=opening,
    )
    rating_prompt = render_prompt(
        rating_prompt_path,
        dialog_transcript=transcript_text(dialog_messages),
        dialog_messages_json=json.dumps(dialog_messages, ensure_ascii=False, indent=2),
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    role_map = {"simulated_user": "assistant", "tested_agent": "user"}
    for index, message in enumerate(dialog_messages, start=1):
        speaker = str(message.get("speaker", ""))
        role = role_map.get(speaker)
        if role is None:
            raise ValueError(f"第 {index} 条对话包含未知 speaker: {speaker}")
        content = str(message.get("content", "")).strip()
        if not content:
            raise ValueError(f"第 {index} 条对话内容为空")
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": rating_prompt})
    return messages


def validate_rating(rating: dict[str, Any]) -> None:
    missing = [key for key in RATING_KEYS if key not in rating]
    if missing:
        raise ValueError(f"评分缺少字段: {', '.join(missing)}")
    for key in RATING_KEYS:
        value = rating[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= float(value) <= 3:
            raise ValueError(f"{key} 必须是 0 到 3 的数值，实际为: {value!r}")


def completed_keys(path: Path) -> set[tuple[str, str, int]]:
    completed: set[tuple[str, str, int]] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            try:
                row = json.loads(raw_line)
                completed.add(
                    (
                        str(row["dialog_source"]),
                        str(row["rating_condition"]),
                        int(float(row["sample_pick_order"])),
                    )
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return completed


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(result_path: Path, summary_path: Path) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    if result_path.exists():
        with result_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                if raw_line.strip():
                    row = json.loads(raw_line)
                    groups[(row["dialog_source"], row["rating_condition"])].append(row["rating"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = ["dialog_source", "rating_condition", "count", *RATING_KEYS, "四项平均"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for source, condition in sorted(groups):
            ratings = groups[(source, condition)]
            means = {
                key: sum(float(rating[key]) for rating in ratings) / len(ratings)
                for key in RATING_KEYS
            }
            writer.writerow(
                {
                    "dialog_source": source,
                    "rating_condition": condition,
                    "count": len(ratings),
                    **{key: f"{value:.6f}" for key, value in means.items()},
                    "四项平均": f"{sum(means.values()) / len(RATING_KEYS):.6f}",
                }
            )


def make_clients(args: argparse.Namespace) -> tuple[OpenAICompatClient | None, LabAgentClient | None]:
    need_official = any(condition.startswith("official-") for condition in args.conditions)
    need_lab = "lab-serialized" in args.conditions
    official_client = None
    lab_client = None
    if need_official:
        if not args.official_base_url or not args.official_api_key:
            raise ValueError("缺少 Official 用户 AI 配置，请设置 SIMULATED_USER_BASE_URL/API_KEY")
        official_client = OpenAICompatClient(
            normalize_base_url(args.official_base_url),
            args.official_api_key,
            args.timeout,
            args.retries,
            args.debug_http,
            auto_append_v1=False,
        )
    if need_lab:
        if not args.lab_api_url or not args.lab_token:
            raise ValueError("缺少 Lab 用户 AI 配置，请设置 LAB_AGENT_API_URL/LAB_SIMULATED_USER_TOKEN")
        lab_client = LabAgentClient(
            args.lab_api_url,
            args.lab_token,
            args.timeout,
            args.retries,
            args.debug_http,
        )
    return official_client, lab_client


def score_condition(
    condition: str,
    messages: list[dict[str, str]],
    official_client: OpenAICompatClient | None,
    lab_client: LabAgentClient | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if condition == "official-native":
        if official_client is None:
            raise ValueError("Official 客户端未初始化")
        call_messages = messages
        client = official_client
    elif condition == "official-serialized":
        if official_client is None:
            raise ValueError("Official 客户端未初始化")
        call_messages = [{"role": "user", "content": serialize_messages_for_lab(messages)}]
        client = official_client
    elif condition == "lab-serialized":
        if lab_client is None:
            raise ValueError("Lab 客户端未初始化")
        call_messages = messages
        client = lab_client
    else:
        raise ValueError(f"未知评分条件: {condition}")
    rating = client.chat_json(
        model=args.simulated_user_model,
        messages=call_messages,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    validate_rating(rating)
    return rating


def run(args: argparse.Namespace) -> None:
    official_records = load_source_records(args.official_input, "official")
    lab_records = load_source_records(args.lab_input, "lab")
    pairs = pair_sources(
        official_records,
        lab_records,
        args.start_pick_order,
        args.end_pick_order,
        args.limit,
    )
    config = load_config()
    dialog_prompt_path = resolve_path(config["paths"]["simulated_user_dialog_prompt"])
    rating_prompt_path = resolve_path(config["paths"]["simulated_user_rating_prompt"])
    sim_records = load_sim_user_records(args.sim_user_jsonl)

    print(f"Official 对话: {args.official_input}")
    print(f"Lab 对话: {args.lab_input}")
    print(f"输出目录: {args.output_dir}")
    print(f"配对样本: {len(pairs)} ({pairs[0][0]}-{pairs[-1][0]})")
    print(f"评分条件: {', '.join(args.conditions)}")
    print(f"计划评分: {len(pairs) * 2 * len(args.conditions)}")

    if args.dry_run:
        for pick_order, official_record, _ in pairs:
            print(f"[DRY] sample_pick_order={pick_order} ID={official_record.get('ID')}")
        return

    official_client, lab_client = make_clients(args)
    completed = set() if args.regenerate_existing else completed_keys(args.result_jsonl)
    success_count = 0
    skip_count = 0
    error_count = 0
    for pick_order, official_record, lab_record in pairs:
        for dialog_source, source_record in (
            ("official-dialog", official_record),
            ("lab-dialog", lab_record),
        ):
            row_id = normalized_id(source_record.get("ID", ""))
            sim_info = sim_records.get(row_id)
            if sim_info is None:
                raise ValueError(f"ID={source_record.get('ID')} 缺少模拟用户信息")
            messages = build_rating_messages(
                source_record,
                sim_info,
                dialog_prompt_path,
                rating_prompt_path,
            )
            for condition in args.conditions:
                key = (dialog_source, condition, pick_order)
                if key in completed:
                    skip_count += 1
                    continue
                try:
                    rating = score_condition(
                        condition,
                        messages,
                        official_client,
                        lab_client,
                        args,
                    )
                    result = {
                        "rescore_run_id": str(uuid.uuid4()),
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                        "ID": str(source_record.get("ID", "")),
                        "sample_pick_order": pick_order,
                        "dialog_source": dialog_source,
                        "source_run_id": source_record.get("run_id", ""),
                        "rating_condition": condition,
                        "simulated_user_model": args.simulated_user_model,
                        "temperature": args.temperature,
                        "max_tokens": args.max_tokens,
                        "message_count": len(messages),
                        "serialized_input_chars": len(serialize_messages_for_lab(messages)),
                        "rating": rating,
                    }
                    append_jsonl(args.result_jsonl, result)
                    completed.add(key)
                    success_count += 1
                    print(f"[OK] {pick_order} {dialog_source} {condition}")
                except Exception as exc:  # noqa: BLE001
                    error_count += 1
                    append_jsonl(
                        args.error_jsonl,
                        {
                            "created_at": datetime.now().isoformat(timespec="seconds"),
                            "ID": str(source_record.get("ID", "")),
                            "sample_pick_order": pick_order,
                            "dialog_source": dialog_source,
                            "rating_condition": condition,
                            "error": str(exc),
                        },
                    )
                    print(f"[ERR] {pick_order} {dialog_source} {condition}: {exc}")
                    if not args.continue_on_error:
                        write_summary(args.result_jsonl, args.summary_csv)
                        raise
                if args.sleep > 0:
                    time.sleep(args.sleep)

    write_summary(args.result_jsonl, args.summary_csv)
    print("\n完成。")
    print(f"成功: {success_count}")
    print(f"跳过: {skip_count}")
    print(f"错误: {error_count}")
    print(f"结果: {args.result_jsonl}")
    print(f"汇总: {args.summary_csv}")


def main() -> None:
    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT.parent / "数据生成代码" / ".env")
    run(parse_args())


if __name__ == "__main__":
    main()
