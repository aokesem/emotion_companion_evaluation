# -*- coding: utf-8 -*-
"""Re-score fixed deepseek-v4-pro dialogs with deepseek-v4-flash."""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from run_langgraph_workflow import (
    OpenAICompatClient,
    ROOT,
    load_config,
    load_dotenv,
    load_sim_user_records,
    normalize_base_url,
    parse_json_content,
    resolve_path,
    transcript_text,
)

SCORES = ("识别层评分", "理解层评分", "行动层评分")


def parse_args() -> argparse.Namespace:
    config = load_config()
    defaults = config["defaults"]
    paths = config["paths"]
    parser = argparse.ArgumentParser(description="使用 deepseek-v4-flash 对固定对话执行评估 AI 评分")
    parser.add_argument("--input", type=Path, default=ROOT / "outputs" / "deepseek_v4_pro" / "dialog_eval_runs.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "deepseek_v4_pro-evaluator-flash-20")
    parser.add_argument("--sim-user-jsonl", type=Path, default=resolve_path(paths["sim_user_jsonl"]))
    parser.add_argument("--prompt", type=Path, default=resolve_path(paths["evaluator_rating_prompt"]))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY") or os.getenv("deepseek_v4_pro_KEY", ""))
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=defaults["temperature"])
    parser.add_argument("--max-tokens", type=int, default=defaults["max_tokens"])
    parser.add_argument("--timeout", type=int, default=defaults["timeout"])
    parser.add_argument("--retries", type=int, default=defaults["retries"])
    parser.add_argument("--sleep", type=float, default=defaults["sleep"])
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--regenerate-existing", action="store_true")
    parser.add_argument("--debug-http", action="store_true")
    args = parser.parse_args()
    args.input = args.input if args.input.is_absolute() else (ROOT / args.input).resolve()
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else (ROOT / "outputs" / args.output_dir).resolve()
    args.sim_user_jsonl = args.sim_user_jsonl if args.sim_user_jsonl.is_absolute() else (ROOT / args.sim_user_jsonl).resolve()
    args.prompt = args.prompt if args.prompt.is_absolute() else (ROOT / args.prompt).resolve()
    return args


def read_records(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    rows.sort(key=lambda row: int(float(row["sample_pick_order"])))
    return rows[:limit]


def positive_info(sim_info: dict[str, Any]) -> dict[str, Any]:
    personality = sim_info.get("用户性格", {})
    return {
        "用户性格": personality.get("性格倾向", ""),
        "对话风格偏好": personality.get("对话风格偏好", ""),
        "生活烦恼": sim_info.get("生活烦恼", {}),
        "当前核心需求": sim_info.get("个人经历", {}).get("当前核心需求", ""),
    }


def build_prompt(record: dict[str, Any], sim_info: dict[str, Any], prompt_path: Path) -> str:
    template = prompt_path.read_text(encoding="utf-8")
    values = {
        "positive_info_json": json.dumps(positive_info(sim_info), ensure_ascii=False, indent=2),
        "dialog_transcript": transcript_text(record.get("dialog_messages", [])),
        "dialog_messages_json": json.dumps(record.get("dialog_messages", []), ensure_ascii=False, indent=2),
        "tested_agent_summary_json": json.dumps(record.get("tested_agent_summary", {}), ensure_ascii=False, indent=2),
    }
    for key, value in values.items():
        template = template.replace("{" + key + "}", value)
    return template


def validate_rating(rating: dict[str, Any]) -> None:
    missing = [key for key in SCORES if key not in rating]
    if missing:
        raise ValueError(f"评分缺少字段: {missing}")
    for key in SCORES:
        value = rating[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= float(value) <= 2:
            raise ValueError(f"{key} 必须是 0 到 2 的数值: {value!r}")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    load_dotenv(ROOT / ".env")
    args = parse_args()
    records = read_records(args.input, args.limit)
    sim_records = load_sim_user_records(args.sim_user_jsonl)
    if not args.api_key:
        raise ValueError("缺少 API key，请设置 OPENAI_API_KEY 或 deepseek_v4_pro_KEY")
    client = OpenAICompatClient(
        normalize_base_url(args.base_url), args.api_key, args.timeout, args.retries, args.debug_http,
        auto_append_v1=True,
    )
    result_path = args.output_dir / "evaluator_flash_runs.jsonl"
    error_path = args.output_dir / "evaluator_flash_errors.jsonl"
    existing = set()
    if result_path.exists() and not args.regenerate_existing:
        for line in result_path.read_text(encoding="utf-8").splitlines():
            try:
                existing.add(int(float(json.loads(line)["sample_pick_order"])))
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                pass
    print(f"输入: {args.input}")
    print(f"模型: {args.model}")
    print(f"样本数: {len(records)}")
    print(f"输出目录: {args.output_dir}")
    for record in records:
        order = int(float(record["sample_pick_order"]))
        if order in existing:
            print(f"[SKIP] sample_pick_order={order}")
            continue
        try:
            sim_info = sim_records.get(str(record.get("ID", "")).lstrip("0"))
            if sim_info is None:
                raise ValueError(f"缺少模拟用户信息: ID={record.get('ID')}")
            prompt = build_prompt(record, sim_info, args.prompt)
            rating = client.chat_json(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            validate_rating(rating)
            append_jsonl(result_path, {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "sample_pick_order": order,
                "ID": record.get("ID", ""),
                "source_run_id": record.get("run_id", ""),
                "model": args.model,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "evaluator_rating": rating,
            })
            print(f"[OK] sample_pick_order={order}")
        except Exception as exc:  # noqa: BLE001
            append_jsonl(error_path, {"created_at": datetime.now().isoformat(timespec="seconds"), "sample_pick_order": order, "ID": record.get("ID", ""), "error": str(exc)})
            print(f"[ERR] sample_pick_order={order}: {exc}")
            if not args.continue_on_error:
                raise
        if args.sleep > 0:
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
