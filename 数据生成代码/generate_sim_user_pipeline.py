# -*- coding: utf-8 -*-
"""四步串行生成模拟用户信息的基础 pipeline。"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

CONFIG_PATH = Path(__file__).resolve().parent / "pipeline_config.json"


def load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    config = load_config(CONFIG_PATH)
    defaults = config["model_defaults"]
    default_input = Path(__file__).resolve().parent.parent / "用户画像数据" / "用户画像_抽样500_事实文本.csv"
    default_out_dir = Path(__file__).resolve().parent.parent / "用户画像数据"

    parser = argparse.ArgumentParser(description="生成模拟用户信息（4步串行）")
    parser.add_argument("--input-csv", type=Path, default=default_input, help="事实文本 CSV 输入路径")
    parser.add_argument("--output-dir", type=Path, default=default_out_dir, help="输出目录")
    parser.add_argument("--model", type=str, default=defaults["model"], help="模型名")
    parser.add_argument("--base-url", type=str, default=os.getenv("OPENAI_BASE_URL", ""), help="API Base URL")
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY", ""), help="API Key")
    parser.add_argument("--temperature", type=float, default=defaults["temperature"], help="采样温度")
    parser.add_argument("--max-tokens", type=int, default=defaults["max_tokens"], help="每步最大输出 token")
    parser.add_argument("--timeout", type=int, default=defaults["timeout"], help="请求超时秒数")
    parser.add_argument("--retries", type=int, default=defaults["retries"], help="失败重试次数")
    parser.add_argument("--sleep", type=float, default=defaults["sleep"], help="每条记录间隔秒数")
    parser.add_argument("--start", type=int, default=0, help="起始行索引（0-based）")
    parser.add_argument("--limit", type=int, default=50, help="处理条数，0 表示全部")
    parser.add_argument("--resume", action="store_true", help="兼容旧参数：默认已跳过已写入 JSONL 的 ID")
    parser.add_argument("--regenerate-existing", action="store_true", help="不跳过已写入 JSONL 的 ID，强制重新生成")
    parser.add_argument("--continue-on-error", action="store_true", help="遇错后继续下一条")
    parser.add_argument("--debug-http", action="store_true", help="输出接口响应诊断信息")
    return parser.parse_args()


def ensure_config(args: argparse.Namespace) -> None:
    if not args.base_url:
        raise ValueError("缺少 OPENAI_BASE_URL（或 --base-url）")
    if not args.api_key:
        raise ValueError("缺少 OPENAI_API_KEY（或 --api-key）")


def normalize_base_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


def row_text(row: pd.Series, col: str) -> str:
    value = row.get(col, "")
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text


def to_json_text(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def render_step_prompt(config: dict[str, Any], step_name: str, input_data: dict[str, Any]) -> str:
    template = config["steps"][step_name]["prompt_template"]
    return template.format(input_json=to_json_text(input_data))


def parse_json_content(content: str) -> dict[str, Any]:
    text = content.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        sub = text[start : end + 1]
        obj = json.loads(sub)
        if isinstance(obj, dict):
            return obj
    raise ValueError(f"模型返回非 JSON 对象: {text[:200]}")


def validate_keys(obj: dict[str, Any], required: list[str], step_name: str) -> None:
    missing = [k for k in required if k not in obj]
    if missing:
        raise ValueError(f"{step_name} 缺少字段: {missing}")


def extract_message_content(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text", "")))
        return "\n".join(chunks)
    return str(content)


class OpenAICompatClient:
    def __init__(self, base_url: str, api_key: str, timeout: int, retries: int, debug_http: bool = False):
        self.base_url = normalize_base_url(base_url)
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries
        self.debug_http = debug_http
        self.session = requests.Session()

    def chat_json(self, model: str, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_err: Exception | None = None
        disable_json_mode = False

        for attempt in range(1, self.retries + 1):
            req_payload = dict(payload)
            if disable_json_mode:
                req_payload.pop("response_format", None)
            try:
                resp = self.session.post(url, headers=headers, json=req_payload, timeout=self.timeout)
                if self.debug_http:
                    print(f"[HTTP] POST {url}")
                    print(f"[HTTP] status={resp.status_code} content-type={resp.headers.get('Content-Type', '')}")
                    print(f"[HTTP] body[:500]={resp.text[:500]}")
                if resp.status_code >= 400:
                    text = resp.text
                    if (not disable_json_mode) and ("response_format" in text.lower()):
                        disable_json_mode = True
                        continue
                    raise RuntimeError(f"HTTP {resp.status_code}: {text[:500]}")

                data = resp.json()
                message = data["choices"][0]["message"]
                content = extract_message_content(message)
                return parse_json_content(content)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < self.retries:
                    time.sleep(min(2 ** attempt, 8))
                else:
                    break

        raise RuntimeError(f"模型调用失败: {last_err}")


def build_step1_prompt(config: dict[str, Any], row: pd.Series) -> str:
    data = {
        "基础信息文本": row_text(row, "基础信息文本"),
        "教育信息文本": row_text(row, "教育信息文本"),
        "婚姻信息文本": row_text(row, "婚姻信息文本"),
        "情绪信息文本": row_text(row, "情绪信息文本"),
        "活动信息文本": row_text(row, "活动信息文本"),
        "上网信息文本": row_text(row, "上网信息文本"),
        "工作信息文本": row_text(row, "工作信息文本"),
    }
    return render_step_prompt(config, "step1", data)


def build_step2_prompt(config: dict[str, Any], row: pd.Series, step1: dict[str, Any]) -> str:
    data = {
        "基础信息文本": row_text(row, "基础信息文本"),
        "教育信息文本": row_text(row, "教育信息文本"),
        "婚姻信息文本": row_text(row, "婚姻信息文本"),
        "健康信息文本": row_text(row, "健康信息文本"),
        "工作信息文本": row_text(row, "工作信息文本"),
        "情绪信息文本": row_text(row, "情绪信息文本"),
        "活动信息文本": row_text(row, "活动信息文本"),
        "子女信息文本": row_text(row, "子女信息文本"),
        "上网信息文本": row_text(row, "上网信息文本"),
        "用户性格": step1,
    }
    return render_step_prompt(config, "step2", data)


def build_step3_prompt(config: dict[str, Any], row: pd.Series, step1: dict[str, Any], step2: dict[str, Any]) -> str:
    data = {
        "基础信息文本": row_text(row, "基础信息文本"),
        "教育信息文本": row_text(row, "教育信息文本"),
        "情绪信息文本": row_text(row, "情绪信息文本"),
        "婚姻信息文本": row_text(row, "婚姻信息文本"),
        "活动信息文本": row_text(row, "活动信息文本"),
        "健康信息文本": row_text(row, "健康信息文本"),
        "子女信息文本": row_text(row, "子女信息文本"),
        "上网信息文本": row_text(row, "上网信息文本"),
        "工作信息文本": row_text(row, "工作信息文本"),
        "用户性格": step1,
        "个人经历": step2,
    }
    return render_step_prompt(config, "step3", data)


def build_step4_prompt(config: dict[str, Any], step1: dict[str, Any], step2: dict[str, Any], step3: dict[str, Any]) -> str:
    data = {
        "用户性格": step1,
        "个人经历": step2,
        "生活烦恼": step3,
    }
    return render_step_prompt(config, "step4", data)


def load_processed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "ID" in obj:
                ids.add(str(obj["ID"]))
    return ids


def save_jsonl_row(path: Path, obj: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False))
        f.write("\n")


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    step1 = record["用户性格"]
    step2 = record["个人经历"]
    step3 = record["生活烦恼"]
    step4 = record["对话锚点"]

    return {
        "sample_pick_order": record["sample_pick_order"],
        "ID": record["ID"],
        "性格倾向": step1.get("性格倾向", step1.get("性格偏好", "")),
        "对话风格偏好": step1.get("对话风格偏好", ""),
        "成长背景": step2.get("成长背景", ""),
        "职业轨迹": step2.get("职业轨迹", ""),
        "健康影响": step2.get("健康影响", ""),
        "社交模式": step2.get("社交模式", ""),
        "家庭联系状态": step2.get("家庭联系状态", ""),
        "当前核心需求": step2.get("当前核心需求", ""),
        "主要烦恼": step3.get("主要烦恼", ""),
        "烦恼类别": step3.get("烦恼类别", ""),
        "表层话题": step3.get("表层话题", ""),
        "深层成因": step3.get("深层成因", step3.get("深层话题", "")),
        "情绪底色": step4.get("情绪底色", step4.get("情绪状态", "")),
        "最近触发事件": step4.get("最近触发事件", step4.get("触发事件", "")),
        "开场方式": step4.get("开场方式", ""),
        "开场首句": step4.get("开场首句", step4.get("开场首句示例", "")),
        "用户性格_json": json.dumps(step1, ensure_ascii=False),
        "个人经历_json": json.dumps(step2, ensure_ascii=False),
        "生活烦恼_json": json.dumps(step3, ensure_ascii=False),
        "对话锚点_json": json.dumps(step4, ensure_ascii=False),
    }


def run(args: argparse.Namespace) -> None:
    config = load_config(CONFIG_PATH)
    step1_keys = config["steps"]["step1"]["required_keys"]
    step2_keys = config["steps"]["step2"]["required_keys"]
    step3_keys = config["steps"]["step3"]["required_keys"]
    step4_keys = config["steps"]["step4"]["required_keys"]

    ensure_config(args)
    if not args.input_csv.exists():
        raise FileNotFoundError(f"输入文件不存在: {args.input_csv}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = args.output_dir / "模拟用户信息_500.jsonl"
    out_csv = args.output_dir / "模拟用户信息_500.csv"
    out_error = args.output_dir / "模拟用户信息_500_errors.jsonl"

    df = pd.read_csv(args.input_csv, encoding="utf-8-sig")
    total = len(df)
    end = total if args.limit <= 0 else min(total, args.start + args.limit)
    df = df.iloc[args.start:end].copy()

    processed_ids = set() if args.regenerate_existing else load_processed_ids(out_jsonl)
    seen_input_ids: set[str] = set()
    client = OpenAICompatClient(args.base_url, args.api_key, args.timeout, args.retries, args.debug_http)

    print(f"输入: {args.input_csv}")
    print(f"处理区间: [{args.start}, {end})，共 {len(df)} 条")
    print(f"输出 JSONL: {out_jsonl}")
    print(f"输出 CSV: {out_csv}")
    print(f"模型: {args.model}")

    done_count = 0
    existing_skip_count = 0
    duplicate_skip_count = 0
    err_count = 0

    for idx, row in df.iterrows():
        row_id = str(row.get("ID", ""))
        if row_id in seen_input_ids:
            duplicate_skip_count += 1
            print(f"[SKIP-DUP] idx={idx} ID={row_id}")
            continue
        seen_input_ids.add(row_id)

        if row_id in processed_ids:
            existing_skip_count += 1
            print(f"[SKIP-EXIST] idx={idx} ID={row_id}")
            continue

        try:
            s1 = client.chat_json(
                model=args.model,
                system_prompt=config["system_prompt"],
                user_prompt=build_step1_prompt(config, row),
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            validate_keys(s1, step1_keys, "步骤1")

            s2 = client.chat_json(
                model=args.model,
                system_prompt=config["system_prompt"],
                user_prompt=build_step2_prompt(config, row, s1),
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            validate_keys(s2, step2_keys, "步骤2")

            s3 = client.chat_json(
                model=args.model,
                system_prompt=config["system_prompt"],
                user_prompt=build_step3_prompt(config, row, s1, s2),
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            validate_keys(s3, step3_keys, "步骤3")

            s4 = client.chat_json(
                model=args.model,
                system_prompt=config["system_prompt"],
                user_prompt=build_step4_prompt(config, s1, s2, s3),
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            validate_keys(s4, step4_keys, "步骤4")

            record = {
                "sample_pick_order": int(float(row.get("sample_pick_order", idx + 1))),
                "ID": row_id,
                "用户性格": s1,
                "个人经历": s2,
                "生活烦恼": s3,
                "对话锚点": s4,
            }
            save_jsonl_row(out_jsonl, record)
            processed_ids.add(row_id)
            done_count += 1
            print(f"[OK] idx={idx} ID={row_id}")

            if args.sleep > 0:
                time.sleep(args.sleep)
        except Exception as exc:  # noqa: BLE001
            err_count += 1
            err_obj = {"index": int(idx), "ID": row_id, "error": str(exc)}
            save_jsonl_row(out_error, err_obj)
            print(f"[ERR] idx={idx} ID={row_id} -> {exc}")
            if not args.continue_on_error:
                raise

    # 汇总 CSV（包含历史 JSONL 全量）
    if out_jsonl.exists():
        records: list[dict[str, Any]] = []
        with out_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and "用户性格" in obj:
                    records.append(flatten_record(obj))
        if records:
            out_df = pd.DataFrame(records)
            out_df = out_df.sort_values(["sample_pick_order", "ID"], kind="stable")
            out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n完成。")
    print(f"新增成功: {done_count}")
    print(f"跳过(已有输出): {existing_skip_count}")
    print(f"跳过(重复ID): {duplicate_skip_count}")
    print(f"错误: {err_count}")


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent / ".env")
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
