# -*- coding: utf-8 -*-
"""LangGraph workflow for simulated-user multi-turn dialog evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypedDict

import pandas as pd
import requests
from langgraph.graph import END, StateGraph

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "workflow_config.json"
TESTED_PROFILES_PATH = ROOT / "tested_model_profiles.json"


class WorkflowState(TypedDict, total=False):
    run_id: str
    created_at: str
    ID: str
    sample_pick_order: int
    turn: int
    max_turns: int
    sim_user_info: dict[str, Any]
    dialog_messages: list[dict[str, Any]]
    simulated_user_messages: list[dict[str, str]]
    tested_agent_messages: list[dict[str, str]]
    tested_agent_summary: dict[str, Any]
    simulated_user_rating: dict[str, Any]
    evaluator_rating: dict[str, Any]
    error: str


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


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_tested_profiles() -> dict[str, dict[str, Any]]:
    if not TESTED_PROFILES_PATH.exists():
        return {}
    with TESTED_PROFILES_PATH.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"被测模型 profile 配置必须是 JSON 对象: {TESTED_PROFILES_PATH}")
    return data


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def resolve_output_dir(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "outputs":
        return (ROOT / path).resolve()
    return (ROOT / "outputs" / path).resolve()


def output_file_path(path_arg: Path | None, config_value: str, output_dir: Path) -> Path:
    if path_arg is not None:
        return resolve_path(str(path_arg))
    if "{output_dir}" in config_value:
        return Path(config_value.format(output_dir=str(output_dir))).resolve()
    config_path = Path(config_value)
    if config_path.is_absolute():
        return config_path
    return output_dir / config_path.name

def env_value(env_names: Any) -> str:
    if isinstance(env_names, list):
        names = [str(name).strip() for name in env_names]
    else:
        names = [str(env_names).strip()]
    for name in names:
        if name and os.getenv(name):
            return os.getenv(name, "")
    return ""


def parse_args() -> argparse.Namespace:
    config = load_config()
    defaults = config["defaults"]
    paths = config["paths"]

    parser = argparse.ArgumentParser(description="运行 LangGraph 多轮对话测试工作流")
    parser.add_argument("--fact-csv", type=Path, default=resolve_path(paths["fact_csv"]), help="用户画像事实文本 CSV")
    parser.add_argument("--sim-user-jsonl", type=Path, default=resolve_path(paths["sim_user_jsonl"]), help="模拟用户信息 JSONL")
    parser.add_argument("--output-dir", type=str, default="", help="输出实验目录；相对路径默认位于 outputs/ 下")
    parser.add_argument("--success-jsonl", type=Path, default=None, help="成功结果 JSONL；默认由 output-dir 派生")
    parser.add_argument("--summary-csv", type=Path, default=None, help="评分摘要 CSV；默认由 output-dir 派生")
    parser.add_argument("--error-jsonl", type=Path, default=None, help="错误记录 JSONL；默认由 output-dir 派生")
    parser.add_argument("--ids", type=str, default="", help="逗号分隔的 ID 列表；为空时按 start/limit 选择")
    parser.add_argument("--start", type=int, default=defaults["start"], help="起始行索引，0-based")
    parser.add_argument("--limit", type=int, default=defaults["limit"], help="处理条数；0 表示全部")
    parser.add_argument("--turns", type=int, default=defaults["turns"], help="对话轮数；一轮为用户一句 + 被测 AI 一句")
    parser.add_argument("--simulated-user-model", type=str, default=defaults["simulated_user_model"], help="模拟用户模型名")
    parser.add_argument("--tested-profile", type=str, default=defaults.get("tested_profile", ""), help="被测模型 profile 名，读取 tested_model_profiles.json")
    parser.add_argument("--tested-agent-model", type=str, default=None, help="被测 AI 模型名；可覆盖 tested-profile 中的模型名")
    parser.add_argument("--evaluator-model", type=str, default=defaults["evaluator_model"], help="评估 AI 模型名")
    parser.add_argument("--base-url", type=str, default=os.getenv("OPENAI_BASE_URL", ""), help="通用 API Base URL，未设置角色专用 URL 时使用")
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY", ""), help="通用 API Key，未设置角色专用 Key 时使用")
    parser.add_argument("--simulated-user-base-url", type=str, default=os.getenv("SIMULATED_USER_BASE_URL", ""), help="模拟用户 API Base URL")
    parser.add_argument("--simulated-user-api-key", type=str, default=os.getenv("SIMULATED_USER_API_KEY", ""), help="模拟用户 API Key")
    parser.add_argument("--tested-agent-base-url", type=str, default=os.getenv("TESTED_AGENT_BASE_URL", ""), help="被测 AI API Base URL")
    parser.add_argument("--tested-agent-api-key", type=str, default=os.getenv("TESTED_AGENT_API_KEY", ""), help="被测 AI API Key")
    parser.add_argument("--evaluator-base-url", type=str, default=os.getenv("EVALUATOR_BASE_URL", ""), help="评估 AI API Base URL")
    parser.add_argument("--evaluator-api-key", type=str, default=os.getenv("EVALUATOR_API_KEY", ""), help="评估 AI API Key")
    parser.add_argument("--temperature", type=float, default=defaults["temperature"], help="采样温度")
    parser.add_argument("--max-tokens", type=int, default=defaults["max_tokens"], help="每次回复最大 token")
    parser.add_argument("--timeout", type=int, default=defaults["timeout"], help="请求超时秒数")
    parser.add_argument("--retries", type=int, default=defaults["retries"], help="失败重试次数")
    parser.add_argument("--sleep", type=float, default=defaults["sleep"], help="每个样本之间间隔秒数")
    parser.add_argument("--continue-on-error", action="store_true", help="遇错后继续下一个样本")
    parser.add_argument("--dry-run", action="store_true", help="只检查数据和配置，不调用模型")
    parser.add_argument("--print-dialog", action="store_true", help="运行时实时打印用户 AI 与被测 AI 的对话内容")
    parser.add_argument("--manual", action="store_true", help="手动输入被测 AI 每轮回复；仅生成用户 AI 主观评分")
    parser.add_argument("--regenerate-existing", action="store_true", help="不跳过已成功记录的 ID")
    parser.add_argument("--debug-http", action="store_true", help="输出接口响应诊断信息")
    args = parser.parse_args()

    apply_tested_profile(args, defaults)

    output_dir_text = args.output_dir or ("manual" if args.manual else args.profile_output_dir or paths.get("output_dir", "outputs"))
    args.output_dir = resolve_output_dir(output_dir_text)
    args.success_jsonl = output_file_path(args.success_jsonl, paths["success_jsonl"], args.output_dir)
    args.summary_csv = output_file_path(args.summary_csv, paths["summary_csv"], args.output_dir)
    args.error_jsonl = output_file_path(args.error_jsonl, paths["error_jsonl"], args.output_dir)
    return args


def apply_tested_profile(args: argparse.Namespace, defaults: dict[str, Any]) -> None:
    args.profile_output_dir = ""
    args.tested_agent_auto_append_v1 = True
    args.tested_agent_chat_completions_path = "/chat/completions"

    if args.tested_profile:
        profiles = load_tested_profiles()
        profile = profiles.get(args.tested_profile)
        if profile is None:
            available = ", ".join(sorted(profiles)) or "无"
            raise ValueError(f"未找到被测模型 profile: {args.tested_profile}。可用 profile: {available}")
        if not isinstance(profile, dict):
            raise ValueError(f"被测模型 profile 必须是对象: {args.tested_profile}")

        base_url = str(profile.get("base_url", "")).strip()
        base_url_env = profile.get("base_url_env", "")
        if base_url_env:
            base_url = env_value(base_url_env)
        args.tested_agent_base_url = base_url

        api_key = str(profile.get("api_key", "")).strip()
        api_key_env = profile.get("api_key_env", "")
        if api_key_env:
            api_key = env_value(api_key_env)
        args.tested_agent_api_key = api_key
        if args.tested_agent_model is None:
            args.tested_agent_model = str(profile.get("model", "")).strip()
        args.profile_output_dir = str(profile.get("output_dir", args.tested_profile)).strip()
        args.tested_agent_auto_append_v1 = bool(profile.get("auto_append_v1", True))
        args.tested_agent_chat_completions_path = str(profile.get("chat_completions_path", "/chat/completions")).strip()
    elif args.tested_agent_model is None:
        args.tested_agent_model = defaults.get("tested_agent_model", "")


def normalize_base_url(base_url: str, auto_append_v1: bool = True) -> str:
    base = base_url.rstrip("/")
    if auto_append_v1 and not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


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
        obj = json.loads(text[start : end + 1])
        if isinstance(obj, dict):
            return obj
    raise ValueError(f"模型返回非 JSON 对象: {text[:200]}")


class OpenAICompatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: int,
        retries: int,
        debug_http: bool = False,
        auto_append_v1: bool = True,
        chat_completions_path: str = "/chat/completions",
    ):
        self.base_url = normalize_base_url(base_url, auto_append_v1)
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries
        self.debug_http = debug_http
        self.chat_completions_path = "/" + chat_completions_path.strip("/")
        self.session = requests.Session()

    def chat_text(self, model: str, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> str:
        return self._chat(model, messages, temperature, max_tokens).strip()

    def chat_json(self, model: str, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> dict[str, Any]:
        content = self._chat(model, messages, temperature, max_tokens)
        try:
            return parse_json_content(content)
        except Exception as first_error:  # noqa: BLE001
            repaired = self.repair_json(model, content, temperature, max_tokens)
            try:
                return parse_json_content(repaired)
            except Exception as repair_error:  # noqa: BLE001
                raw_preview = content.strip().replace("\n", " ")[:500]
                repair_preview = repaired.strip().replace("\n", " ")[:500]
                raise ValueError(
                    "模型返回 JSON 解析失败，且自动修复失败。"
                    f"首次错误: {first_error}; 修复错误: {repair_error}; "
                    f"原始输出片段: {raw_preview}; 修复输出片段: {repair_preview}"
                ) from repair_error

    def repair_json(self, model: str, content: str, temperature: float, max_tokens: int) -> str:
        repair_messages = [
            {
                "role": "system",
                "content": "你是 JSON 修复器。请只输出一个合法 JSON 对象，不要输出 markdown，不要解释，不要增删字段含义。",
            },
            {
                "role": "user",
                "content": "下面内容本应是 JSON 对象，但格式可能有错误。请修复为合法 JSON 对象，保留原始语义：\n" + content,
            },
        ]
        return self._chat(model, repair_messages, min(temperature, 0.1), max_tokens)

    def _chat(self, model: str, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> str:
        url = f"{self.base_url}{self.chat_completions_path}"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_err: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                resp = self.session.post(url, headers=headers, json=payload, timeout=self.timeout)
                if self.debug_http:
                    print(f"[HTTP] POST {url}")
                    print(f"[HTTP] status={resp.status_code} body[:500]={resp.text[:500]}")
                if resp.status_code >= 400:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
                data = resp.json()
                return extract_message_content(data["choices"][0]["message"])
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < self.retries:
                    time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"模型调用失败: {last_err}")


@dataclass
class WorkflowContext:
    args: argparse.Namespace
    prompt_paths: dict[str, Path]
    simulated_user_client: OpenAICompatClient | None
    tested_agent_client: OpenAICompatClient | None
    evaluator_client: OpenAICompatClient | None

    def prompt(self, name: str, **kwargs: str) -> str:
        path = self.prompt_paths[name]
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"提示词文件为空: {path}")
        for key, value in kwargs.items():
            text = text.replace("{" + key + "}", value)
        return text


def load_sim_user_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            row_id = str(obj.get("ID", "")).lstrip("0")
            if row_id and row_id not in records:
                records[row_id] = obj
    return records


def load_success_tasks(path: Path) -> tuple[set[int], set[str]]:
    """Load completed draws, with ID fallback only for legacy records."""
    if not path.exists():
        return set(), set()
    pick_orders: set[int] = set()
    legacy_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            pick_order = obj.get("sample_pick_order")
            if pick_order not in (None, ""):
                try:
                    pick_orders.add(int(float(pick_order)))
                    continue
                except (TypeError, ValueError):
                    pass
            row_id = normalized_id(obj.get("ID", ""))
            if row_id:
                legacy_ids.add(row_id)
    return pick_orders, legacy_ids

def normalized_id(value: object) -> str:
    return str(value).strip().lstrip("0")


def select_fact_rows(
    df: pd.DataFrame,
    ids: str,
    start: int,
    limit: int,
    success_pick_orders: set[int],
    legacy_success_ids: set[str],
    sim_records: dict[str, dict[str, Any]],
    allow_existing: bool,
) -> pd.DataFrame:
    if ids.strip():
        wanted = {part.strip().lstrip("0") for part in ids.split(",") if part.strip()}
        return df[df["ID"].astype(str).str.lstrip("0").isin(wanted)].copy()

    rows: list[pd.Series] = []
    for _, row in df.iloc[start:].iterrows():
        row_id = normalized_id(row.get("ID", ""))
        pick_order = int(float(row.get("sample_pick_order", 0)))
        if not row_id:
            continue
        if (not allow_existing) and (pick_order in success_pick_orders or row_id in legacy_success_ids):
            continue
        if row_id not in sim_records:
            continue
        rows.append(row)
        if limit > 0 and len(rows) >= limit:
            break
    return pd.DataFrame(rows, columns=df.columns)

def get_opening_sentence(sim_info: dict[str, Any]) -> str:
    anchor = sim_info.get("对话锚点", {})
    if isinstance(anchor, dict):
        opening = str(anchor.get("开场首句", "")).strip()
        if opening:
            return opening
    return str(sim_info.get("开场首句", "")).strip()


def transcript_text(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for msg in messages:
        speaker = "用户" if msg["speaker"] == "simulated_user" else "助手"
        lines.append(f"{speaker}: {msg['content']}")
    return "\n".join(lines)


def save_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False))
        f.write("\n")


def print_dialog_line(row_id: str, round_no: int, speaker: str, content: str) -> None:
    print(f"\n[{row_id}][第 {round_no} 轮][{speaker}]")
    print(content.strip())


def read_manual_reply(row_id: str, turn: int) -> str:
    while True:
        reply = input(f"\n[{row_id}][第 {turn} 轮][请输入被测AI回复] > ").strip()
        if reply:
            return reply
        print("回复不能为空，请重新输入。")


def success_result(state: WorkflowState, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "run_id": state["run_id"],
        "created_at": state["created_at"],
        "ID": state["ID"],
        "sample_pick_order": state.get("sample_pick_order", ""),
        "turns": state["max_turns"],
        "run_config": {
            "tested_agent_mode": "manual" if args.manual else "openai",
            "tested_profile": args.tested_profile if not args.manual else "",
            "simulated_user_model": args.simulated_user_model,
            "tested_agent_model": "manual" if args.manual else args.tested_agent_model,
            "evaluator_model": "" if args.manual else args.evaluator_model,
        },
        "dialog_messages": state.get("dialog_messages", []),
        "simulated_user_rating": state.get("simulated_user_rating", {}),
        "tested_agent_summary": state.get("tested_agent_summary", {}),
        "evaluator_rating": state.get("evaluator_rating", {}),
    }


def append_summary_csv(path: Path, state: WorkflowState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    row = {
        "run_id": state["run_id"],
        "ID": state["ID"],
        "sample_pick_order": state.get("sample_pick_order", ""),
        "turns": state["max_turns"],
        "simulated_user_rating_json": json.dumps(state.get("simulated_user_rating", {}), ensure_ascii=False),
        "tested_agent_summary_json": json.dumps(state.get("tested_agent_summary", {}), ensure_ascii=False),
        "evaluator_rating_json": json.dumps(state.get("evaluator_rating", {}), ensure_ascii=False),
    }
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def build_graph(ctx: WorkflowContext):
    if ctx.simulated_user_client is None:
        raise ValueError("dry-run 模式不需要构建可执行 graph")
    simulated_user_client = ctx.simulated_user_client
    tested_agent_client = ctx.tested_agent_client
    evaluator_client = ctx.evaluator_client
    args = ctx.args

    def user_opening(state: WorkflowState) -> WorkflowState:
        sim_info = state["sim_user_info"]
        opening = get_opening_sentence(sim_info)
        if not opening:
            raise ValueError(f"ID={state['ID']} 缺少开场首句")

        sim_system = ctx.prompt(
            "simulated_user_dialog",
            sim_user_info_json=json.dumps(sim_info, ensure_ascii=False, indent=2),
            opening_sentence=opening,
        )
        tested_system = "" if args.manual else ctx.prompt("tested_agent_dialog")

        if args.print_dialog or args.manual:
            print_dialog_line(state["ID"], 1, "用户AI", opening)

        next_state: WorkflowState = {
            "turn": 1,
            "dialog_messages": [{"round": 1, "speaker": "simulated_user", "content": opening}],
            "simulated_user_messages": [
                {"role": "system", "content": sim_system},
                {"role": "assistant", "content": opening},
            ],
            "tested_agent_messages": [],
        }
        if not args.manual:
            next_state["tested_agent_messages"] = [
                {"role": "system", "content": tested_system},
                {"role": "user", "content": opening},
            ]
        return next_state

    def tested_agent_reply(state: WorkflowState) -> WorkflowState:
        turn = state["turn"]
        if args.manual:
            reply = read_manual_reply(state["ID"], turn)
        else:
            if tested_agent_client is None:
                raise ValueError("缺少被测 AI 客户端")
            reply = tested_agent_client.chat_text(
                model=args.tested_agent_model,
                messages=state["tested_agent_messages"],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        if args.print_dialog and not args.manual:
            print_dialog_line(state["ID"], turn, "被测AI", reply)
        next_state: WorkflowState = {
            "dialog_messages": state["dialog_messages"]
            + [{"round": turn, "speaker": "tested_agent", "content": reply}],
            "simulated_user_messages": state["simulated_user_messages"] + [{"role": "user", "content": reply}],
        }
        if not args.manual:
            next_state["tested_agent_messages"] = state["tested_agent_messages"] + [{"role": "assistant", "content": reply}]
        return next_state

    def route_after_tested_reply(state: WorkflowState) -> Literal["continue_dialog", "finish_dialog"]:
        return "finish_dialog" if state["turn"] >= state["max_turns"] else "continue_dialog"

    def simulated_user_reply(state: WorkflowState) -> WorkflowState:
        reply = simulated_user_client.chat_text(
            model=args.simulated_user_model,
            messages=state["simulated_user_messages"],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        next_turn = state["turn"] + 1
        if args.print_dialog or args.manual:
            print_dialog_line(state["ID"], next_turn, "用户AI", reply)
        next_state: WorkflowState = {
            "turn": next_turn,
            "dialog_messages": state["dialog_messages"]
            + [{"round": next_turn, "speaker": "simulated_user", "content": reply}],
            "simulated_user_messages": state["simulated_user_messages"] + [{"role": "assistant", "content": reply}],
        }
        if not args.manual:
            next_state["tested_agent_messages"] = state["tested_agent_messages"] + [{"role": "user", "content": reply}]
        return next_state

    def simulated_user_rating(state: WorkflowState) -> WorkflowState:
        prompt = ctx.prompt(
            "simulated_user_rating",
            dialog_transcript=transcript_text(state["dialog_messages"]),
            dialog_messages_json=json.dumps(state["dialog_messages"], ensure_ascii=False, indent=2),
        )
        rating = simulated_user_client.chat_json(
            model=args.simulated_user_model,
            messages=state["simulated_user_messages"] + [{"role": "user", "content": prompt}],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        return {"simulated_user_rating": rating}

    def tested_agent_summary(state: WorkflowState) -> WorkflowState:
        if args.manual:
            return {"tested_agent_summary": {}}
        if tested_agent_client is None:
            raise ValueError("缺少被测 AI 客户端")
        prompt = ctx.prompt("tested_agent_summary")
        summary = tested_agent_client.chat_json(
            model=args.tested_agent_model,
            messages=state["tested_agent_messages"] + [{"role": "user", "content": prompt}],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        return {"tested_agent_summary": summary}

    def evaluator_rating(state: WorkflowState) -> WorkflowState:
        if args.manual:
            return {"evaluator_rating": {}}
        if evaluator_client is None:
            raise ValueError("缺少评估 AI 客户端")
        sim_info = state["sim_user_info"]
        user_personality = sim_info.get("用户性格", {})
        positive = {
            "用户性格": user_personality.get("性格倾向", ""),
            "对话风格偏好": user_personality.get("对话风格偏好", ""),
            "生活烦恼": sim_info.get("生活烦恼", {}),
            "当前核心需求": sim_info.get("个人经历", {}).get("当前核心需求", ""),
        }
        prompt = ctx.prompt(
            "evaluator_rating",
            positive_info_json=json.dumps(positive, ensure_ascii=False, indent=2),
            dialog_transcript=transcript_text(state["dialog_messages"]),
            dialog_messages_json=json.dumps(state["dialog_messages"], ensure_ascii=False, indent=2),
            tested_agent_summary_json=json.dumps(state.get("tested_agent_summary", {}), ensure_ascii=False, indent=2),
        )
        rating = evaluator_client.chat_json(
            model=args.evaluator_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        return {"evaluator_rating": rating}

    def save_success(state: WorkflowState) -> WorkflowState:
        save_jsonl(args.success_jsonl, success_result(state, args))
        append_summary_csv(args.summary_csv, state)
        return state

    graph = StateGraph(WorkflowState)
    graph.add_node("user_opening", user_opening)
    graph.add_node("tested_agent_reply", tested_agent_reply)
    graph.add_node("simulated_user_reply", simulated_user_reply)
    graph.add_node("simulated_user_rating", simulated_user_rating)
    graph.add_node("tested_agent_summary", tested_agent_summary)
    graph.add_node("evaluator_rating", evaluator_rating)
    graph.add_node("save_success", save_success)

    graph.set_entry_point("user_opening")
    graph.add_edge("user_opening", "tested_agent_reply")
    graph.add_conditional_edges(
        "tested_agent_reply",
        route_after_tested_reply,
        {
            "continue_dialog": "simulated_user_reply",
            "finish_dialog": "simulated_user_rating",
        },
    )
    graph.add_edge("simulated_user_reply", "tested_agent_reply")
    if args.manual:
        graph.add_edge("simulated_user_rating", "save_success")
    else:
        graph.add_edge("simulated_user_rating", "tested_agent_summary")
        graph.add_edge("tested_agent_summary", "evaluator_rating")
        graph.add_edge("evaluator_rating", "save_success")
    graph.add_edge("save_success", END)
    return graph.compile()


def ensure_files(args: argparse.Namespace, prompt_paths: dict[str, Path]) -> None:
    if args.turns <= 0:
        raise ValueError("--turns 必须为正整数")
    required_prompts = [prompt_paths["simulated_user_dialog"], prompt_paths["simulated_user_rating"]]
    if not args.manual:
        required_prompts.extend(
            [
                prompt_paths["tested_agent_dialog"],
                prompt_paths["tested_agent_summary"],
                prompt_paths["evaluator_rating"],
            ]
        )
    for path in [args.fact_csv, args.sim_user_jsonl, *required_prompts]:
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
    if not args.dry_run:
        role_credentials(args)


def role_credentials(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    credentials = {
        "simulated_user": {
            "base_url": args.simulated_user_base_url or args.base_url,
            "api_key": args.simulated_user_api_key or args.api_key,
            "auto_append_v1": True,
            "chat_completions_path": "/chat/completions",
        }
    }
    if not args.manual:
        credentials.update(
            {
                "tested_agent": {
                    "base_url": args.tested_agent_base_url or args.base_url,
                    "api_key": args.tested_agent_api_key or args.api_key,
                    "auto_append_v1": args.tested_agent_auto_append_v1,
                    "chat_completions_path": args.tested_agent_chat_completions_path,
                },
                "evaluator": {
                    "base_url": args.evaluator_base_url or args.base_url,
                    "api_key": args.evaluator_api_key or args.api_key,
                    "auto_append_v1": True,
                    "chat_completions_path": "/chat/completions",
                },
            }
        )
    missing = [name for name, config in credentials.items() if not config["base_url"] or not config["api_key"]]
    if missing:
        raise ValueError(
            "缺少以下角色的 API 配置: "
            + ", ".join(missing)
            + "。可设置角色专用参数/环境变量，或设置通用 OPENAI_BASE_URL 与 OPENAI_API_KEY。"
        )
    return credentials


def create_clients(args: argparse.Namespace) -> dict[str, OpenAICompatClient]:
    credentials = role_credentials(args)
    return {
        name: OpenAICompatClient(
            config["base_url"],
            config["api_key"],
            args.timeout,
            args.retries,
            args.debug_http,
            config["auto_append_v1"],
            config["chat_completions_path"],
        )
        for name, config in credentials.items()
    }


def prompt_paths_from_config() -> dict[str, Path]:
    paths = load_config()["paths"]
    return {
        "simulated_user_dialog": resolve_path(paths["simulated_user_dialog_prompt"]),
        "tested_agent_dialog": resolve_path(paths["tested_agent_dialog_prompt"]),
        "simulated_user_rating": resolve_path(paths["simulated_user_rating_prompt"]),
        "tested_agent_summary": resolve_path(paths["tested_agent_summary_prompt"]),
        "evaluator_rating": resolve_path(paths["evaluator_rating_prompt"]),
    }


def initial_state(row: pd.Series, sim_info: dict[str, Any], turns: int) -> WorkflowState:
    row_id = str(row.get("ID", ""))
    return {
        "run_id": str(uuid.uuid4()),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "ID": row_id,
        "sample_pick_order": int(float(row.get("sample_pick_order", 0))),
        "turn": 0,
        "max_turns": turns,
        "sim_user_info": sim_info,
        "dialog_messages": [],
        "simulated_user_messages": [],
        "tested_agent_messages": [],
    }


def run(args: argparse.Namespace) -> None:
    prompt_paths = prompt_paths_from_config()
    ensure_files(args, prompt_paths)

    fact_df = pd.read_csv(args.fact_csv, encoding="utf-8-sig")
    sim_records = load_sim_user_records(args.sim_user_jsonl)
    success_pick_orders, legacy_success_ids = (set(), set()) if args.regenerate_existing else load_success_tasks(args.success_jsonl)
    explicit_ids = bool(args.ids.strip())
    selected = select_fact_rows(
        df=fact_df,
        ids=args.ids,
        start=args.start,
        limit=args.limit,
        success_pick_orders=success_pick_orders,
        legacy_success_ids=legacy_success_ids,
        sim_records=sim_records,
        allow_existing=args.regenerate_existing or explicit_ids,
    )

    print(f"用户画像: {args.fact_csv}")
    print(f"模拟用户信息: {args.sim_user_jsonl}")
    print(f"输出目录: {args.output_dir}")
    print(f"被测模式: {'manual' if args.manual else 'openai'}")
    if not args.manual:
        print(f"被测 profile: {args.tested_profile or '未指定'}")
        print(f"被测模型: {args.tested_agent_model}")
    print(f"候选样本: {len(selected)}")
    print(f"对话轮数: {args.turns}")
    print(f"成功输出: {args.success_jsonl}")
    print(f"摘要输出: {args.summary_csv}")
    print(f"错误输出: {args.error_jsonl}")

    if args.dry_run:
        for name, path in prompt_paths.items():
            status = "非空" if path.read_text(encoding="utf-8").strip() else "空"
            print(f"[PROMPT] {name}: {status} ({path})")
        for _, row in selected.iterrows():
            row_id = str(row.get("ID", "")).lstrip("0")
            print(
                f"[DRY] sample_pick_order={row.get('sample_pick_order', '')} ID={row.get('ID', '')} "
                f"has_sim_info={row_id in sim_records} "
                f"already_success={int(float(row.get('sample_pick_order', 0))) in success_pick_orders or row_id in legacy_success_ids}"
            )
        return

    clients = create_clients(args)
    ctx = WorkflowContext(
        args=args,
        prompt_paths=prompt_paths,
        simulated_user_client=clients["simulated_user"],
        tested_agent_client=clients.get("tested_agent"),
        evaluator_client=clients.get("evaluator"),
    )
    app = build_graph(ctx)

    done_count = 0
    skip_count = 0
    error_count = 0
    for _, row in selected.iterrows():
        row_id = normalized_id(row.get("ID", ""))
        pick_order = int(float(row.get("sample_pick_order", 0)))
        if (not explicit_ids) and (not args.regenerate_existing) and (pick_order in success_pick_orders or row_id in legacy_success_ids):
            skip_count += 1
            print(f"[SKIP] sample_pick_order={pick_order} ID={row.get('ID', '')} 已有成功记录")
            continue
        sim_info = sim_records.get(row_id)
        if sim_info is None:
            error_count += 1
            err = {"ID": str(row.get("ID", "")), "sample_pick_order": pick_order, "error": "缺少模拟用户信息", "created_at": datetime.now().isoformat(timespec="seconds")}
            save_jsonl(args.error_jsonl, err)
            print(f"[ERR] ID={row.get('ID', '')} 缺少模拟用户信息")
            if not args.continue_on_error:
                raise ValueError(err["error"])
            continue
        try:
            app.invoke(initial_state(row, sim_info, args.turns))
            success_pick_orders.add(pick_order)
            done_count += 1
            print(f"[OK] sample_pick_order={pick_order} ID={row.get('ID', '')}")
            if args.sleep > 0:
                time.sleep(args.sleep)
        except Exception as exc:  # noqa: BLE001
            error_count += 1
            err = {
                "ID": str(row.get("ID", "")),
                "sample_pick_order": pick_order,
                "error": str(exc),
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            save_jsonl(args.error_jsonl, err)
            print(f"[ERR] ID={row.get('ID', '')} -> {exc}")
            if not args.continue_on_error:
                raise

    print("\n完成。")
    print(f"成功: {done_count}")
    print(f"跳过: {skip_count}")
    print(f"错误: {error_count}")


def main() -> None:
    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT.parent / "数据生成代码" / ".env")
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()











