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


class WorkflowState(TypedDict, total=False):
    run_id: str
    created_at: str
    case_id: str
    sample_pick_order: int
    turn: int
    max_turns: int
    user_profile: str
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
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    config = load_config()
    defaults = config["defaults"]
    paths = config["paths"]

    parser = argparse.ArgumentParser(description="运行 LangGraph 多轮对话测试工作流")
    parser.add_argument("--fact-csv", type=Path, default=resolve_path(paths["fact_csv"]), help="用户画像事实文本 CSV")
    parser.add_argument("--sim-user-jsonl", type=Path, default=resolve_path(paths["sim_user_jsonl"]), help="模拟用户信息 JSONL")
    parser.add_argument("--success-jsonl", type=Path, default=resolve_path(paths["success_jsonl"]), help="成功结果 JSONL")
    parser.add_argument("--summary-csv", type=Path, default=resolve_path(paths["summary_csv"]), help="评分摘要 CSV")
    parser.add_argument("--error-jsonl", type=Path, default=resolve_path(paths["error_jsonl"]), help="错误记录 JSONL")
    parser.add_argument("--ids", type=str, default="", help="逗号分隔的 ID 列表；为空时按 start/limit 选择")
    parser.add_argument("--start", type=int, default=defaults["start"], help="起始行索引，0-based")
    parser.add_argument("--limit", type=int, default=defaults["limit"], help="处理条数；0 表示全部")
    parser.add_argument("--turns", type=int, default=defaults["turns"], help="对话轮数；一轮为用户一句 + 被测 AI 一句")
    parser.add_argument("--simulated-user-model", type=str, default=defaults["simulated_user_model"], help="模拟用户模型名")
    parser.add_argument("--tested-agent-model", type=str, default=defaults["tested_agent_model"], help="被测 AI 模型名")
    parser.add_argument("--evaluator-model", type=str, default=defaults["evaluator_model"], help="评估 AI 模型名")
    parser.add_argument("--base-url", type=str, default=os.getenv("OPENAI_BASE_URL", ""), help="API Base URL")
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY", ""), help="API Key")
    parser.add_argument("--temperature", type=float, default=defaults["temperature"], help="采样温度")
    parser.add_argument("--max-tokens", type=int, default=defaults["max_tokens"], help="每次回复最大 token")
    parser.add_argument("--timeout", type=int, default=defaults["timeout"], help="请求超时秒数")
    parser.add_argument("--retries", type=int, default=defaults["retries"], help="失败重试次数")
    parser.add_argument("--sleep", type=float, default=defaults["sleep"], help="每个样本之间间隔秒数")
    parser.add_argument("--continue-on-error", action="store_true", help="遇错后继续下一个样本")
    parser.add_argument("--dry-run", action="store_true", help="只检查数据和配置，不调用模型")
    parser.add_argument("--regenerate-existing", action="store_true", help="不跳过已成功记录的 ID")
    parser.add_argument("--debug-http", action="store_true", help="输出接口响应诊断信息")
    return parser.parse_args()


def normalize_base_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if not base.endswith("/v1"):
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
    def __init__(self, base_url: str, api_key: str, timeout: int, retries: int, debug_http: bool = False):
        self.base_url = normalize_base_url(base_url)
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries
        self.debug_http = debug_http
        self.session = requests.Session()

    def chat_text(self, model: str, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> str:
        return self._chat(model, messages, temperature, max_tokens).strip()

    def chat_json(self, model: str, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> dict[str, Any]:
        return parse_json_content(self._chat(model, messages, temperature, max_tokens))

    def _chat(self, model: str, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> str:
        url = f"{self.base_url}/chat/completions"
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
    client: OpenAICompatClient | None

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


def load_success_ids(path: Path) -> set[str]:
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
            row_id = str(obj.get("ID", "")).lstrip("0")
            if row_id:
                ids.add(row_id)
    return ids


def row_text(row: pd.Series, col: str) -> str:
    value = row.get(col, "")
    if pd.isna(value):
        return ""
    return str(value).strip()


def select_fact_rows(df: pd.DataFrame, ids: str, start: int, limit: int) -> pd.DataFrame:
    if ids.strip():
        wanted = {part.strip().lstrip("0") for part in ids.split(",") if part.strip()}
        return df[df["ID"].astype(str).str.lstrip("0").isin(wanted)].copy()
    end = len(df) if limit <= 0 else min(len(df), start + limit)
    return df.iloc[start:end].copy()


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


def append_summary_csv(path: Path, state: WorkflowState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    row = {
        "run_id": state["run_id"],
        "ID": state["case_id"],
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
    if ctx.client is None:
        raise ValueError("dry-run 模式不需要构建可执行 graph")
    client = ctx.client
    args = ctx.args

    def user_opening(state: WorkflowState) -> WorkflowState:
        sim_info = state["sim_user_info"]
        user_profile = state["user_profile"]
        opening = get_opening_sentence(sim_info)
        if not opening:
            raise ValueError(f"ID={state['case_id']} 缺少开场首句")

        sim_system = ctx.prompt(
            "simulated_user_dialog",
            user_profile=user_profile,
            sim_user_info_json=json.dumps(sim_info, ensure_ascii=False, indent=2),
            opening_sentence=opening,
        )
        tested_system = ctx.prompt("tested_agent_dialog", user_profile=user_profile)

        return {
            "turn": 1,
            "dialog_messages": [{"round": 1, "speaker": "simulated_user", "content": opening}],
            "simulated_user_messages": [
                {"role": "system", "content": sim_system},
                {"role": "assistant", "content": opening},
            ],
            "tested_agent_messages": [
                {"role": "system", "content": tested_system},
                {"role": "user", "content": opening},
            ],
        }

    def tested_agent_reply(state: WorkflowState) -> WorkflowState:
        reply = client.chat_text(
            model=args.tested_agent_model,
            messages=state["tested_agent_messages"],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        turn = state["turn"]
        return {
            "dialog_messages": state["dialog_messages"]
            + [{"round": turn, "speaker": "tested_agent", "content": reply}],
            "tested_agent_messages": state["tested_agent_messages"] + [{"role": "assistant", "content": reply}],
            "simulated_user_messages": state["simulated_user_messages"] + [{"role": "user", "content": reply}],
        }

    def route_after_tested_reply(state: WorkflowState) -> Literal["continue_dialog", "finish_dialog"]:
        return "finish_dialog" if state["turn"] >= state["max_turns"] else "continue_dialog"

    def simulated_user_reply(state: WorkflowState) -> WorkflowState:
        reply = client.chat_text(
            model=args.simulated_user_model,
            messages=state["simulated_user_messages"],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        next_turn = state["turn"] + 1
        return {
            "turn": next_turn,
            "dialog_messages": state["dialog_messages"]
            + [{"round": next_turn, "speaker": "simulated_user", "content": reply}],
            "simulated_user_messages": state["simulated_user_messages"] + [{"role": "assistant", "content": reply}],
            "tested_agent_messages": state["tested_agent_messages"] + [{"role": "user", "content": reply}],
        }

    def simulated_user_rating(state: WorkflowState) -> WorkflowState:
        prompt = ctx.prompt(
            "simulated_user_rating",
            dialog_transcript=transcript_text(state["dialog_messages"]),
            dialog_messages_json=json.dumps(state["dialog_messages"], ensure_ascii=False, indent=2),
        )
        rating = client.chat_json(
            model=args.simulated_user_model,
            messages=state["simulated_user_messages"] + [{"role": "user", "content": prompt}],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        return {"simulated_user_rating": rating}

    def tested_agent_summary(state: WorkflowState) -> WorkflowState:
        prompt = ctx.prompt(
            "tested_agent_summary",
            dialog_transcript=transcript_text(state["dialog_messages"]),
            dialog_messages_json=json.dumps(state["dialog_messages"], ensure_ascii=False, indent=2),
        )
        summary = client.chat_json(
            model=args.tested_agent_model,
            messages=state["tested_agent_messages"] + [{"role": "user", "content": prompt}],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        return {"tested_agent_summary": summary}

    def evaluator_rating(state: WorkflowState) -> WorkflowState:
        sim_info = state["sim_user_info"]
        positive = {
            "生活烦恼": sim_info.get("生活烦恼", {}),
            "当前核心需求": sim_info.get("个人经历", {}).get("当前核心需求", ""),
            "对话锚点": sim_info.get("对话锚点", {}),
        }
        prompt = ctx.prompt(
            "evaluator_rating",
            positive_info_json=json.dumps(positive, ensure_ascii=False, indent=2),
            dialog_transcript=transcript_text(state["dialog_messages"]),
            dialog_messages_json=json.dumps(state["dialog_messages"], ensure_ascii=False, indent=2),
            tested_agent_summary_json=json.dumps(state.get("tested_agent_summary", {}), ensure_ascii=False, indent=2),
        )
        rating = client.chat_json(
            model=args.evaluator_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        return {"evaluator_rating": rating}

    def save_success(state: WorkflowState) -> WorkflowState:
        result = dict(state)
        save_jsonl(args.success_jsonl, result)
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
    graph.add_edge("simulated_user_rating", "tested_agent_summary")
    graph.add_edge("tested_agent_summary", "evaluator_rating")
    graph.add_edge("evaluator_rating", "save_success")
    graph.add_edge("save_success", END)
    return graph.compile()


def ensure_files(args: argparse.Namespace, prompt_paths: dict[str, Path]) -> None:
    if args.turns <= 0:
        raise ValueError("--turns 必须为正整数")
    for path in [args.fact_csv, args.sim_user_jsonl, *prompt_paths.values()]:
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
    if not args.dry_run:
        if not args.base_url:
            raise ValueError("缺少 OPENAI_BASE_URL（或 --base-url）")
        if not args.api_key:
            raise ValueError("缺少 OPENAI_API_KEY（或 --api-key）")


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
    user_profile = row_text(row, "画像事实文本")
    if not user_profile:
        raise ValueError(f"ID={row_id} 缺少画像事实文本")
    return {
        "run_id": str(uuid.uuid4()),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "case_id": row_id,
        "sample_pick_order": int(float(row.get("sample_pick_order", 0))),
        "turn": 0,
        "max_turns": turns,
        "user_profile": user_profile,
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
    selected = select_fact_rows(fact_df, args.ids, args.start, args.limit)
    success_ids = set() if args.regenerate_existing else load_success_ids(args.success_jsonl)

    print(f"用户画像: {args.fact_csv}")
    print(f"模拟用户信息: {args.sim_user_jsonl}")
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
                f"[DRY] ID={row.get('ID', '')} "
                f"has_sim_info={row_id in sim_records} "
                f"already_success={row_id in success_ids}"
            )
        return

    ctx = WorkflowContext(
        args=args,
        prompt_paths=prompt_paths,
        client=OpenAICompatClient(args.base_url, args.api_key, args.timeout, args.retries, args.debug_http),
    )
    app = build_graph(ctx)

    done_count = 0
    skip_count = 0
    error_count = 0
    for _, row in selected.iterrows():
        row_id = str(row.get("ID", "")).lstrip("0")
        if row_id in success_ids:
            skip_count += 1
            print(f"[SKIP] ID={row.get('ID', '')} 已有成功记录")
            continue
        sim_info = sim_records.get(row_id)
        if sim_info is None:
            error_count += 1
            err = {"ID": str(row.get("ID", "")), "error": "缺少模拟用户信息", "created_at": datetime.now().isoformat(timespec="seconds")}
            save_jsonl(args.error_jsonl, err)
            print(f"[ERR] ID={row.get('ID', '')} 缺少模拟用户信息")
            if not args.continue_on_error:
                raise ValueError(err["error"])
            continue
        try:
            app.invoke(initial_state(row, sim_info, args.turns))
            success_ids.add(row_id)
            done_count += 1
            print(f"[OK] ID={row.get('ID', '')}")
            if args.sleep > 0:
                time.sleep(args.sleep)
        except Exception as exc:  # noqa: BLE001
            error_count += 1
            err = {
                "ID": str(row.get("ID", "")),
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
