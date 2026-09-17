# -*- coding: utf-8 -*-
"""Diagnose Gemini mirror URL/key without printing the secret."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import requests

from run_langgraph_workflow import ROOT, load_dotenv


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="测试 Gemini 中转站 API Key 和认证方式")
    parser.add_argument("--base-url", default=os.getenv("GEMINI_BASE_URL", "https://api.aicodemirror.ai/api/gemini"))
    parser.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-2.5-pro"))
    parser.add_argument("--api-key", default=os.getenv("GEMINI_API_KEY", ""))
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--auth", choices=["header", "bearer", "query", "all"], default="all")
    args = parser.parse_args()

    key = args.api_key.strip()
    print(f"Base URL: {args.base_url.rstrip('/')}")
    print(f"Model: {args.model}")
    print(f"API Key loaded: {'yes' if key else 'no'}")
    print(f"API Key length: {len(key)}")
    if not key:
        raise SystemExit("未读取到 GEMINI_API_KEY；请将它写入工作流目录下的 .env")

    endpoint = f"{args.base_url.rstrip('/')}/v1beta/models/{args.model}:generateContent"
    body = {
        "contents": [{"role": "user", "parts": [{"text": "Reply with exactly API_OK"}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 32},
    }
    modes = [args.auth] if args.auth != "all" else ["header", "bearer", "query"]
    for mode in modes:
        headers = {"Content-Type": "application/json"}
        params = None
        if mode == "header":
            headers["x-goog-api-key"] = key
        elif mode == "bearer":
            headers["Authorization"] = f"Bearer {key}"
        else:
            params = {"key": key}
        try:
            response = requests.post(endpoint, headers=headers, params=params, json=body, timeout=args.timeout)
            print(f"\n[{mode}] HTTP {response.status_code}")
            print(response.text[:1000])
        except requests.RequestException as exc:
            print(f"\n[{mode}] request error: {exc}")


if __name__ == "__main__":
    main()
