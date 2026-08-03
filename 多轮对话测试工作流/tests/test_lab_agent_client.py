# -*- coding: utf-8 -*-
import sys
import unittest
from pathlib import Path


WORKFLOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

import run_langgraph_workflow as workflow  # noqa: E402


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self):
        return self.payload


class RecordingSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, headers, json, timeout):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return self.responses.pop(0)


class LabAgentClientTests(unittest.TestCase):
    def make_client(self, responses):
        client = workflow.LabAgentClient("https://lab.example/agent/qa", "secret-token", 90, 1)
        client.session = RecordingSession(responses)
        return client

    def test_chat_text_serializes_messages_and_uses_stateless_context(self):
        messages = [
            {"role": "system", "content": "persona"},
            {"role": "assistant", "content": "opening"},
            {"role": "user", "content": "reply"},
        ]
        client = self.make_client(
            [FakeResponse({"data": {"answer": " first "}}), FakeResponse({"data": {"answer": "second"}})]
        )

        self.assertEqual(client.chat_text("ignored", messages, 0.2, 10000), "first")
        self.assertEqual(client.chat_text("ignored", messages, 0.2, 10000), "second")

        first, second = client.session.calls
        self.assertEqual(first["headers"]["Authorization"], "Token secret-token")
        sent_messages = first["json"]["messages"]
        self.assertEqual(len(sent_messages), 1)
        self.assertEqual(sent_messages[0]["role"], "user")
        serialized = sent_messages[0]["content"]
        self.assertIn("系统消息（最高优先级）", serialized)
        self.assertIn("persona", serialized)
        self.assertIn("opening", serialized)
        self.assertIn("reply", serialized)
        self.assertEqual(first["json"]["inputs"], {})
        self.assertFalse(first["json"]["stream"])
        self.assertEqual(first["timeout"], 90)
        self.assertLessEqual(len(first["json"]["end_user"]), 32)
        self.assertNotEqual(first["json"]["context_id"], second["json"]["context_id"])
        self.assertNotEqual(first["json"]["end_user"], second["json"]["end_user"])

    def test_chat_json_reads_answer(self):
        client = self.make_client([FakeResponse({"data": {"answer": '{"score": 2}'}})])

        result = client.chat_json("ignored", [{"role": "user", "content": "score"}], 0.2, 10000)

        self.assertEqual(result, {"score": 2})

    def test_missing_answer_is_reported(self):
        client = self.make_client([FakeResponse({"data": {}})])

        with self.assertRaisesRegex(RuntimeError, "data.answer"):
            client.chat_text("ignored", [{"role": "user", "content": "hello"}], 0.2, 10000)

    def test_leading_think_block_is_removed(self):
        client = self.make_client([FakeResponse({"data": {"answer": "<think>internal reasoning</think> final answer"}})])

        result = client.chat_text("ignored", [{"role": "user", "content": "hello"}], 0.2, 10000)

        self.assertEqual(result, "final answer")

    def test_workflow_failure_message_is_reported(self):
        client = self.make_client([FakeResponse({"data": {"answer": "工作流执行失败，请稍后重试"}})])

        with self.assertRaisesRegex(RuntimeError, "工作流执行失败"):
            client.chat_text("ignored", [{"role": "user", "content": "hello"}], 0.2, 10000)


if __name__ == "__main__":
    unittest.main()
