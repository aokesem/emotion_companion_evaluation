# -*- coding: utf-8 -*-
import sys
import unittest
from pathlib import Path

WORKFLOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

import run_langgraph_workflow as workflow  # noqa: E402


class GeminiClientTests(unittest.TestCase):
    def test_build_payload_converts_roles_and_parameters(self):
        payload = workflow.GeminiClient.build_payload(
            [
                {"role": "system", "content": "Follow the persona."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "How are you?"},
            ], 0.2, 10000
        )
        self.assertEqual(payload["systemInstruction"]["parts"], [{"text": "Follow the persona."}])
        self.assertEqual([item["role"] for item in payload["contents"]], ["user", "model", "user"])
        self.assertEqual(payload["generationConfig"]["maxOutputTokens"], 10000)

    def test_extract_text_ignores_thought_parts(self):
        data = {"candidates": [{"content": {"parts": [{"thought": True, "text": "hidden"}, {"text": "answer"}]}}]}
        self.assertEqual(workflow.GeminiClient.extract_text(data), "answer")


if __name__ == "__main__":
    unittest.main()
