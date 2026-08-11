# -*- coding: utf-8 -*-
import argparse
import sys
import unittest
from pathlib import Path


WORKFLOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

import run_langgraph_workflow as workflow  # noqa: E402


class HybridLabModeTests(unittest.TestCase):
    def test_lab_flag_resolves_role_specific_providers(self):
        args = argparse.Namespace(lab=True, manual=False, aux_provider="official")
        workflow.apply_provider_mode(args)
        self.assertEqual(args.provider_mode, "lab-hybrid")
        self.assertEqual(args.simulated_user_provider, "lab")
        self.assertEqual(args.subjective_rating_provider, "official")
        self.assertEqual(args.evaluator_provider, "lab")

    def test_existing_aux_provider_lab_keeps_all_lab_behavior(self):
        args = argparse.Namespace(lab=False, manual=False, aux_provider="lab")
        workflow.apply_provider_mode(args)
        self.assertEqual(args.provider_mode, "lab")
        self.assertEqual(args.simulated_user_provider, "lab")
        self.assertEqual(args.subjective_rating_provider, "lab")
        self.assertEqual(args.evaluator_provider, "lab")

    def test_hybrid_clients_use_official_only_for_subjective_rating(self):
        args = argparse.Namespace(
            manual=False,
            simulated_user_provider="lab",
            subjective_rating_provider="official",
            evaluator_provider="lab",
            simulated_user_base_url="https://official.example/v1",
            simulated_user_api_key="official-user-key",
            evaluator_base_url="",
            evaluator_api_key="",
            tested_agent_base_url="https://tested.example/v1",
            tested_agent_api_key="tested-key",
            base_url="",
            api_key="",
            tested_agent_auto_append_v1=False,
            tested_agent_chat_completions_path="/chat/completions",
            lab_api_url="https://lab.example/agent/qa",
            lab_simulated_user_token="lab-user-token",
            lab_evaluator_token="lab-evaluator-token",
            timeout=90,
            retries=1,
            debug_http=False,
        )
        clients = workflow.create_clients(args)
        self.assertIsInstance(clients["simulated_user"], workflow.LabAgentClient)
        self.assertIsInstance(clients["subjective_rating"], workflow.OpenAICompatClient)
        self.assertIsInstance(clients["tested_agent"], workflow.OpenAICompatClient)
        self.assertIsInstance(clients["evaluator"], workflow.LabAgentClient)


if __name__ == "__main__":
    unittest.main()
