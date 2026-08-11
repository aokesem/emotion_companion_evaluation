# -*- coding: utf-8 -*-
import json
import sys
import tempfile
import unittest
from pathlib import Path


WORKFLOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

import rescore_subjective_ratings as rescore  # noqa: E402


class RescoreSubjectiveRatingsTests(unittest.TestCase):
    def test_build_rating_messages_restores_native_roles(self):
        record = {
            "dialog_messages": [
                {"round": 1, "speaker": "simulated_user", "content": "opening"},
                {"round": 1, "speaker": "tested_agent", "content": "reply"},
                {"round": 2, "speaker": "simulated_user", "content": "follow-up"},
                {"round": 2, "speaker": "tested_agent", "content": "last reply"},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dialog_prompt = root / "dialog.txt"
            rating_prompt = root / "rating.txt"
            dialog_prompt.write_text("persona={sim_user_info_json}; opening={opening_sentence}", encoding="utf-8")
            rating_prompt.write_text("transcript={dialog_transcript}", encoding="utf-8")
            messages = rescore.build_rating_messages(
                record,
                {"ID": "1", "trait": "quiet"},
                dialog_prompt,
                rating_prompt,
            )

        self.assertEqual([message["role"] for message in messages], ["system", "assistant", "user", "assistant", "user", "user"])
        self.assertIn('"trait": "quiet"', messages[0]["content"])
        self.assertIn("opening=opening", messages[0]["content"])
        self.assertIn("用户: opening", messages[-1]["content"])
        self.assertIn("助手: last reply", messages[-1]["content"])

    def test_load_source_records_filters_provider(self):
        rows = [
            {"sample_pick_order": 10, "run_config": {"aux_provider": "official"}},
            {"sample_pick_order": 11, "run_config": {"aux_provider": "lab"}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            selected = rescore.load_source_records(path, "lab")
        self.assertEqual(list(selected), [11])

    def test_pair_sources_checks_ids(self):
        official = {172: {"ID": "001"}, 173: {"ID": "002"}}
        lab = {172: {"ID": "1"}, 173: {"ID": "2"}}
        pairs = rescore.pair_sources(official, lab)
        self.assertEqual([pair[0] for pair in pairs], [172, 173])

        lab[173] = {"ID": "different"}
        with self.assertRaisesRegex(ValueError, "ID 不一致"):
            rescore.pair_sources(official, lab)

    def test_validate_rating_rejects_out_of_range_score(self):
        rating = {key: 2 for key in rescore.RATING_KEYS}
        rescore.validate_rating(rating)
        rating[rescore.RATING_KEYS[0]] = 4
        with self.assertRaisesRegex(ValueError, "0 到 3"):
            rescore.validate_rating(rating)


if __name__ == "__main__":
    unittest.main()
