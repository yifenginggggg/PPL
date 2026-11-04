import unittest

from profile_eval.blackbox import (
    _fair_relation_scores,
    build_final_request,
    evaluate_final,
)
from profile_eval.fact_pipeline import _derive_history_withdrawals, _is_withdrawal_request


def _source():
    return {
        "fact_id": "S1", "origin": "source", "stage": 1,
        "profile_id": None, "unit_id": None,
        "text": "I like spicy food.", "kind": "preference", "owner": "user",
        "evidence_strength": "explicit", "importance": "required", "state": "active",
        "assertion_status": "current", "condition_type": "none", "condition": None,
        "evidence": [{"kind": "source", "sid": "u1", "stage": 1,
                      "quote": "I like spicy food."}],
        "links": [], "uncertainty": None,
    }


def _profile(*, stage=8):
    return {
        "fact_id": "P1", "origin": "profile", "stage": stage,
        "profile_id": "final", "unit_id": "final/C1",
        "text": "I like spicy food.", "kind": "preference", "owner": "user",
        "evidence_strength": "explicit", "importance": "required", "state": "active",
        "assertion_status": "current", "condition_type": "none", "condition": None,
        "evidence": [{"kind": "profile", "profile_id": "final",
                      "unit_id": "final/C1", "cid": 1,
                      "span": "I like spicy food."}],
        "links": [{"fact_id": "S1", "relation": "equivalent"}],
        "uncertainty": None,
    }


class BlackBoxTests(unittest.TestCase):
    def test_request_has_no_filesystem_contract(self):
        request = build_final_request(
            history=[{"id": "u1", "text": "I like spicy food.", "stage": 1}],
            profile={"id": "final", "stage": 8, "text": "I like spicy food."},
        )
        self.assertEqual(request["profile_mode"], "final")
        self.assertEqual(request["profiles"][0]["profile_id"], "final")
        self.assertNotIn("source_dir", request)
        self.assertNotIn("profile_dir", request)

    def test_unspecified_stages_inherit_the_current_stage(self):
        request = build_final_request(
            history=[
                {"id": "u1", "text": "I like spicy food.", "stage": 2},
                {"id": "u2", "text": "I also like tea."},
                {"id": "u3", "text": "I play tennis.", "stage": 3},
            ],
            profile={"id": "final", "stage": 8, "text": "I like spicy food."},
        )
        self.assertEqual(
            [row["stage"] for row in request["history"]["messages"]],
            [2, 2, 3],
        )

    def test_evaluate_final_runs_fact_materials_tasks_and_scores(self):
        graph = {
            "source_blocks": [_source()],
            "profile_blocks": [_profile()],
            "events": [],
            "coverage": {"profiles": {"final": {
                "unit_status": {"final/C1": "covered"}
            }}},
            "uncertainty": [],
        }

        calls = []
        prompts = []

        def fake_llm(*, system_prompt, payload, model_id):
            self.assertIn("source_blocks" if payload["stage"] == "history_source" else "profile_blocks", system_prompt)
            calls.append(payload["stage"])
            prompts.append(system_prompt)
            if payload["stage"] == "history_source":
                self.assertNotIn("profile", payload)
                self.assertNotIn("source_blocks", payload)
                return {"source_blocks": graph["source_blocks"]}
            self.assertEqual(payload["stage"], "profile_projection")
            self.assertIn("profile", payload)
            self.assertEqual(payload["source_blocks"][0]["fact_id"], "S1")
            return {
                "profile_blocks": graph["profile_blocks"],
                "coverage": graph["coverage"],
                "uncertainty": [],
            }

        result = evaluate_final(
            history=[{"id": "u1", "text": "I like spicy food.", "stage": 1}],
            profile={"id": "final", "stage": 8, "text": "I like spicy food."},
            llm=fake_llm,
        )
        self.assertEqual(calls, ["history_source", "profile_projection"])
        self.assertEqual(len(set(prompts)), 2)
        self.assertEqual(result["validation"]["source_blocks_n"], 1)
        self.assertEqual(result["materials"]["profile_claims"][0]["profile_id"], "final")
        self.assertEqual(set(result["materials"]), {"source_catalogue", "profile_claims", "uncertain"})
        self.assertEqual(result["scores"]["completeness"]["rate"], 1.0)
        self.assertEqual(result["scores"]["support"]["rate"], 1.0)
        self.assertEqual(set(result["tasks"]), {"completeness", "support", "reliability"})
        self.assertEqual(set(result["scores"]["primary"]), {"completeness", "support", "reliability"})
        self.assertEqual(result["scores"]["reliability"]["rate"], 1.0)
        self.assertNotIn("condition", result["scores"])
        self.assertNotIn("attribution", result["scores"])
        self.assertNotIn("withdrawal", result["scores"])

    def test_reliability_penalizes_unresolved_profile_claims(self):
        graph = {
            "source_blocks": [_source()],
            "profile_blocks": [{**_profile(), "assertion_status": "uncertain"}],
            "events": [],
            "coverage": {"profiles": {"final": {
                "unit_status": {"final/C1": "covered"}
            }}},
            "uncertainty": [],
        }

        def fake_llm(*, payload, **_):
            if payload["stage"] == "history_source":
                return {"source_blocks": graph["source_blocks"]}
            return {
                "profile_blocks": graph["profile_blocks"],
                "coverage": graph["coverage"],
                "uncertainty": [],
            }

        result = evaluate_final(
            history=[{"id": "u1", "text": "I like spicy food.", "stage": 1}],
            profile={"id": "final", "stage": 8, "text": "I like spicy food."},
            llm=fake_llm,
        )
        reliability = result["scores"]["reliability"]
        self.assertEqual(reliability["eligible_n"], 1)
        self.assertEqual(reliability["pass_n"], 0)
        self.assertEqual(reliability["rate"], 0.0)

    def test_profile_stage_is_input_data_not_an_s8_gate(self):
        graph = {
            "source_blocks": [_source()],
            "profile_blocks": [{**_profile(stage=3), "stage": 3}],
            "events": [],
            "coverage": {"profiles": {"final": {
                "unit_status": {"final/C1": "covered"}
            }}},
            "uncertainty": [],
        }
        def fake_llm(*, payload, **_):
            if payload["stage"] == "history_source":
                return {"source_blocks": graph["source_blocks"]}
            return {
                "profile_blocks": graph["profile_blocks"],
                "coverage": graph["coverage"],
                "uncertainty": [],
            }

        result = evaluate_final(
            history=[{"id": "u1", "text": "I like spicy food.", "stage": 1}],
            profile={"id": "final", "stage": 3, "text": "I like spicy food."},
            llm=fake_llm,
        )
        self.assertEqual(result["request"]["profiles"][0]["stage"], 3)
        self.assertEqual(result["scores"]["completeness"]["rate"], 1.0)

    def test_invalid_source_evidence_downgrades_link_to_uncertain(self):
        graph = {
            "source_blocks": [{**_source(), "evidence": []}],
            "profile_blocks": [_profile()],
            "events": [],
            "coverage": {"profiles": {"final": {
                "unit_status": {"final/C1": "covered"}
            }}},
            "uncertainty": [],
        }

        def fake_llm(*, payload, **_):
            if payload["stage"] == "history_source":
                return {"source_blocks": [{**graph["source_blocks"][0],
                                             "evidence": [{"kind": "source", "sid": "u1", "stage": 1, "quote": "wrong"}]}]}
            return {
                "profile_blocks": graph["profile_blocks"],
                "coverage": graph["coverage"],
                "uncertainty": [],
            }

        result = evaluate_final(
            history=[{"id": "u1", "text": "I like spicy food.", "stage": 1}],
            profile={"id": "final", "stage": 8, "text": "I like spicy food."},
            llm=fake_llm,
        )
        self.assertEqual(result["fact_graph"]["profile_blocks"][0]["links"][0]["relation"], "uncertain")
        self.assertEqual(result["scores"]["reliability"]["rate"], 0.0)

    def test_evaluate_batch_shares_history_source_call(self):
        from profile_eval.blackbox import evaluate_batch

        calls = []
        source_hashes = []

        def fake_llm(*, payload, **_):
            calls.append(payload["stage"])
            if payload["stage"] == "history_source":
                self.assertNotIn("profile", payload)
                return {"source_blocks": [_source()]}
            profile_id = payload["profile"]["profile_id"]
            unit_id = f"{profile_id}/C1"
            source_hashes.append(payload["source_catalogue_hash"])
            return {
                "profile_blocks": [{
                    **_profile(),
                    "fact_id": f"{profile_id}/P1",
                    "profile_id": profile_id,
                    "unit_id": unit_id,
                    "evidence": [{
                        "kind": "profile", "profile_id": profile_id,
                        "unit_id": unit_id, "cid": 1,
                        "span": "I like spicy food.",
                    }],
                }],
                "coverage": {"profiles": {profile_id: {
                    "unit_status": {unit_id: "covered"}
                }}},
                "uncertainty": [],
            }

        result = evaluate_batch(
            history=[{"id": "u1", "text": "I like spicy food.", "stage": 1}],
            profiles=[
                {"id": "teacher", "stage": 8, "text": "I like spicy food."},
                {"id": "student", "stage": 8, "text": "I like spicy food."},
            ],
            llm=fake_llm,
        )
        self.assertEqual(calls, ["history_source", "profile_projection", "profile_projection"])
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(source_hashes[0], source_hashes[1])
        self.assertTrue(all(row["provider_metadata"]["shared_history"]
                            for row in result["results"]))
        self.assertEqual(
            result["source_catalogue"]["source_blocks"][0]["fact_id"], "S1")


    def test_support_is_zero_not_missing_when_nothing_is_linked(self):
        fair = _fair_relation_scores({
            "source_blocks": [_source()],
            "profile_blocks": [{**_profile(), "links": []}],
        })
        self.assertEqual(fair["source_recall"], 0.0)
        self.assertEqual(fair["profile_precision"], 0.0)
        self.assertEqual(fair["grounded_f1"], 0.0)

    def test_withdrawal_prefers_an_active_source_fact(self):
        source = {
            **_source(),
            "fact_id": "S1",
            "text": "I like coffee.",
            "evidence": [{"kind": "source", "sid": "u1", "stage": 1,
                          "quote": "I like coffee."}],
        }
        withdrawn = {
            **source,
            "fact_id": "S2",
            "stage": 2,
            "state": "withdrawn",
        }
        events = _derive_history_withdrawals(
            {"source_blocks": [source, withdrawn], "events": []},
            {"history": {"messages": [{
                "sid": "u2",
                "stage": 2,
                "text": "Please forget that I like coffee.",
            }]}},
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["target_fact_id"], "S1")

    def test_withdrawal_request_ignores_negated_forget(self):
        self.assertTrue(_is_withdrawal_request("Please forget that I like coffee."))
        self.assertFalse(_is_withdrawal_request("I will never forget that day."))
        self.assertFalse(_is_withdrawal_request("Please don't forget that I like coffee."))


if __name__ == "__main__":
    unittest.main()

