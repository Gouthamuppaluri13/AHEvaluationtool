import os
import unittest

class TestVCAHE(unittest.TestCase):
    def test_single_score_keys_and_ranges(self):
        from metrics import compute_vc_ahe_score
        out = compute_vc_ahe_score(
            prompt="Explain your market thesis.",
            response="AI dominates VC funding in 2025; ARR is $1.2M with strong margins.",
            projected_cf=[1_200_000, 1_600_000, 2_000_000]
        )
        # Expected keys exist
        for k in [
            "base_score","rouge_l","bert_f","uncertainty",
            "trend_relevance","math_score","finance_score",
            "vc_ahe_score","bias_risk","valuation_risk"
        ]:
            self.assertIn(k, out)
        # Ranged values
        for k in ["base_score","rouge_l","bert_f","uncertainty","trend_relevance","math_score","finance_score","vc_ahe_score"]:
            self.assertGreaterEqual(out[k], 0.0)
            self.assertLessEqual(out[k], 1.0)
        self.assertIn(out["bias_risk"], ["low","medium","high"])
        self.assertIn(out["valuation_risk"], ["low","medium","high"])

    def test_batch_interface(self):
        from metrics import compute_vc_ahe_score_batch
        prompts = ["p1","p2"]
        resps = ["AI and IPO","nothing here"]
        outs = compute_vc_ahe_score_batch(prompts, resps)
        self.assertEqual(len(outs), 2)
        self.assertIn("vc_ahe_score", outs[0])

if __name__ == "__main__":
    unittest.main()
