import unittest
from unittest import mock

from gpt_vln.cost import SessionCostLedger
from gpt_vln.model import navigate_cli


class SessionCostLedgerTest(unittest.TestCase):
    def test_resumed_session_uses_increments(self):
        ledger = SessionCostLedger()
        self.assertAlmostEqual(ledger.add({'session_id': 'nav', 'cost_usd': .23}, 'a'), .23)
        self.assertAlmostEqual(ledger.add({'session_id': 'nav', 'cost_usd': .41}, 'b'), .18)
        self.assertAlmostEqual(ledger.add({'session_id': 'nav', 'cost_usd': .41}, 'b'), 0)
        self.assertAlmostEqual(ledger.add({'session_id': 'review', 'cost_usd': .12}, 'c'), .12)

    def test_uncosted_response_does_not_change_total(self):
        ledger = SessionCostLedger()
        self.assertEqual(ledger.add({'error': 'provider limit'}, 'a'), 0)
        self.assertEqual(ledger.add({'cost_usd': None}, 'b'), 0)

    def test_claude_prompt_keeps_review_without_repeating_schema(self):
        payload = {'provider': 'claude', 'context': {
            'pending_route_review': {'route': 'contradicted'}},
            'frames': [], 'memory_catalog': {}}
        with mock.patch('gpt_vln.claude.complete', return_value={'output': {}}) as call:
            navigate_cli(payload)
        prompt = call.call_args.args[0]
        self.assertIn('pending_route_review', prompt)
        self.assertIn('contradicted', prompt)
        self.assertNotIn('"history_panorama_ids":{"type":', prompt)


if __name__ == '__main__':
    unittest.main()
