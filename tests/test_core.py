import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from agentnav.actions import ActionId, parse_chunk, to_primitives, describe
from agentnav.agent import _outcome
from agentnav.config import AgentConfig, HistoryConfig, SelectorConfig
from agentnav.controller import forward_clearance_m
from agentnav.history import HistoryManager, uniform_indices
from agentnav.mapping import FREE, OCCUPIED, UNKNOWN, LocalMapper, depth_to_meters, occlusion_depth
from agentnav.selector import PROMPT_TEMPLATE, Decision, build_prompt, load_codex_credentials, _response_text
from agentnav.types import CameraIntrinsics, Waypoint

CM = (25.0, 100.0)
DEG = (15.0, 90.0)


class ActionChunkTest(unittest.TestCase):
    def chunk(self, text):
        return parse_chunk(text, CM, DEG)

    def test_two_sub_actions(self):
        self.assertEqual(describe(self.chunk("turn left 30, forward 75")), "turn left 30, forward 75")

    def test_quantisation_to_primitives(self):
        # 0.25 m step, 15 deg turn: 30 deg -> 2 turns, 75 cm -> 3 forwards.
        self.assertEqual(to_primitives(self.chunk("turn left 30, forward 75"), 0.25, 15.0), [2, 2, 1, 1, 1])

    def test_chunk_length_is_unbounded(self):
        text = "forward 50, turn left 30, forward 100, turn right 45, forward 25"
        self.assertEqual(len(self.chunk(text)), 5)

    def test_magnitudes_are_clamped(self):
        self.assertEqual(self.chunk("forward 500")[0].forward_cm, 100.0)
        self.assertEqual(self.chunk("turn left 5")[0].turn_deg, 15.0)

    def test_answer_tags_are_unwrapped(self):
        self.assertEqual(describe(self.chunk("<answer>turn right 45</answer>")), "turn right 45")

    def test_prose_around_the_command_is_tolerated(self):
        self.assertEqual(describe(self.chunk("I will turn right 60 degrees")), "turn right 60")

    def test_stop_truncates_the_chunk(self):
        chunk = self.chunk("forward 75, stop, forward 100")
        self.assertEqual(len(chunk), 2)
        self.assertEqual(chunk[-1].action, ActionId.STOP)

    def test_empty_output_is_safe(self):
        self.assertEqual(self.chunk(""), [])
        self.assertEqual(to_primitives([], 0.25, 15.0), [])

    def test_magnitude_never_rounds_away_to_nothing(self):
        # 25 cm / 0.25 m = 1; a 15 deg turn must not quantise to zero primitives.
        self.assertEqual(to_primitives(self.chunk("forward 25"), 0.25, 15.0), [1])
        self.assertEqual(to_primitives(self.chunk("turn left 15"), 0.25, 15.0), [2])

    def test_stop_only_chunk(self):
        self.assertTrue(Decision(chunk=self.chunk("stop")).is_stop_only)
        self.assertFalse(Decision(chunk=self.chunk("forward 50")).is_stop_only)

    def test_trailing_stop_still_executes_the_motion_first(self):
        # Regression: "forward 100, stop" must drive, then stop -- not stop at once.
        decision = Decision(chunk=self.chunk("forward 100, forward 100, forward 25, stop"))
        self.assertFalse(decision.is_stop_only, "a chunk with motion must not short-circuit to STOP")
        self.assertTrue(decision.stops_at_end)
        prims = to_primitives(decision.chunk, 0.25, 15.0)
        self.assertEqual(prims.count(1), 9)
        self.assertEqual(prims[-1], 0)


class PromptTest(unittest.TestCase):
    def waypoint(self):
        return Waypoint(1, (0, 0, 0), (10, 10), 0.0, 2.0, 2.0, 0.0, 2.0, 2.0, "bearing")

    def test_template_is_used_verbatim(self):
        prompt = build_prompt("walk to the kitchen", [], SelectorConfig(), 8)
        self.assertIn(PROMPT_TEMPLATE.format("walk to the kitchen"), prompt)

    def test_ranges_are_stated(self):
        prompt = build_prompt("go", [], SelectorConfig(), 8)
        for token in ("forward <25-100>", "turn left <15-90>", "turn right <15-90>", "stop"):
            self.assertIn(token, prompt)

    def test_history_count_is_declared(self):
        self.assertIn("first 8 images", build_prompt("go", [], SelectorConfig(), 8))
        self.assertIn("single image", build_prompt("go", [], SelectorConfig(), 0))

    def test_circles_are_reference_only(self):
        prompt = build_prompt("go", [self.waypoint()], SelectorConfig(), 8)
        self.assertIn("reference only", prompt)
        self.assertIn("answer with motion", prompt)


class FeedbackTest(unittest.TestCase):
    """Regression: ep1378 repeated 'forward 100' five times against a wall
    because the history recorded only the command, never the outcome."""

    def test_blocked_execution_is_reported_as_blocked(self):
        entry = _outcome("forward 100", {"status": "collision", "travelled_m": 0.0, "actions": [1, 1]})
        self.assertEqual(entry["moved_m"], 0.0)
        self.assertIn("BLOCKED", entry["result"])

    def test_successful_execution_reports_distance(self):
        entry = _outcome("forward 100", {"status": "chunk_complete", "travelled_m": 0.97, "actions": [1] * 4})
        self.assertEqual(entry["moved_m"], 0.97)
        self.assertNotIn("BLOCKED", entry["result"])

    def test_prompt_tells_the_model_not_to_repeat_a_blocked_command(self):
        prompt = build_prompt("go", [], SelectorConfig(), 8,
                              [_outcome("forward 100", {"status": "collision", "travelled_m": 0.0})])
        self.assertIn("BLOCKED", prompt)
        self.assertIn("turn to face a different direction", prompt)


class DepthTest(unittest.TestCase):
    def test_explicit_normalization(self):
        np.testing.assert_allclose(depth_to_meters(np.array([[0.25, 0.5]], np.float32), 10.0, True), [[2.5, 5.0]])

    def test_metric_depth_under_one_metre_is_not_rescaled(self):
        np.testing.assert_allclose(depth_to_meters(np.array([[0.3]], np.float32), 10.0, False), [[0.3]])

    def test_occluder_on_the_marker_is_caught(self):
        patch = np.full((9, 9), 5.0, dtype=np.float32)
        patch[3:6, 3:6] = 1.0
        self.assertEqual(float(np.median(patch)), 5.0)
        self.assertEqual(occlusion_depth(patch, (4, 4), 4, 25.0), 1.0)

    def test_empty_patch_returns_none(self):
        self.assertIsNone(occlusion_depth(np.zeros((9, 9), np.float32), (4, 4), 4, 25.0))


class ClearanceTest(unittest.TestCase):
    def setUp(self):
        self.intr = CameraIntrinsics.from_hfov(64, 48, 90)

    def test_wall_ahead_is_detected(self):
        depth = np.full((48, 64), 5.0, dtype=np.float32)
        depth[:, 28:36] = 0.2
        self.assertLess(forward_clearance_m(depth, self.intr, (0.3, 0.8), 0.18, 5), 0.35)

    def test_open_space_is_clear(self):
        depth = np.full((48, 64), 5.0, dtype=np.float32)
        self.assertGreater(forward_clearance_m(depth, self.intr, (0.3, 0.8), 0.18, 5), 1.0)

    def test_blind_depth_never_fabricates_an_obstacle(self):
        self.assertEqual(forward_clearance_m(np.zeros((48, 64), np.float32), self.intr, (0.3, 0.8), 0.18, 30),
                         float("inf"))


class HistoryTest(unittest.TestCase):
    def test_all_atomic_frames_are_kept_and_8_sampled(self):
        manager = HistoryManager(HistoryConfig())
        for i in range(25):
            manager.add(np.full((2, 2, 3), i, dtype=np.uint8), "atomic")
        ids = [int(f[0, 0, 0]) for f in manager.select()]
        self.assertEqual(manager.describe()["stored"], 25)
        self.assertEqual(len(ids), 8)
        self.assertEqual((ids[0], ids[-1]), (0, 24), "the 8 frames must span the whole episode")

    def test_fewer_frames_than_the_budget_are_all_returned(self):
        manager = HistoryManager(HistoryConfig())
        for i in range(3):
            manager.add(np.full((2, 2, 3), i, dtype=np.uint8), "atomic")
        self.assertEqual(len(manager.select()), 3)

    def test_uniform_indices_span_endpoints(self):
        self.assertEqual(uniform_indices(20, 8)[0], 0)
        self.assertEqual(uniform_indices(20, 8)[-1], 19)


class SamplingTest(unittest.TestCase):
    def mapper(self):
        m = object.__new__(LocalMapper)
        m.config = AgentConfig()
        return m

    def corridor(self):
        local = np.full((160, 160), UNKNOWN, dtype=np.uint8)
        local[74:86, 40:120] = FREE
        local[73, 40:120] = OCCUPIED
        local[86, 40:120] = OCCUPIED
        return local

    def test_narrow_corridor_still_yields_candidates(self):
        mapper = self.mapper()
        local = np.full((160, 160), UNKNOWN, dtype=np.uint8)
        local[74:86, :] = FREE
        local[73, :] = OCCUPIED
        local[86, :] = OCCUPIED
        samples = mapper._ray_march_samples(local, mapper._clearance(local))
        self.assertTrue(samples)
        self.assertTrue(any(abs(s[2]) > 60 for s in samples))

    def test_clearance_gate_never_rejects_the_agent_cell(self):
        mapper = self.mapper()
        local = np.full((160, 160), UNKNOWN, dtype=np.uint8)
        local[76:84, 76:84] = FREE
        clearance = mapper._clearance(local)
        self.assertLessEqual(mapper._required_clearance_px(clearance), max(1.0, clearance[80, 80]))

    def test_frontiers_survive_a_walled_corridor(self):
        mapper = self.mapper()
        local = self.corridor()
        self.assertTrue(mapper._frontier_samples(local, mapper._clearance(local)))

    def test_unexplored_space_is_not_treated_as_an_obstacle(self):
        mapper = self.mapper()
        local = self.corridor()
        self.assertGreater(mapper._obstacle_clearance(local)[80, 40], mapper._clearance(local)[80, 40])


class MarkerProjectionTest(unittest.TestCase):
    """Camera sits 1.25 m up; a marker fixed at 0.40 m leaves frame below ~1.13 m."""

    def setUp(self):
        self.intr = CameraIntrinsics.from_hfov(640, 480, 90)

    def v_for(self, lift, z):
        return self.intr.fy * (1.25 - lift) / z + self.intr.cy

    def test_fixed_height_marker_falls_off_the_bottom_when_close(self):
        self.assertGreaterEqual(self.v_for(0.40, 1.0), self.intr.height)

    def test_adaptive_lift_brings_it_back(self):
        v_max = self.intr.height - 9.0
        for z in (0.5, 0.8, 1.0, 1.13):
            needed = (1.25 - 0.40) - (v_max - self.intr.cy) * z / self.intr.fy
            self.assertLess(self.v_for(min(1.10, max(0.40, 0.40 + needed)), z), self.intr.height, f"z={z}")


class ConfigTest(unittest.TestCase):
    def test_task_contract(self):
        c = AgentConfig()
        self.assertEqual(c.hfov_deg, 90)
        self.assertEqual(c.budget.max_agent_turns, 10)
        self.assertEqual(c.history.max_frames, 8)
        self.assertEqual(c.history.strategy, "all_uniform")
        self.assertFalse(c.history.annotate)
        self.assertEqual(c.selector.forward_cm_range, CM)
        self.assertEqual(c.selector.turn_deg_range, DEG)

    def test_codex_loader_reads_url_key_and_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "config.toml").write_text(
                'model_provider = "test"\nmodel = "gpt-x"\n[model_providers.test]\n'
                'base_url = "https://example.test/v1"\n', encoding="utf-8")
            (root / "auth.json").write_text(json.dumps({"OPENAI_API_KEY": "secret"}), encoding="utf-8")
            value = load_codex_credentials(root / "config.toml", root / "auth.json")
            self.assertEqual((value.base_url, value.api_key, value.model),
                             ("https://example.test/v1", "secret", "gpt-x"))

    def test_response_text_extraction(self):
        self.assertEqual(
            _response_text({"output": [{"type": "message",
                                        "content": [{"type": "output_text", "text": "<answer>forward 50</answer>"}]}]}),
            "<answer>forward 50</answer>")


if __name__ == "__main__":
    unittest.main()
