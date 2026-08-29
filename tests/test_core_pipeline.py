"""Exercise the end-to-end Unitree G1 data and control pipeline."""

from __future__ import annotations

from typing import ClassVar
import unittest

import numpy as np
import pytest

from examples.unitree_g1.common.trajectory import make_execution_plan
from examples.unitree_g1.rtc.rtc_chunker import RtcChunker
from g1_pi07.data.chunks import make_action_chunk
from g1_pi07.data.normalization import QuantileStats
from g1_pi07.data.time_sync import ActionCentricSynchronizer
from g1_pi07.data.time_sync import AlignmentError
from g1_pi07.data.types import ActionTargetFrame
from g1_pi07.data.types import CameraFrame
from g1_pi07.data.types import RobotStateFrame
from g1_pi07.deployment.rolling_controller import RollingChunkController
from g1_pi07.joints import ACTIVE_ACTION_DIM
from g1_pi07.joints import DEFAULT_LAYOUT
from g1_pi07.joints import FULL_DOF
from g1_pi07.joints import MODEL_ACTION_DIM
from g1_pi07.teleop.ik import DampedLeastSquaresIK
from g1_pi07.teleop.ik import Pose
from g1_pi07.teleop.xr import RigidXrCalibration
from openpi.training.g1_training import INTERVENTION_KEY
from openpi.training.g1_training import ActionMaskDataset
from openpi.training.g1_training import AppendMemoryToPrompt
from openpi.training.g1_training import SidecarDataset
from openpi.training.g1_training import read_sidecar

try:
    import torch

    from g1_pi07.training.objectives import JointObjectiveConfig
    from g1_pi07.training.objectives import knowledge_insulated_backward
except ModuleNotFoundError:
    torch = None


class JointLayoutTest(unittest.TestCase):
    """Verify lossless conversion across canonical joint representations."""

    def test_round_trip_preserves_inactive_joints(self) -> None:
        full = np.arange(FULL_DOF, dtype=np.float32)
        policy = DEFAULT_LAYOUT.full_to_policy(full)
        model = DEFAULT_LAYOUT.policy_to_model(policy)
        rebuilt = DEFAULT_LAYOUT.policy_to_full(DEFAULT_LAYOUT.model_to_policy(model), base_full=full)
        np.testing.assert_array_equal(rebuilt, full)
        assert model.shape == (MODEL_ACTION_DIM,)
        assert np.all(model[ACTIVE_ACTION_DIM:] == 0)


class ChunkTest(unittest.TestCase):
    """Verify horizon padding, masks, and source-index tracking."""

    def test_tail_padding_and_masks(self) -> None:
        trajectory = np.arange(3 * ACTIVE_ACTION_DIM, dtype=np.float32).reshape(3, ACTIVE_ACTION_DIM)
        chunk = make_action_chunk(trajectory, 1, horizon=5)
        assert chunk.actions.shape == (5, MODEL_ACTION_DIM)
        np.testing.assert_array_equal(chunk.step_mask, [True, True, False, False, False])
        np.testing.assert_array_equal(chunk.actions[1, :ACTIVE_ACTION_DIM], chunk.actions[-1, :ACTIVE_ACTION_DIM])
        assert int(chunk.loss_mask.sum()) == 2 * ACTIVE_ACTION_DIM


class SyncTest(unittest.TestCase):
    """Verify action-centered state and multi-camera synchronization."""

    def test_action_centred_nearest_match(self) -> None:
        sync = ActionCentricSynchronizer(max_state_delta_ms=10, max_camera_delta_ms=20)
        state = RobotStateFrame(
            timestamp_ns=100_000_000,
            sequence=7,
            q=np.zeros(43),
            dq=np.zeros(43),
            tau_est=np.zeros(43),
            imu=np.zeros(10),
            validity_mask=np.ones(43, dtype=np.bool_),
        )
        sync.push_state(state)
        for name, delta in (("head", -3_000_000), ("left_wrist", 2_000_000), ("right_wrist", 5_000_000)):
            sync.push_camera(CameraFrame(100_000_000 + delta, name, np.zeros((4, 5, 3), dtype=np.uint8)))
        action = ActionTargetFrame(101_000_000, "ep-1", 0, 7, np.zeros(43))
        aligned = sync.align(action)
        assert aligned.state.sequence == 7
        assert set(aligned.cameras) == {"head", "left_wrist", "right_wrist"}
        assert aligned.state_delta_ns == -1000000

    def test_declared_source_sequence_still_obeys_time_tolerance(self) -> None:
        sync = ActionCentricSynchronizer(max_state_delta_ms=5, max_camera_delta_ms=20)
        sync.push_state(
            RobotStateFrame(0, 9, np.zeros(43), np.zeros(43), np.zeros(43), np.zeros(10), np.ones(43, bool))
        )
        for name in sync.camera_names:
            sync.push_camera(CameraFrame(100_000_000, name, np.zeros((2, 2, 3), dtype=np.uint8)))
        action = ActionTargetFrame(100_000_000, "ep", 0, 9, np.zeros(43))
        with pytest.raises(AlignmentError):
            sync.align(action)


class NormalizationTest(unittest.TestCase):
    """Verify robust quantile normalization and inverse transforms."""

    def test_quantile_round_trip(self) -> None:
        values = np.linspace(-2, 3, 1000, dtype=np.float32)[:, None]
        stats = QuantileStats.fit(values)
        probe = np.asarray([[stats.q01[0]], [stats.q99[0]]], dtype=np.float32)
        normalized = stats.normalize(probe)
        np.testing.assert_allclose(normalized[:, 0], [-1.0, 1.0], atol=1e-5)
        np.testing.assert_allclose(stats.denormalize(normalized), probe, atol=1e-5)

    def test_constant_dimension_maps_to_zero_and_round_trips(self) -> None:
        values = np.ones((20, 2), dtype=np.float32) * np.asarray([3.0, -2.0])
        stats = QuantileStats.fit(values)
        normalized = stats.normalize(values[:1])
        np.testing.assert_array_equal(normalized, np.zeros((1, 2), dtype=np.float32))
        np.testing.assert_array_equal(stats.denormalize(normalized), values[:1])


class _TranslationKinematics:
    """Analytic translation-only kinematics fixture for deterministic IK tests."""

    def forward(self, q: np.ndarray) -> Pose:
        return Pose(q[:3], np.asarray([0.0, 0.0, 0.0, 1.0]))

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        jacobian = np.zeros((6, len(q)), dtype=np.float64)
        jacobian[:3, :3] = np.eye(3)
        return jacobian


class IKTest(unittest.TestCase):
    """Verify bounded damped least-squares inverse kinematics."""

    def test_dls_reaches_simple_translation(self) -> None:
        solver = DampedLeastSquaresIK(
            _TranslationKinematics(),
            lower_limits=np.full(3, -1.0),
            upper_limits=np.full(3, 1.0),
            max_iterations=30,
        )
        result = solver.solve(Pose(np.asarray([0.2, -0.1, 0.3]), np.asarray([0, 0, 0, 1])), np.zeros(3))
        assert result.converged
        np.testing.assert_allclose(result.q, [0.2, -0.1, 0.3], atol=2e-3)


class XrCalibrationTest(unittest.TestCase):
    """Verify rigid XR-to-robot frame calibration behavior."""

    def test_scale_rotation_and_translation(self) -> None:
        # 180 degrees about z: (x, y) -> (-x, -y).
        calibration = RigidXrCalibration(
            np.asarray([1.0, 2.0, 3.0]),
            np.asarray([0.0, 0.0, 1.0, 0.0]),
            position_scale=2.0,
        )
        point = calibration.transform_points(np.asarray([0.5, 0.0, 0.0]))
        np.testing.assert_allclose(point, [0.0, 2.0, 3.0], atol=1e-7)


class _EpisodeMeta:
    """Minimal episode metadata fixture for wrapped dataset tests."""

    episodes: ClassVar[list[dict[str, int]]] = [{"length": 3}]


class _TinyEpisodeDataset:
    """Small deterministic dataset fixture with one short episode."""

    meta = _EpisodeMeta()

    def __getitem__(self, index: int) -> dict:
        return {"episode_index": 0, "frame_index": index, "index": index}

    def __len__(self) -> int:
        return 3


class ActionMaskDatasetTest(unittest.TestCase):
    """Verify action masks and sidecar metadata injection."""

    def test_sidecar_preserves_episode_boundary_for_padding_mask(self) -> None:
        wrapped = SidecarDataset(_TinyEpisodeDataset(), {})
        masked = ActionMaskDataset(wrapped, action_horizon=5, valid_action_dim=28)
        item = masked[1]
        np.testing.assert_array_equal(item["g1_action_step_mask"], [True, True, False, False, False])

    def test_string_false_sidecar_value_is_not_truthy(self) -> None:
        wrapped = SidecarDataset(
            _TinyEpisodeDataset(),
            {(0, 0): {"is_human_intervention": "false"}},
        )
        assert wrapped[0][INTERVENTION_KEY] == 0.0

    def test_duplicate_sidecar_key_is_rejected(self) -> None:
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.jsonl"
            row = '{"episode_index": 0, "frame_index": 1}'
            path.write_text(f"{row}\n{row}\n", encoding="utf-8")
            with pytest.raises(ValueError, match="Duplicate episode/frame key"):
                read_sidecar(path)

    def test_memory_is_appended_before_robot_transform(self) -> None:
        item = AppendMemoryToPrompt()({"prompt": "Pick the box", "g1_memory": "Both hands are aligned."})
        assert item["prompt"] == "Pick the box\nMemory: Both hands are aligned."


@unittest.skipIf(torch is None, "PyTorch training dependencies are not installed")
class KnowledgeInsulationTest(unittest.TestCase):
    """Verify gradient isolation between VLM and action-expert objectives."""

    def test_flow_gradient_is_removed_from_vlm_only(self) -> None:
        vlm = torch.nn.Parameter(torch.tensor(2.0))
        expert = torch.nn.Parameter(torch.tensor(3.0))
        result = knowledge_insulated_backward(
            fast_loss_fn=lambda: vlm.square(),
            flow_loss_fn=lambda: (vlm + expert).square(),
            vlm_parameters=[vlm],
            config=JointObjectiveConfig(enabled=True),
        )
        assert float(vlm.grad) == pytest.approx(4.0)
        assert float(expert.grad) == pytest.approx(10.0)
        assert float(result["fast_ce_loss"]) == pytest.approx(4.0)


class RollingControllerTest(unittest.TestCase):
    """Verify latency alignment and fallback behavior of rolling execution."""

    def test_observed_latency_skips_committed_prefix(self) -> None:
        controller = RollingChunkController(horizon=5, min_replan_steps=1, initial_delay_steps=1, blend_steps=0)
        controller.begin_request(7)
        controller.next_action(np.zeros(2, dtype=np.float32))
        controller.next_action(np.zeros(2, dtype=np.float32))
        delay = controller.accept(7, np.arange(10, dtype=np.float32).reshape(5, 2))
        assert delay == 2
        np.testing.assert_array_equal(controller.next_action(np.zeros(2)), [4.0, 5.0])

    def test_short_suffix_is_zero_padded_to_full_prefix_horizon(self) -> None:
        controller = RollingChunkController(horizon=5, min_replan_steps=1, initial_delay_steps=1, blend_steps=0)
        controller.begin_request(0)
        controller.accept(0, np.arange(10, dtype=np.float32).reshape(5, 2))
        for _ in range(4):
            controller.next_action(np.zeros(2, dtype=np.float32))
        context = controller.begin_request(1)
        assert context.prefix.shape == (5, 2)
        np.testing.assert_array_equal(context.prefix[0], [8.0, 9.0])
        np.testing.assert_array_equal(context.prefix[1:], np.zeros((4, 2), dtype=np.float32))


class RtcChunkerTest(unittest.TestCase):
    """Verify RTC request lifecycle, prefixes, and stale-result rejection."""

    def test_failed_request_can_be_cancelled_and_replanned(self) -> None:
        chunker = RtcChunker(horizon=5, min_horizon=1, delay_buffer_size=4, initial_delay_steps=1)
        chunker.make_request_context(4)
        assert not chunker.should_request()
        assert chunker.cancel_request(4)
        assert chunker.should_request()

    def test_unsolicited_result_is_rejected(self) -> None:
        chunker = RtcChunker(horizon=5, min_horizon=1, delay_buffer_size=4, initial_delay_steps=1)
        with pytest.raises(ValueError, match="request_id does not match"):
            chunker.accept_new_chunk(7, np.zeros((5, 2), dtype=np.float32))


class ExecutionPlanTest(unittest.TestCase):
    """Verify action interpolation, clipping, and invalid-input rejection."""

    def test_non_finite_policy_action_is_rejected(self) -> None:
        actions = np.zeros((5, 28), dtype=np.float32)
        actions[0, 0] = np.nan
        with pytest.raises(ValueError, match="must not contain NaN/Inf"):
            make_execution_plan(
                actions,
                state_dim=28,
                max_action_chunk_len=5,
                policy_action_hz=20,
                control_hz=20,
                interpolation="linear",
                lower_limits=np.full(28, -1.0),
                upper_limits=np.full(28, 1.0),
            )


if __name__ == "__main__":
    unittest.main()
