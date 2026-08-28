"""Combines two arm IK solvers and two Dex3 retargeters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from g1_pi07.joints import DEFAULT_LAYOUT
from g1_pi07.joints import G1JointLayout
from g1_pi07.teleop.ik import DampedLeastSquaresIK
from g1_pi07.teleop.ik import IKResult
from g1_pi07.teleop.ik import Pose
from g1_pi07.teleop.retarget import Dex3Retargeter


@dataclass(frozen=True)
class TeleopTargets:
    full_position: np.ndarray
    policy_position: np.ndarray
    model_position: np.ndarray
    left_ik: IKResult
    right_ik: IKResult


@dataclass
class BimanualTeleopMapper:
    left_ik: DampedLeastSquaresIK
    right_ik: DampedLeastSquaresIK
    left_hand: Dex3Retargeter
    right_hand: Dex3Retargeter
    layout: G1JointLayout = DEFAULT_LAYOUT
    require_convergence: bool = True

    def map(
        self,
        *,
        current_q43: np.ndarray,
        left_wrist: Pose,
        right_wrist: Pose,
        left_keypoints: np.ndarray,
        right_keypoints: np.ndarray,
    ) -> TeleopTargets:
        policy_seed = self.layout.full_to_policy(current_q43)
        left_result = self.left_ik.solve(left_wrist, policy_seed[:7])
        right_result = self.right_ik.solve(right_wrist, policy_seed[7:14])
        if self.require_convergence and (not left_result.converged or not right_result.converged):
            raise RuntimeError("Dual-arm IK did not converge for both arms; hold the previous safe command")
        policy = np.concatenate(
            [
                left_result.q,
                right_result.q,
                self.left_hand.retarget(left_keypoints),
                self.right_hand.retarget(right_keypoints),
            ]
        ).astype(np.float32)
        return TeleopTargets(
            full_position=self.layout.policy_to_full(policy, base_full=current_q43),
            policy_position=policy,
            model_position=self.layout.policy_to_model(policy),
            left_ik=left_result,
            right_ik=right_result,
        )
