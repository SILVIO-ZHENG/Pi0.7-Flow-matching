import json
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

from openpi.training import g1_training


def test_sidecar_dataset_injects_numeric_and_memory(tmp_path):
    sidecar_path = tmp_path / "recap.jsonl"
    sidecar_path.write_text(
        json.dumps(
            {
                "episode_index": 1,
                "frame_index": 2,
                "advantage_indicator": 1.0,
                "is_human_intervention": False,
                "memory": "The gripper is aligned with the target.",
            },
            ensure_ascii=False,
        )
        + "\n"
    )

    class Dataset:
        def __getitem__(self, index):
            return {"episode_index": 1, "index": 2, "value": index}

        def __len__(self):
            return 1

    dataset = g1_training.SidecarDataset(
        Dataset(), g1_training.read_sidecar(sidecar_path), include_text=True
    )
    item = dataset[0]
    assert item[g1_training.ADVANTAGE_KEY] == 1.0
    assert item[g1_training.MEMORY_KEY] == "The gripper is aligned with the target."


@unittest.skipIf(torch is None, "PyTorch training dependencies are not installed")
def test_rl_token_weights_apply_to_losses():
    metadata = {
        g1_training.ADVANTAGE_KEY: torch.tensor([1.0, 0.0]),
        g1_training.INTERVENTION_KEY: torch.tensor([0.0, 1.0]),
    }
    config = g1_training.RLTokenConfig(
        enabled=True,
        positive_weight=2.0,
        negative_weight=0.5,
        intervention_weight=3.0,
    )
    weights = g1_training.make_rl_token_weights(metadata, config)
    losses = torch.ones(2, 4, 3)
    weighted = g1_training.apply_loss_weights(losses, weights)
    assert torch.allclose(weighted[0], torch.full((4, 3), 2.0))
    assert torch.allclose(weighted[1], torch.full((4, 3), 1.5))


@unittest.skipIf(torch is None, "PyTorch training dependencies are not installed")
def test_knowledge_insulation_freezes_vlm_like_parameters():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.paligemma_with_expert = torch.nn.Module()
            self.paligemma_with_expert.paligemma = torch.nn.Linear(2, 2)
            self.paligemma_with_expert.gemma_expert = torch.nn.Linear(2, 2)
            self.action_in_proj = torch.nn.Linear(2, 2)

    model = Model()
    stats = g1_training.apply_knowledge_insulation(
        model, g1_training.KnowledgeInsulationConfig(enabled=True)
    )
    assert stats["frozen"] > 0
    assert model.paligemma_with_expert.paligemma.weight.requires_grad is False
    assert model.paligemma_with_expert.gemma_expert.weight.requires_grad is True
    assert model.action_in_proj.weight.requires_grad is True
