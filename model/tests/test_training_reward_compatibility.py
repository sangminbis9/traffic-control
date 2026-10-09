"""A resumed run must retain the reward objective used by its saved replay."""

from dataclasses import asdict, replace
import json

import gymnasium as gym
import numpy as np
import pytest

from model.training import TrainingOptions, TrainingRunner
from model.utils.config import RewardConfig


class ObservationOnlyEnv(gym.Env):
    observation_space = gym.spaces.Box(0.0, 1.0, shape=(60,), dtype=np.float32)
    action_space = gym.spaces.Discrete(8)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(60, dtype=np.float32), {}


def runner(tmp_path, checkpoint=None):
    result = TrainingRunner(
        "reward-test", TrainingOptions(), output_dir=tmp_path,
        resume_checkpoint=checkpoint,
    )
    result.config = replace(
        result.config, dqn=replace(result.config.dqn, buffer_size=8),
    )
    return result


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.parametrize("changed", [
    {"queue_weight": 0.8}, {"waiting_weight": 0.1},
    {"max_waiting_weight": 0.1}, {"switch_penalty": 0.2},
    {"queue_scale": 20.0}, {"waiting_scale": 3000.0},
    {"max_waiting_scale": 60.0},
])
def test_incompatible_reward_is_rejected_before_model_load(tmp_path, monkeypatch, changed):
    checkpoint = tmp_path / "final.zip"
    checkpoint.write_bytes(b"must not be loaded")
    write_json(tmp_path / "experiment_config.json", {
        "reward_config": {**asdict(RewardConfig()), **changed},
    })
    monkeypatch.setattr(
        "model.training.TrafficDQN.load",
        lambda *_args, **_kwargs: pytest.fail("incompatible checkpoint was loaded"),
    )
    with pytest.raises(ValueError, match=next(iter(changed))):
        runner(tmp_path, checkpoint)._create_model(ObservationOnlyEnv())


@pytest.mark.parametrize("source", ["experiment_config.json", "training_metadata.json"])
def test_existing_metadata_formats_allow_matching_reward(tmp_path, source):
    metadata = {"reward_config": asdict(RewardConfig())}
    if source == "training_metadata.json":
        metadata = {"reproducibility": metadata}
    write_json(tmp_path / source, metadata)
    runner(tmp_path, tmp_path / "final.zip")._check_resume_reward()


@pytest.mark.parametrize("metadata", [None, {"reward_config": {}}, {"reward_config": None}])
def test_unverifiable_reward_is_not_silently_assumed_current(tmp_path, metadata):
    if metadata is not None:
        write_json(tmp_path / "experiment_config.json", metadata)
    with pytest.raises(ValueError, match="Start a new training run"):
        runner(tmp_path, tmp_path / "final.zip")._check_resume_reward()


def test_checkpoint_and_replay_resume_with_their_recorded_reward(tmp_path):
    original = runner(tmp_path)
    # Record a custom objective to check metadata uses the actual runner config.
    original.config = replace(original.config, reward=replace(original.config.reward, waiting_weight=0.4))
    original.model = original._create_model(ObservationOnlyEnv())
    observation = np.zeros((1, 60), dtype=np.float32)
    original.model.replay_buffer.add(
        observation, observation, np.array([3]), np.array([-1.25]), np.array([False]), [{}],
    )
    checkpoint = original._save_checkpoint(1)
    metadata = json.loads(checkpoint.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    assert metadata["reward_config"] == asdict(original.config.reward)
    assert len(metadata["model_sha256"]) == 64

    # Session-level stale metadata cannot override checkpoint-specific evidence.
    write_json(tmp_path / "training_metadata.json", {"reproducibility": {
        "reward_config": asdict(RewardConfig()),
    }})
    restored = runner(tmp_path, checkpoint)
    restored.config = original.config
    restored.model = restored._create_model(ObservationOnlyEnv())
    assert restored.model.replay_buffer.size() == 1
    assert restored.model.replay_buffer.rewards[0, 0] == -1.25

    with pytest.raises(ValueError, match="waiting_weight"):
        runner(tmp_path, checkpoint)._create_model(ObservationOnlyEnv())

    checkpoint.write_bytes(b"replaced after metadata was recorded")
    with pytest.raises(ValueError, match="checkpoint hash differs"):
        restored._check_resume_reward()


def test_session_metadata_records_actual_reward(tmp_path, monkeypatch):
    original = runner(tmp_path)
    original.config = replace(original.config, reward=replace(original.config.reward, waiting_weight=0.4))
    monkeypatch.setattr("model.training.collect_reproducibility_metadata", lambda _path: {
        "reward_config": asdict(RewardConfig()), "git_commit_sha": "test-commit",
    })
    original._write_metadata(tmp_path / "final.zip")
    metadata = json.loads((tmp_path / "training_metadata.json").read_text(encoding="utf-8"))
    assert metadata["reproducibility"]["reward_config"] == asdict(original.config.reward)
    assert metadata["reproducibility"]["git_commit_sha"] == "test-commit"


def test_cli_canonical_export_preserves_reward_metadata_for_resume(tmp_path, monkeypatch):
    from model import train

    original = runner(tmp_path / "session")
    original.model = original._create_model(ObservationOnlyEnv())
    observation = np.zeros((1, 60), dtype=np.float32)
    original.model.replay_buffer.add(
        observation, observation, np.array([3]), np.array([-1.25]), np.array([False]), [{}],
    )
    original._save_checkpoint(1, name="final")
    monkeypatch.setattr(original, "run", lambda: {"status": "COMPLETED"})
    config = replace(original.config, results_dir=tmp_path / "canonical")
    monkeypatch.setattr(train, "ProjectConfig", lambda: config)
    monkeypatch.setattr(train, "TrainingRunner", lambda *_args, **_kwargs: original)
    monkeypatch.setattr("sys.argv", ["model.train"])

    assert train.main() == 0
    checkpoint = config.results_dir / "dqn_intersection.zip"
    assert checkpoint.with_suffix(".metadata.json").read_bytes() == (
        original.output_dir / "final.metadata.json"
    ).read_bytes()
    # The canonical export retains the existing model-only/replay behavior.
    assert not checkpoint.with_suffix(".replay.pkl").exists()
    restored = runner(tmp_path / "resumed", checkpoint)._create_model(ObservationOnlyEnv())
    assert restored.replay_buffer.size() == 0
    for key, value in original.model.q_net.state_dict().items():
        np.testing.assert_array_equal(value.cpu().numpy(), restored.q_net.state_dict()[key].cpu().numpy())
