import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from agents import switcher as switcher_module
from agents.arpe import ARPEWeights
from agents.hsrr import HSRR, HSRRConfig
from agents.switcher import Switcher, SwitcherConfig
from agents.controller import SWITCHER_FIELD_SHAPES, switcher_observation_space


class _FrozenPolicy:
    def __init__(self):
        self.weight = torch.nn.Parameter(torch.ones(()))

    def parameters(self):
        return [self.weight]


class _FakeCandidate:
    def __init__(self, _artifact, *, seed, device):
        self.artifact = _artifact
        self.seed = seed
        self.device = torch.device("cpu")
        self.ppo = _FrozenPolicy()

    def after_reset(self):
        pass


class _FakeSwitcher:
    def __init__(self, _cfg):
        pass


class _FakePlanner:
    def __init__(self, **_kwargs):
        pass

    def reset(self):
        pass


def _config():
    return HSRRConfig(
        switcher=SwitcherConfig(path_to_weights="unused"),
    )


def test_hsrr_defaults_to_current_arpe_manifest_without_switcher_metadata():
    algorithm = HSRR(
        _config(),
        candidate_factory=_FakeCandidate,
        switcher_factory=_FakeSwitcher,
        planner_factory=_FakePlanner,
    )

    assert algorithm.candidate.seed == 0
    assert algorithm.candidate.device == torch.device("cpu")
    root = Path(__file__).resolve().parents[1]
    mapping = json.loads((root / "configs/arpe.json").read_text())
    assert algorithm.candidate.artifact == ARPEWeights.from_mapping(mapping, root)


def test_hsrr_uses_explicit_arpe_paths_without_a_configuration_roundtrip(monkeypatch):
    monkeypatch.setattr(Path, "read_text", lambda *args, **kwargs: pytest.fail("read manifest"))
    artifact = object()
    algorithm = HSRR(
        _config(),
        candidate=artifact,
        candidate_factory=_FakeCandidate,
        switcher_factory=_FakeSwitcher,
        planner_factory=_FakePlanner,
    )
    assert algorithm.candidate.artifact is artifact


@pytest.mark.parametrize("option,value", [("deterministic", True), ("checkpoint_kind", "best")])
def test_switcher_has_no_variant_switches(option, value):
    with pytest.raises(ValueError):
        SwitcherConfig(**{option: value})


def test_switcher_loads_latest_checkpoint_before_best(tmp_path):
    best = tmp_path / "best_000001.pth"
    best.touch()
    assert Switcher._resolve_checkpoint(tmp_path) == best.resolve()
    latest = tmp_path / "checkpoint_000002_000000200.pth"
    latest.touch()
    assert Switcher._resolve_checkpoint(tmp_path) == latest.resolve()


def _assert_config_free_switcher(weights_dir, monkeypatch):
    assert not (weights_dir / "config.json").exists()
    monkeypatch.setattr(
        Path, "read_text",
        lambda *args, **kwargs: pytest.fail("Switcher read a training configuration"),
    )
    policy = Switcher(SwitcherConfig(path_to_weights=str(weights_dir), device="cpu", seed=42))
    checkpoint = torch.load(policy.checkpoint_path, map_location="cpu", weights_only=False)
    loaded = policy.ppo.state_dict()
    assert loaded.keys() == checkpoint["model"].keys()
    for key, value in loaded.items():
        torch.testing.assert_close(value, checkpoint["model"][key], rtol=0, atol=0)
    assert policy.ppo.encoder.get_out_size() == 128
    assert policy.ppo.cfg.use_rnn is False
    assert policy.rnn_state_size == 1  # Sample Factory's feed-forward dummy state.
    assert not policy.ppo.training
    assert all(not parameter.requires_grad for parameter in policy.ppo.parameters())
    state = {
        key: np.zeros((16, *shape), dtype=np.float32)
        for key, shape in SWITCHER_FIELD_SHAPES.items()
    }
    state["caar_action"][:, 1] = 1.0
    state["aoreplan_action"][:, 4] = 1.0
    actions = policy.choose(state)
    assert actions.shape == (16,)
    assert np.all(np.isin(actions, (0, 1)))
    policy.after_reset()
    np.testing.assert_array_equal(policy.choose(state), actions)


def test_switcher_loads_full_model_without_config_or_training_recipe(tmp_path, monkeypatch):
    switcher_module.register_custom_components()
    _, cfg = switcher_module.validate_config({
        "name": "test-switcher",
        "experiment_settings": {
            "encoder_custom": "switcher", "hidden_size": 128, "normalize_input": False,
        },
        "async_ppo": {"use_rnn": False},
    })
    model = switcher_module.create_actor_critic(
        cfg, switcher_observation_space(), gym.spaces.Discrete(2)
    )
    checkpoint_dir = tmp_path / "checkpoint_p0"
    checkpoint_dir.mkdir()
    torch.save({"model": model.state_dict()}, checkpoint_dir / "checkpoint_000001_000000100.pth")
    _assert_config_free_switcher(tmp_path, monkeypatch)


def test_published_switcher_checkpoint_loads_without_config_when_available(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / SwitcherConfig().path_to_weights / "checkpoint_p0"
    if not list(source.glob("*.pth")):
        pytest.skip("Published Switcher weights are distributed separately")
    checkpoint = Switcher._resolve_checkpoint(source)
    checkpoint_dir = tmp_path / "checkpoint_p0"
    checkpoint_dir.mkdir()
    shutil.copyfile(checkpoint, checkpoint_dir / checkpoint.name)
    _assert_config_free_switcher(tmp_path, monkeypatch)


def test_switcher_uses_sampled_actor_actions_and_preserves_reset_seed(monkeypatch):
    count = 16
    state = {
        key: np.zeros((count, *shape), dtype=np.float32)
        for key, shape in SWITCHER_FIELD_SHAPES.items()
    }
    state["caar_action"][:, 1] = 1.0
    state["aoreplan_action"][:, 4] = 1.0

    def actor(observations, rnn_states):
        assert torch.all(observations["switch_allowed"] == 1.0)
        assert rnn_states.shape == (count, 0)
        return {"actions": torch.randint(2, (count,))}

    switcher = Switcher.__new__(Switcher)
    switcher.cfg = SwitcherConfig(seed=42)
    switcher.device = torch.device("cpu")
    switcher.rnn_state_size = 0
    switcher.ppo = actor
    monkeypatch.setattr(
        switcher_module, "prepare_and_normalize_obs", lambda _actor, obs: obs
    )

    torch.manual_seed(42)
    expected_first = torch.randint(2, (count,)).numpy()
    expected_second = torch.randint(2, (count,)).numpy()
    switcher.after_reset()
    np.testing.assert_array_equal(switcher.choose(state), expected_first)
    np.testing.assert_array_equal(switcher.choose(state), expected_second)
    switcher.after_reset()
    np.testing.assert_array_equal(switcher.choose(state), expected_first)


@pytest.mark.parametrize("allowed", [(False, True, False), (False, False, False)])
def test_hsrr_only_sends_nonwait_rows_to_switcher(allowed):
    features = {"obs": np.arange(3, dtype=np.float32)[:, None]}
    prepared = SimpleNamespace(switch_allowed_mask=allowed, switcher_state=features)
    selected_batches = []
    resolved_branches = []

    def choose(state):
        selected_batches.append(state["obs"])
        return np.ones(len(state["obs"]), dtype=np.int64)

    def resolve(branches):
        resolved_branches.append(tuple(branches))
        return (1, 4, 3)

    algorithm = HSRR.__new__(HSRR)
    algorithm.switcher = SimpleNamespace(choose=choose)
    algorithm.controller = SimpleNamespace(
        prepare_actions=lambda *_args: prepared,
        resolve_actions=resolve,
    )
    assert algorithm.act([{}, {}, {}]) == [1, 4, 3]
    assert resolved_branches == [(1,) if any(allowed) else ()]
    if any(allowed):
        assert len(selected_batches) == 1
        np.testing.assert_array_equal(selected_batches[0], features["obs"][[1]])
    else:
        assert selected_batches == []
