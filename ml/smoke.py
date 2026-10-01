"""Exercise the selected model on synthetic CPU inputs; no historical training.

Run from the repository root: python ml/smoke.py
There is no game engine, teacher policy, downloaded data, or checkpoint here.
The randomly initialized small model has no meaningful competition performance.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from replay_learning.full_action_model import full_action_loss
from replay_learning.full_action_model_v73 import FullActionModelV73
from replay_learning.full_action_schema import (
    ACTOR_DIM,
    GLOBAL_DIM,
    MAX_ACTORS,
    MAX_ORDERS,
    ActionCodec,
)


def main() -> None:
    torch.manual_seed(7)
    torch.set_num_threads(2)
    device = torch.device("cpu")
    actions = [
        {
            "farmer": ["PASS"],
            "hands": [["HARVEST"], ["WATER"]],
            "market": [["SELL", "WHEAT", 2], ["BUY_PRODUCT", "FERTILIZER", 1]],
        },
        {"farmer": ["WATER"], "hands": [["PASS"], ["HARVEST"]], "market": []},
    ]
    codec = ActionCodec.fit(actions)
    targets = [codec.encode(action) for action in actions]
    for action, encoded in zip(actions, targets):
        assert codec.decode(encoded["units"], int(encoded["hands_length"]), encoded["market"]) == action
    unit_targets = torch.tensor(np.stack([row["units"] for row in targets])[None], device=device)
    market_targets = torch.tensor(np.stack([row["market"] for row in targets])[None], device=device)
    length_targets = torch.tensor([[int(row["hands_length"]) for row in targets]], device=device)

    global_features = torch.randn(1, 2, GLOBAL_DIM, device=device) * 0.05
    actor_features = torch.zeros(1, 2, MAX_ACTORS, ACTOR_DIM, device=device)
    actor_features[:, :, :3] = torch.randn(1, 2, 3, ACTOR_DIM, device=device) * 0.05
    actor_features[:, :, :3, 0] = 1.0  # Three active slots; padding is masked.
    actor_features[:, :, 0, 1] = 1.0  # Farmer indicator for the first actor.
    model = FullActionModelV73(
        global_dim=GLOBAL_DIM,
        actor_dim=ACTOR_DIM,
        unit_vocab_size=codec.unit_size,
        market_vocab_size=codec.market_size,
        market_bos_id=codec.market_bos_id,
        market_eos_id=codec.market_eos_id,
        hidden_size=32,  # Deliberately small smoke model; historical model used 192.
        max_actors=MAX_ACTORS,
        max_market_tokens=MAX_ORDERS + 1,
        value_head=False,
    ).to(device)

    teacher_forced = model(global_features, actor_features, market_targets=market_targets)
    losses = full_action_loss(
        teacher_forced,
        unit_targets,
        length_targets,
        market_targets,
        unit_pass_id=codec.unit_pass_id,
        unit_unk_id=codec.unit_unk_id,
        market_unk_id=codec.market_unk_id,
        non_pass_weight=2.0,
        hands_length_weight=0.2,
    )
    assert torch.isfinite(losses["total"])
    losses["total"].backward()  # Check gradient flow; no optimizer or training run.
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(bool(torch.isfinite(gradient).all()) for gradient in gradients)

    model.eval()
    with torch.inference_mode():
        free_decoded = model(global_features, actor_features)
    assert tuple(free_decoded["units_logits"].shape) == (1, 2, MAX_ACTORS, codec.unit_size)
    assert tuple(free_decoded["market_logits"].shape) == (1, 2, MAX_ORDERS + 1, codec.market_size)
    predicted_market = free_decoded["market_logits"].argmax(dim=-1)
    assert not bool((predicted_market == codec.market_bos_id).any())
    market_tokens = predicted_market[0, 0].tolist()
    if codec.market_eos_id in market_tokens:
        first_eos = market_tokens.index(codec.market_eos_id)
        assert all(token == codec.market_eos_id for token in market_tokens[first_eos:])

    report = {
        "status": "passed",
        "verification_kind": "synthetic_cpu_smoke_only",
        "historical_training_reproduced": False,
        "device": str(device),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "smoke_hidden_size": model.hidden_size,
        "global_shape": list(global_features.shape),
        "actor_shape": list(actor_features.shape),
        "unit_logits_shape": list(free_decoded["units_logits"].shape),
        "market_logits_shape": list(free_decoded["market_logits"].shape),
        "synthetic_loss": {name: float(losses[name].detach()) for name in ("total", "units", "market", "hands_length")},
        "finite_gradient_tensors": len(gradients),
        "codec_roundtrip": True,
        "free_decode_excludes_bos": True,
        "free_decode_eos_absorbing_if_emitted": True,
        "optimizer_steps": 0,
        "game_evaluations": 0,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
