"""Sequence model and masked objectives for complete Kaggriculture actions.

The model deliberately knows nothing about replay identities or teachers.  It
maps numeric observations to one token per active field actor and an ordered,
autoregressive market token sequence.  Episode history is carried by the
temporal GRU; callers may detach the returned hidden state between truncated
BPTT chunks.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


PAD_TARGET = -100


class FullActionModel(nn.Module):
    """Compact recurrent policy for field actors and ordered market actions."""

    def __init__(
        self,
        *,
        global_dim: int,
        actor_dim: int,
        unit_vocab_size: int,
        market_vocab_size: int,
        market_bos_id: int,
        market_eos_id: int,
        hidden_size: int = 192,
        max_actors: int = 32,
        max_market_tokens: int = 11,
        actor_embedding_size: int = 32,
        market_embedding_size: int = 64,
        value_head: bool = True,
    ) -> None:
        super().__init__()
        values = {
            "global_dim": global_dim,
            "actor_dim": actor_dim,
            "unit_vocab_size": unit_vocab_size,
            "market_vocab_size": market_vocab_size,
            "hidden_size": hidden_size,
            "max_actors": max_actors,
            "max_market_tokens": max_market_tokens,
        }
        if any(int(value) <= 0 for value in values.values()):
            raise ValueError(f"model dimensions must be positive: {values}")
        if int(market_bos_id) < 0:
            raise ValueError("market_bos_id must be nonnegative")
        if not 0 <= int(market_eos_id) < int(market_vocab_size):
            raise ValueError("market_eos_id is outside the market vocabulary")

        self.global_dim = int(global_dim)
        self.actor_dim = int(actor_dim)
        self.unit_vocab_size = int(unit_vocab_size)
        self.market_vocab_size = int(market_vocab_size)
        self.market_bos_id = int(market_bos_id)
        self.market_eos_id = int(market_eos_id)
        self.hidden_size = int(hidden_size)
        self.max_actors = int(max_actors)
        self.max_market_tokens = int(max_market_tokens)
        self.has_value_head = bool(value_head)

        self.global_encoder = nn.Sequential(
            nn.Linear(self.global_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
        )
        self.temporal = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)

        self.actor_encoder = nn.Sequential(
            nn.Linear(self.actor_dim, self.hidden_size),
            nn.SiLU(),
        )
        self.actor_identity = nn.Embedding(self.max_actors, int(actor_embedding_size))
        self.unit_head = nn.Sequential(
            nn.Linear(self.hidden_size * 2 + int(actor_embedding_size), self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.unit_vocab_size),
        )
        self.hands_length_head = nn.Linear(self.hidden_size, self.max_actors)

        self.market_embedding = nn.Embedding(
            max(self.market_vocab_size, self.market_bos_id + 1),
            int(market_embedding_size),
        )
        self.market_cell = nn.GRUCell(
            int(market_embedding_size) + self.hidden_size, self.hidden_size
        )
        self.market_initial = nn.Linear(self.hidden_size, self.hidden_size)
        self.market_head = nn.Linear(self.hidden_size, self.market_vocab_size)
        self.value = nn.Linear(self.hidden_size, 1) if self.has_value_head else None

    def model_config(self) -> dict[str, Any]:
        """Return constructor fields needed to recreate this model."""

        return {
            "global_dim": self.global_dim,
            "actor_dim": self.actor_dim,
            "unit_vocab_size": self.unit_vocab_size,
            "market_vocab_size": self.market_vocab_size,
            "market_bos_id": self.market_bos_id,
            "market_eos_id": self.market_eos_id,
            "hidden_size": self.hidden_size,
            "max_actors": self.max_actors,
            "max_market_tokens": self.max_market_tokens,
            "actor_embedding_size": self.actor_identity.embedding_dim,
            "market_embedding_size": self.market_embedding.embedding_dim,
            "value_head": self.has_value_head,
        }

    def _validate_inputs(
        self,
        global_features: torch.Tensor,
        actor_features: torch.Tensor,
        market_targets: torch.Tensor | None,
    ) -> tuple[int, int]:
        if global_features.ndim != 3:
            raise ValueError("global_features must have shape [B,T,D]")
        if actor_features.ndim != 4:
            raise ValueError("actor_features must have shape [B,T,actors,A]")
        batch, steps, dimension = global_features.shape
        if dimension != self.global_dim:
            raise ValueError(f"global feature dim {dimension} != {self.global_dim}")
        if actor_features.shape[:2] != (batch, steps):
            raise ValueError("global and actor batch/time dimensions differ")
        if actor_features.shape[2:] != (self.max_actors, self.actor_dim):
            raise ValueError(
                "actor feature shape "
                f"{tuple(actor_features.shape[2:])} != "
                f"{(self.max_actors, self.actor_dim)}"
            )
        if market_targets is not None and market_targets.shape != (
            batch,
            steps,
            self.max_market_tokens,
        ):
            raise ValueError(
                "market_targets must have shape "
                f"[B,T,{self.max_market_tokens}]"
            )
        return batch, steps

    def _decode_market(
        self, context: torch.Tensor, market_targets: torch.Tensor | None
    ) -> torch.Tensor:
        """Decode each turn causally; targets supply only preceding tokens."""

        flat_context = context.reshape(-1, self.hidden_size)
        rows = flat_context.shape[0]
        state = torch.tanh(self.market_initial(flat_context))
        previous = torch.full(
            (rows,), self.market_bos_id, dtype=torch.long, device=context.device
        )
        flat_targets = (
            None
            if market_targets is None
            else market_targets.reshape(rows, self.max_market_tokens)
        )
        if flat_targets is not None and bool(
            ((flat_targets != PAD_TARGET) & (flat_targets == self.market_bos_id)).any()
        ):
            raise ValueError("market BOS is a decoder input and cannot be a target")
        finished = torch.zeros(rows, dtype=torch.bool, device=context.device)
        outputs: list[torch.Tensor] = []
        for position in range(self.max_market_tokens):
            embedded = self.market_embedding(previous)
            state = self.market_cell(torch.cat((embedded, flat_context), dim=-1), state)
            logits = self.market_head(state)
            # BOS is a decoder input sentinel, never a valid emitted order.
            if self.market_bos_id < self.market_vocab_size:
                logits = logits.clone()
                logits[:, self.market_bos_id] = torch.finfo(logits.dtype).min
            if market_targets is None and bool(finished.any()):
                forced = torch.full_like(logits, torch.finfo(logits.dtype).min)
                forced[:, self.market_eos_id] = 0.0
                logits = torch.where(finished[:, None], forced, logits)
            outputs.append(logits)

            if flat_targets is not None:
                previous = flat_targets[:, position]
                previous = torch.where(
                    previous == PAD_TARGET,
                    torch.full_like(previous, self.market_eos_id),
                    previous,
                )
                if bool(((previous < 0) | (previous >= self.market_vocab_size)).any()):
                    raise ValueError("market target token is outside the vocabulary")
            else:
                previous = torch.argmax(logits, dim=-1)
                finished |= previous == self.market_eos_id
        return torch.stack(outputs, dim=1).reshape(
            *context.shape[:2], self.max_market_tokens, self.market_vocab_size
        )

    def forward(
        self,
        global_features: torch.Tensor,
        actor_features: torch.Tensor,
        hidden: torch.Tensor | None = None,
        market_targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return logits and the recurrent state after the final input step.

        Supplying ``market_targets`` enables teacher forcing.  At market slot
        ``k`` only slots strictly before ``k`` are consumed, so changing a
        future target cannot alter earlier logits.
        """

        batch, steps = self._validate_inputs(
            global_features, actor_features, market_targets
        )
        if hidden is not None and hidden.shape != (1, batch, self.hidden_size):
            raise ValueError(
                f"hidden shape {tuple(hidden.shape)} != {(1, batch, self.hidden_size)}"
            )
        encoded_global = self.global_encoder(global_features)
        context, next_hidden = self.temporal(encoded_global, hidden)

        encoded_actors = self.actor_encoder(actor_features)
        identity = self.actor_identity(
            torch.arange(self.max_actors, device=actor_features.device)
        ).view(1, 1, self.max_actors, -1)
        identity = identity.expand(batch, steps, -1, -1)
        actor_context = context.unsqueeze(2).expand(-1, -1, self.max_actors, -1)
        unit_logits = self.unit_head(
            torch.cat((actor_context, encoded_actors, identity), dim=-1)
        )
        result = {
            "units_logits": unit_logits,
            "hands_length_logits": self.hands_length_head(context),
            "market_logits": self._decode_market(context, market_targets),
            "hidden": next_hidden,
        }
        if self.value is not None:
            result["value"] = self.value(context).squeeze(-1)
        return result


def _masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    pad_target: int,
    class_weights: torch.Tensor | None = None,
    doubled_token: int | None = None,
    doubled_weight: float = 2.0,
    ignored_token: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1).long()
    valid = flat_targets != int(pad_target)
    if ignored_token is not None:
        valid &= flat_targets != int(ignored_token)
    count = valid.sum()
    if not bool(valid.any()):
        return flat_logits.sum() * 0.0, count, count.to(dtype=flat_logits.dtype)
    selected_targets = flat_targets[valid]
    losses = F.cross_entropy(flat_logits[valid], selected_targets, reduction="none")
    weights = torch.ones_like(losses)
    if class_weights is not None:
        if class_weights.shape != (logits.shape[-1],):
            raise ValueError("class_weights must have one value per output token")
        weights = weights * class_weights.to(losses.device)[selected_targets]
    if doubled_token is not None and doubled_weight != 1.0:
        weights = torch.where(
            selected_targets == int(doubled_token),
            weights,
            torch.full_like(weights, float(doubled_weight)),
        )
    weight_sum = weights.sum()
    return (losses * weights).sum() / weight_sum.clamp_min(1.0), count, weight_sum


def full_action_loss(
    output: dict[str, torch.Tensor],
    unit_targets: torch.Tensor,
    hands_length_targets: torch.Tensor,
    market_targets: torch.Tensor,
    *,
    unit_pass_id: int | None = None,
    unit_unk_id: int | None = None,
    market_unk_id: int | None = None,
    unit_class_weights: torch.Tensor | None = None,
    market_class_weights: torch.Tensor | None = None,
    non_pass_weight: float = 2.0,
    hands_length_weight: float = 0.2,
    pad_target: int = PAD_TARGET,
) -> dict[str, torch.Tensor]:
    """Compute independently normalized losses over known, non-padding targets."""

    units, unit_count, unit_weight_sum = _masked_cross_entropy(
        output["units_logits"],
        unit_targets,
        pad_target=pad_target,
        class_weights=unit_class_weights,
        doubled_token=unit_pass_id,
        doubled_weight=non_pass_weight,
        ignored_token=unit_unk_id,
    )
    market, market_count, market_weight_sum = _masked_cross_entropy(
        output["market_logits"],
        market_targets,
        pad_target=pad_target,
        class_weights=market_class_weights,
        ignored_token=market_unk_id,
    )
    hands, hands_count, hands_weight_sum = _masked_cross_entropy(
        output["hands_length_logits"], hands_length_targets, pad_target=pad_target
    )
    total = units + market + float(hands_length_weight) * hands
    return {
        "total": total,
        "units": units,
        "market": market,
        "hands_length": hands,
        "unit_token_count": unit_count,
        "market_token_count": market_count,
        "hands_count": hands_count,
        "unit_loss_weight": unit_weight_sum,
        "market_loss_weight": market_weight_sum,
        "hands_loss_weight": hands_weight_sum,
    }


__all__ = ["FullActionModel", "PAD_TARGET", "full_action_loss"]
