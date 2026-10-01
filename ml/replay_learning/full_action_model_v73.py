"""Full-action model with masked communication between all field actors.

V66 pooled actor features into the shared context.  V73 keeps the same complete
action decoder but lets farmer and hands attend to each other before command
prediction and ordered-market decoding.  Inactive actor slots are masked so
padding cannot influence coordination.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from replay_learning.full_action_model import FullActionModel


MODEL_FAMILY = "v73_masked_actor_attention"


class FullActionModelV73(FullActionModel):
    """FullActionModel plus a masked self-attention block over active actors."""

    def __init__(self, **kwargs: Any) -> None:
        route_vocab_size = int(kwargs.pop("route_vocab_size", 0))
        super().__init__(**kwargs)
        self.route_vocab_size = route_vocab_size
        self.route_embedding = (
            nn.Embedding(route_vocab_size, self.hidden_size)
            if route_vocab_size > 0
            else None
        )
        if self.route_embedding is not None:
            nn.init.zeros_(self.route_embedding.weight)
        self.actor_mixer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=4,
                dim_feedforward=self.hidden_size * 2,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=1,
            norm=nn.LayerNorm(self.hidden_size),
        )
        self.actor_identity_projection = nn.Linear(
            self.actor_identity.embedding_dim, self.hidden_size
        )
        self.actor_pool_norm = nn.LayerNorm(self.hidden_size)

    def model_config(self) -> dict[str, Any]:
        return {
            **super().model_config(),
            "route_vocab_size": self.route_vocab_size,
        }

    def forward(
        self,
        global_features: torch.Tensor,
        actor_features: torch.Tensor,
        hidden: torch.Tensor | None = None,
        market_targets: torch.Tensor | None = None,
        route_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, steps = self._validate_inputs(
            global_features, actor_features, market_targets
        )
        if hidden is not None and hidden.shape != (1, batch, self.hidden_size):
            raise ValueError(
                f"hidden shape {tuple(hidden.shape)} != {(1, batch, self.hidden_size)}"
            )

        encoded_global = self.global_encoder(global_features)
        context, next_hidden = self.temporal(encoded_global, hidden)
        if self.route_embedding is not None:
            if route_ids is None or route_ids.shape != (batch, steps):
                raise ValueError(
                    f"route_ids must have shape {(batch, steps)} when route_vocab_size > 0"
                )
            if bool((route_ids < 0).any()) or bool(
                (route_ids >= self.route_vocab_size).any()
            ):
                raise ValueError("route_ids contain an out-of-range value")
            context = context + self.route_embedding(route_ids)

        actor_exists = actor_features[..., 0:1]
        actor_features = actor_features * actor_exists
        encoded_actors = self.actor_encoder(actor_features)
        identity = self.actor_identity(
            torch.arange(self.max_actors, device=actor_features.device)
        ).view(1, 1, self.max_actors, -1)
        identity = identity.expand(batch, steps, -1, -1)
        mixed_input = encoded_actors + self.actor_identity_projection(identity)

        flat_actors = mixed_input.reshape(batch * steps, self.max_actors, self.hidden_size)
        exists = actor_features[..., 0].reshape(batch * steps, self.max_actors) > 0.5
        no_active = ~exists.any(dim=1)
        if bool(no_active.any()):
            exists = exists.clone()
            exists[no_active, 0] = True
        attended = self.actor_mixer(
            flat_actors, src_key_padding_mask=~exists
        ).reshape(batch, steps, self.max_actors, self.hidden_size)

        active = exists.reshape(batch, steps, self.max_actors).unsqueeze(-1)
        pooled = (attended * active.to(attended.dtype)).sum(dim=2)
        pooled = pooled / active.sum(dim=2).clamp_min(1)
        policy_context = context + self.actor_pool_norm(pooled)

        actor_context = policy_context.unsqueeze(2).expand(
            -1, -1, self.max_actors, -1
        )
        unit_logits = self.unit_head(
            torch.cat((actor_context, attended, identity), dim=-1)
        )
        result = {
            "units_logits": unit_logits,
            "hands_length_logits": self.hands_length_head(policy_context),
            "market_logits": self._decode_market(policy_context, market_targets),
            "hidden": next_hidden,
        }
        if self.value is not None:
            result["value"] = self.value(policy_context).squeeze(-1)
        return result


__all__ = ["FullActionModelV73", "MODEL_FAMILY"]
