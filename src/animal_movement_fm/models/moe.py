"""
Mixture-of-experts (MoE) trunk for our pooling study.

The dense feedforward block of every encoder layer becomes a router and eight experts, of which each token uses two. The router sees only 
the token's state, so routing needs no species or domain label and an unseen species is routed like any other window.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_EXPERTS = 8
TOP_K = 2


class MoEFeedForward(nn.Module):
    """
    Top k routed experts, each shaped like the dense block: Linear, ReLU, dropout, Linear.

    Every expert runs on every token and the gate zeroes all but the top k. At this size that is cheaper than dispatching tokens, and it 
    keeps the forward deterministic.
    """

    def __init__(
        self, d_model: int, hidden: int, n_experts: int = N_EXPERTS, top_k: int = TOP_K, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.n_experts, self.top_k = n_experts, top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.w1 = nn.Parameter(torch.empty(n_experts, d_model, hidden))
        self.b1 = nn.Parameter(torch.empty(n_experts, hidden))
        self.w2 = nn.Parameter(torch.empty(n_experts, hidden, d_model))
        self.b2 = nn.Parameter(torch.empty(n_experts, d_model))
        self.dropout = nn.Dropout(dropout)
        # each expert starts as nn.Linear would; uniform within 1 / sqrt(fan_in) for weight and bias
        for weight, bias, fan_in in ((self.w1, self.b1, d_model), (self.w2, self.b2, hidden)):
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(weight, -bound, bound)
            nn.init.uniform_(bias, -bound, bound)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # router probabilities for every token and expert
        router_probs = self.router(x).softmax(dim=-1)

        # keep only the top k experts per token
        top_probs, top_indices = router_probs.topk(self.top_k, dim=-1)

        # renormalise the selected experts so their weights sum to 1
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)

        gates = torch.zeros_like(router_probs)
        gates.scatter_(-1, top_indices, top_probs)

        # run every token through the first layer of every expert; (B, S, D) x (E, D, H) -> (B, S, E, H)
        expert_hidden = torch.einsum(
            "...d,edh->...eh",
            x,
            self.w1,
        )
        expert_hidden = F.relu(expert_hidden + self.b1)
        expert_hidden = self.dropout(expert_hidden)

        # run each expert's hidden state through its second layer; B, S, E, H) × (E, H, D) -> (B, S, E, D)
        expert_outputs = torch.einsum(
            "...eh,ehd->...ed",
            expert_hidden,
            self.w2,
        )
        expert_outputs = expert_outputs + self.b2

        # weighted mixture of experts
        output = torch.einsum(
            "...e,...ed->...d",
            gates,
            expert_outputs,
        )

        # fraction of top k assignments going to each expert
        assignments = torch.zeros_like(router_probs)
        assignments.scatter_(-1, top_indices, 1.0)

        expert_share = (
            assignments
            .flatten(0, -2)
            .mean(dim=0)
            / self.top_k
        )

        # average probability assigned to each expert
        mean_router_prob = (
            router_probs
            .flatten(0, -2)
            .mean(dim=0)
        )

        # try have balanced expert usage
        aux_loss = (
            self.n_experts
            * (expert_share * mean_router_prob).sum()
        )

        return output, router_probs, aux_loss

class MoEEncoderLayer(nn.Module):
    """
    Post norm encoder layer laid out as `nn.TransformerEncoderLayer(batch_first=True)`, with the feed-forward routed.

    Written out rather than subclassed: in eval mode under no_grad the stock layer's fused fast path reads `linear1` and `linear2` directly and would 
    never call a replaced feedforward block.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        n_experts: int = N_EXPERTS,
        top_k: int = TOP_K,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # the dense width is split across the k active experts, so a token's active FFN matches the dense layers
        self.moe = MoEFeedForward(d_model, dim_feedforward // top_k, n_experts, top_k, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # standard post norm self-attention block
        attn_out, _ = self.self_attn(x, x, x, need_weights=False,)
        x = self.norm1(x + self.dropout1(attn_out))

        # routed feedforward block.
        ff, probs, aux = self.moe(x)
        x = self.norm2(x + self.dropout2(ff))

        return x, probs, aux


class MoEEncoder(nn.Module):
    """
    A stack of MoE layers standing in for `nn.TransformerEncoder`, with no final norm, as `MovementFM` builds it.

    The layers are copies of one initialised layer, as `nn.TransformerEncoder` clones its layer, so the two trunks start from the same convention. 
    After each forward, `aux_loss` holds the load balancing loss averaged over layers and `router_probs` the router probabilities of each layer, 
    `(batch, seq, n_experts)`.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        dropout: float = 0.1,
        n_experts: int = N_EXPERTS,
        top_k: int = TOP_K,
    ) -> None:
        super().__init__()
        layer = MoEEncoderLayer(d_model, nhead, dim_feedforward, dropout, n_experts, top_k)
        self.layers = nn.ModuleList(copy.deepcopy(layer) for _ in range(num_layers))
        self.aux_loss: torch.Tensor | None = None
        self.router_probs: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs, aux = [], []
        for layer in self.layers:
            x, p, a = layer(x)
            probs.append(p)
            aux.append(a)
        self.router_probs = probs
        self.aux_loss = torch.stack(aux).mean()
        return x
