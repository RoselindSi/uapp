"""Stability Oracle backbone wrapper.

This is the ONE file where you (the student) need to write code to
integrate with the real Stability Oracle repo. Everything else in this
package is already written and tested.

You need to fill in two functions:

    1. load_stability_oracle(checkpoint_path)
       -> returns a frozen nn.Module

    2. extract_graph_embedding(model, protein_input)
       -> returns a (d,) tensor h_G for a single protein/mutation

The rest of this file provides:
    - A BackboneAdapter class that wraps your loaded model and exposes
      a consistent interface to the rest of the codebase.
    - A batch helper that loops over mutations and returns a stacked
      (N, d) tensor of embeddings.

## What to look for in the Stability Oracle repo

Clone from https://github.com/danny305/StabilityOracle and read:
    - scripts/run_stability_oracle.py  (their CLI for inference)
    - StabilityOracle/models/              (model definitions)

The architecture is a pretrained feature extractor (MutComputeXGT) plus
a regression head. For this project you want the output of the feature
extractor BEFORE the regression head — that's the h_G used in equation (1)
of the proposal.

Typical pattern in their code:
    model = StabilityOracleModel(...)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

And their forward pass likely looks like:
    logits = model(graph_inputs, from_aa, to_aa)

You want to hook into the intermediate representation. Depending on
their code, you can either:
    (a) call a method like model.encode(graph_inputs) if it exists, or
    (b) register a forward hook on the last layer of the feature extractor
        and grab its output.

Option (b) is more robust and does not require modifying their code. An
example is shown in `_EmbeddingHook` below.

## Fallback: mean-pool residue embeddings yourself

If the cleanest signal you can grab is a per-residue embedding tensor of
shape (n_residues, d), just mean-pool it to get h_G of shape (d,). That's
exactly equation (1) in the proposal.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm

log = logging.getLogger("uapp.backbone")


# ---------------------------------------------------------------------------
# TODO: YOU FILL THESE IN
# ---------------------------------------------------------------------------

def load_stability_oracle(checkpoint_path: str | Path) -> nn.Module:
    """Load the pretrained Stability Oracle backbone and freeze it.

    Steps:
        1. Import the Stability Oracle model class from their repo.
           You'll need to add their repo to your PYTHONPATH or install it.
        2. Instantiate with whatever config it expects.
        3. Load the checkpoint.
        4. Call .eval() and set requires_grad=False on all parameters.
        5. Return it.

    Example skeleton (you'll need to adapt to their actual API):

        sys.path.insert(0, "/path/to/StabilityOracle")
        from StabilityOracle.models.graph_transformer import StabilityOracleModel

        model = StabilityOracleModel(config=default_config)
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model
    """
    # TODO(you): replace this with the real loader.
    raise NotImplementedError(
        "load_stability_oracle is not implemented yet. See uapp/backbone.py "
        "for the template. The smoke test does not call this function."
    )


def extract_graph_embedding(
    model: nn.Module,
    protein_input: Any,
) -> torch.Tensor:
    """Run one forward pass and return the graph-level embedding h_G.

    Parameters
    ----------
    model : frozen Stability Oracle module (from load_stability_oracle)
    protein_input : whatever shape Stability Oracle expects for a single
                    mutation — typically a graph dict or a processed
                    microenvironment. Match their inference script.

    Returns
    -------
    Tensor of shape (d,) — the pooled graph-level embedding.

    Implementation strategy:
        Option A: if their model has an .encode() or .extract_features()
                  method, just call it and mean-pool if needed.
        Option B: register a forward hook on the last transformer layer,
                  run the full forward, grab the hook output, pool.

    Example with a hook:

        hook = _EmbeddingHook()
        handle = model.encoder.layers[-1].register_forward_hook(hook)
        with torch.no_grad():
            _ = model(protein_input)
        handle.remove()
        # residue_embeddings has shape (n_residues, d)
        residue_embeddings = hook.last_output
        h_G = residue_embeddings.mean(dim=0)  # (d,)
        return h_G
    """
    # TODO(you): replace this with the real extraction.
    raise NotImplementedError(
        "extract_graph_embedding is not implemented yet. See uapp/backbone.py "
        "for the template. The smoke test does not call this function."
    )


# ---------------------------------------------------------------------------
# Helper: forward hook for grabbing intermediate activations
# ---------------------------------------------------------------------------

class _EmbeddingHook:
    """Tiny forward-hook container. Usage in extract_graph_embedding above."""

    def __init__(self) -> None:
        self.last_output: torch.Tensor | None = None

    def __call__(self, module: nn.Module, inputs: Any, output: torch.Tensor) -> None:
        # If the layer returns a tuple (e.g., (hidden, attn)), take the first.
        if isinstance(output, tuple):
            output = output[0]
        self.last_output = output.detach()


# ---------------------------------------------------------------------------
# Adapter that the rest of the codebase talks to
# ---------------------------------------------------------------------------

class BackboneAdapter:
    """Consistent interface the rest of the codebase calls.

    Wrapping the backbone here means the training/evaluation code never
    has to know whether it's talking to the real Stability Oracle or a
    mock backbone in tests.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def embed_one(self, protein_input: Any) -> torch.Tensor:
        """Run one forward pass and return h_G of shape (d,)."""
        return extract_graph_embedding(self.model, protein_input)

    @torch.no_grad()
    def embed_many(
        self,
        protein_inputs: list[Any],
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Embed a list of mutations and return a stacked (N, d) tensor."""
        embeddings: list[torch.Tensor] = []
        iterator = protein_inputs
        if show_progress:
            iterator = tqdm(protein_inputs, desc="embedding", unit="mut")
        for x in iterator:
            h = self.embed_one(x)
            embeddings.append(h.cpu())
        return torch.stack(embeddings, dim=0)
