from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .model import MicroVQVAEModel


def load_microvqvae_checkpoint(checkpoint_path: str | Path, device: str = 'auto') -> Tuple[MicroVQVAEModel, Dict[str, object]]:
    from .model import MicroVQVAEModel

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    resolved_device = 'cuda' if device == 'auto' and torch.cuda.is_available() else ('cpu' if device == 'auto' else device)
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

    hyper_parameters = dict(checkpoint.get('hyper_parameters', {}))
    if 'embed_dim' not in hyper_parameters:
        raise KeyError('Checkpoint hyper_parameters are missing embed_dim')

    model = MicroVQVAEModel(**hyper_parameters)
    incompatible = model.load_state_dict(checkpoint['state_dict'], strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(f'Missing required checkpoint keys: {incompatible.missing_keys}')

    model.eval()
    model.to(resolved_device)
    return model, hyper_parameters
