from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch

from .fasta import ProteinRecord


def _resolve_device(device: str) -> torch.device:
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


def _resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
    if dtype == 'auto':
        if device.type == 'cuda':
            return torch.float16
        return torch.float32
    mapping = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    if dtype not in mapping:
        raise ValueError(f'Unsupported dtype: {dtype}')
    resolved = mapping[dtype]
    if device.type == 'cpu' and resolved != torch.float32:
        return torch.float32
    return resolved


@dataclass
class PairESMEmbedder:
    model_name_or_path: str
    device: torch.device
    dtype: torch.dtype
    tokenizer: object
    model: torch.nn.Module

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, device: str = 'auto', dtype: str = 'auto') -> 'PairESMEmbedder':
        from transformers import AutoModel, AutoTokenizer

        resolved_device = _resolve_device(device)
        resolved_dtype = _resolve_dtype(dtype, resolved_device)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = model.to(resolved_device)
        if resolved_device.type == 'cuda':
            model = model.to(resolved_dtype)
        model.eval()
        return cls(
            model_name_or_path=model_name_or_path,
            device=resolved_device,
            dtype=resolved_dtype,
            tokenizer=tokenizer,
            model=model,
        )

    def embed_records(self, records: List[ProteinRecord], batch_size: int = 32, max_length: int = 1024) -> torch.Tensor:
        embeddings: List[torch.Tensor] = []
        with torch.inference_mode():
            for start in range(0, len(records), batch_size):
                batch = records[start:start + batch_size]
                sequences = [item.sequence for item in batch]
                tokenized = self.tokenizer(
                    sequences,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_attention_mask=True,
                )
                input_ids = tokenized['input_ids'].to(self.device)
                attention_mask = tokenized['attention_mask'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                hidden = outputs.last_hidden_state
                mask = attention_mask.unsqueeze(-1).expand_as(hidden)
                masked_hidden = hidden * mask
                summed = masked_hidden.sum(dim=1)
                counts = mask.sum(dim=1).clamp_min(1)
                pooled = summed / counts
                embeddings.append(pooled.detach().to(torch.float32).cpu())
        return torch.cat(embeddings, dim=0)
