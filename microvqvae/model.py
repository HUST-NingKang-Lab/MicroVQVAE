from typing import Dict, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model]
        L = x.size(1)
        x = x + self.pe[:, :L, :]
        return self.dropout(x)


class TransformerEncoderStack(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            layer_norm_eps=layer_norm_eps,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model, eps=layer_norm_eps))

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, L, d_model]; pad_mask: [B, L], True=pad
        return self.encoder(x, src_key_padding_mask=pad_mask)


class VectorQuantizer(nn.Module):
    """
    Standard VQ (no EMA) with straight-through estimator.
    Computes codebook loss and commitment loss; supports mask to ignore padding.
    """
    def __init__(self, num_codes: int, code_dim: int, commitment_beta: float = 0.25, eps: float = 1e-8):
        super().__init__()
        self.codebook = nn.Embedding(num_codes, code_dim)
        self.code_dim = code_dim
        self.num_codes = num_codes
        self.commitment_beta = commitment_beta
        self.eps = eps
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)

    @torch.no_grad()
    def _perplexity(self, indices: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # indices: [N], valid_mask: [N] or None
        if valid_mask is not None:
            indices = indices[valid_mask]
        if indices.numel() == 0:
            return torch.tensor(0.0, device=self.codebook.weight.device)
        counts = torch.bincount(indices, minlength=self.num_codes).float()
        probs = counts / (counts.sum() + self.eps)
        perp = torch.exp(-(probs * (probs + self.eps).log()).sum())
        return perp

    def forward(self, z_e: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        z_e: [B, L, C], mask: [B, L], 1=valid, 0=pad
        Returns:
          - z_q: quantized (no straight-through), [B, L, C]
          - z_q_st: straight-through quantized, [B, L, C]
          - indices: [B, L] long
          - vq_loss, codebook_loss, commitment_loss, perplexity
        """
        B, L, C = z_e.shape
        assert C == self.code_dim, f"Expected code_dim={self.code_dim}, got {C}"
        flat = z_e.reshape(-1, C)  # [N, C], N=B*L
        if mask is not None:
            flat_mask = mask.reshape(-1).bool()  # [N]

        # Compute distances to codebook: ||z - e||^2 = z^2 - 2 z·e + e^2
        e_weight = self.codebook.weight  # [K, C]
        z_norm = (flat ** 2).sum(dim=1, keepdim=True)           # [N,1]
        e_norm = (e_weight ** 2).sum(dim=1).unsqueeze(0)        # [1,K]
        distances = z_norm - 2 * flat @ e_weight.T + e_norm     # [N,K]

        indices = torch.argmin(distances, dim=1)  # [N]
        z_q = F.embedding(indices, e_weight)      # [N,C]

        # Mask handling: zero-out quantized vectors on padding to avoid leakage
        if mask is not None:
            z_q = z_q.masked_fill(~flat_mask.unsqueeze(1), 0.0)

        z_q = z_q.view(B, L, C)
        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        # Losses with mask
        if mask is not None:
            m = mask.unsqueeze(-1).float()  # [B,L,1]
            valid = m.sum().clamp_min(1.0)
            codebook_loss = ((z_q.detach() - z_e) ** 2 * m).sum() / valid
            commitment_loss = ((z_q - z_e.detach()) ** 2 * m).sum() / valid
            valid_mask_flat = flat_mask
        else:
            codebook_loss = F.mse_loss(z_q.detach(), z_e)
            commitment_loss = F.mse_loss(z_q, z_e.detach())
            valid_mask_flat = None

        vq_loss = codebook_loss + self.commitment_beta * commitment_loss
        with torch.no_grad():
            perplexity = self._perplexity(indices, valid_mask_flat)

        return {
            "z_q": z_q,
            "z_q_st": z_q_st,
            "indices": indices.view(B, L),
            "vq_loss": vq_loss,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "perplexity": perplexity,
        }


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self,
        num_codes: int,
        code_dim: int,
        commitment_beta: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
        restart_unused: bool = True,
        restart_threshold: float = 1.0,
        use_cosine: bool = True,
        use_gumbel: bool = True,
        gumbel_tau_start: float = 1.0,
        gumbel_tau_end: float = 0.1,
        gumbel_warmup_steps: int = 10000,
        code_dropout_rate: float = 0.05,
        normalize_after_ema: bool = True,
        quantize_with_soft: bool = False,
        ema_use_soft: bool = False,
        min_step_before_restart: int = 100,
    ):
        super().__init__()
        self.codebook = nn.Embedding(num_codes, code_dim)
        self.code_dim = code_dim
        self.num_codes = num_codes
        self.commitment_beta = commitment_beta
        self.decay = decay
        self.eps = eps
        self.restart_unused = restart_unused
        self.restart_threshold = restart_threshold
        self.use_cosine = use_cosine
        self.use_gumbel = use_gumbel
        self.gumbel_tau_start = gumbel_tau_start
        self.gumbel_tau_end = gumbel_tau_end
        self.gumbel_warmup_steps = gumbel_warmup_steps
        self.code_dropout_rate = code_dropout_rate
        self.normalize_after_ema = normalize_after_ema
        self.quantize_with_soft = quantize_with_soft
        self.ema_use_soft = ema_use_soft
        self.min_step_before_restart = min_step_before_restart

        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)
        self.register_buffer("ema_count", torch.zeros(num_codes))
        self.register_buffer("ema_weight", torch.zeros(num_codes, code_dim))
        self.register_buffer("step", torch.zeros((), dtype=torch.long))

    def _current_tau(self) -> torch.Tensor:
        if self.gumbel_warmup_steps <= 0:
            return torch.as_tensor(self.gumbel_tau_end, device=self.codebook.weight.device)
        t = torch.minimum(self.step.float(), torch.tensor(float(self.gumbel_warmup_steps), device=self.step.device))
        r = t / float(self.gumbel_warmup_steps)
        return self.gumbel_tau_start * (self.gumbel_tau_end / self.gumbel_tau_start) ** r

    @torch.no_grad()
    def _perplexity(self, indices: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if valid_mask is not None:
            indices = indices[valid_mask]
        if indices.numel() == 0:
            return torch.tensor(0.0, device=self.codebook.weight.device)
        counts = torch.bincount(indices, minlength=self.num_codes).float()
        probs = counts / (counts.sum() + self.eps)
        return torch.exp(-(probs * (probs + self.eps).log()).sum())

    def forward(self, z_e: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, L, C = z_e.shape
        assert C == self.code_dim
        flat = z_e.reshape(-1, C)  # [N, C]
        if mask is not None:
            flat_mask = mask.reshape(-1).bool()  # [N]
        else:
            flat_mask = torch.ones(flat.size(0), dtype=torch.bool, device=flat.device)

        e_weight = self.codebook.weight  # [K, C]

        if self.use_cosine:
            flat_s = F.normalize(flat, dim=1, eps=1e-8)
            e_s = F.normalize(e_weight, dim=1, eps=1e-8)
            logits = flat_s @ e_s.T
        else:
            z_norm = (flat ** 2).sum(dim=1, keepdim=True)
            e_norm = (e_weight ** 2).sum(dim=1).unsqueeze(0)
            distances = z_norm - 2 * flat @ e_weight.T + e_norm
            logits = -distances

        if self.training and self.code_dropout_rate > 0.0:
            K = logits.size(1)
            drop = torch.rand(K, device=logits.device) < self.code_dropout_rate
            if drop.all():
                drop[torch.randint(0, K, (1,), device=logits.device)] = False
            logits = logits.masked_fill(drop.unsqueeze(0), -1e9)

        # Clamp logits and remove NaN or Inf values before softmax.
        logits = torch.clamp(torch.nan_to_num(logits, nan=0.0, neginf=-1e9, posinf=1e9), min=-60.0, max=60.0)

        if self.training and self.use_gumbel:
            U = torch.rand_like(logits).clamp_(1e-6, 1 - 1e-6)
            g = -torch.log(-torch.log(U))
            tau = self._current_tau()
            probs_soft = F.softmax((logits + g) / tau, dim=1)
        else:
            probs_soft = F.softmax(logits, dim=1)
        hard_idx = logits.argmax(dim=1)
        probs_hard = F.one_hot(hard_idx, num_classes=self.num_codes).type_as(flat)

        probs_soft = probs_soft * flat_mask.unsqueeze(1).float()
        probs_hard = probs_hard * flat_mask.unsqueeze(1).float()

        probs_for_quant = probs_soft if (self.training and self.quantize_with_soft) else probs_hard
        row_sums = probs_for_quant.sum(dim=1, keepdim=True).clamp_min(1e-8)
        probs_for_quant = probs_for_quant / row_sums

        z_q_flat = probs_for_quant @ e_weight
        z_q_flat = torch.nan_to_num(z_q_flat)
        z_q = z_q_flat.view(B, L, C).masked_fill((~flat_mask).view(B, L, 1), 0.0)

        z_q_st = z_e + (z_q - z_e).detach()

        with torch.no_grad():
            probs_for_ema = probs_soft if self.ema_use_soft else probs_hard
            assign_sum = probs_for_ema.sum(dim=0)               # [K]
            dw = probs_for_ema.T @ flat                         # [K,C]

            self.ema_count.mul_(self.decay).add_(assign_sum, alpha=1.0 - self.decay)
            self.ema_weight.mul_(self.decay).add_(dw, alpha=1.0 - self.decay)

            n = self.ema_count + self.eps
            embed = self.ema_weight / n.unsqueeze(1)
            embed = torch.nan_to_num(embed)

            if self.training:
                self.codebook.weight.data.copy_(embed)

            if self.training and self.restart_unused and (int(self.step.item()) >= self.min_step_before_restart):
                dead = self.ema_count < self.restart_threshold
                if dead.any():
                    candidates = flat[flat_mask]
                    if candidates.numel() > 0:
                        num_dead = int(dead.sum().item())
                        idx = torch.randint(0, candidates.size(0), (num_dead,), device=candidates.device)
                        new_embeds = candidates[idx]
                    else:
                        new_embeds = torch.randn((int(dead.sum().item()), C), device=embed.device) * 0.1
                    self.codebook.weight.data[dead] = new_embeds
                    self.ema_weight.data[dead] = new_embeds
                    self.ema_count.data[dead] = self.restart_threshold

            if self.training and self.use_cosine and self.normalize_after_ema:
                self.codebook.weight.data = F.normalize(self.codebook.weight.data, dim=1, eps=1e-8)

        m = mask.unsqueeze(-1).float() if mask is not None else torch.ones(B, L, 1, device=z_e.device)
        valid = m.sum().clamp_min(1.0)
        commitment_loss = ((z_q - z_e.detach()) ** 2 * m).sum() / valid
        vq_loss = self.commitment_beta * commitment_loss

        with torch.no_grad():
            perplexity = self._perplexity(hard_idx, flat_mask)
            self.step.add_(1)

        return {
            "z_q": z_q,
            "z_q_st": z_q_st,
            "indices": hard_idx.view(B, L),
            "vq_loss": vq_loss,
            "codebook_loss": torch.tensor(0.0, device=z_e.device),
            "commitment_loss": commitment_loss,
            "perplexity": perplexity,
        }


class DVAEMaskedTransformer(pl.LightningModule):
    """
    dVAE for genomic context:
      - Encoder: TransformerEncoder over x (protein embeddings) with mask
      - VectorQuantizer: per-position discrete token
      - Decoder: TransformerEncoder over quantized tokens with mask
      - Loss: masked MSE + VQ loss
    Batch format: {'x': [B,L,E], 'mask':[B,L], 'genome_id':..., 'start':...}
    1 in mask = valid position; 0 = padding; -1 = manually masked (to be replaced with mask embedding)
    """
    def __init__(
        self,
        embed_dim: int,
        d_model: int = 512,
        nhead: int = 8,
        num_enc_layers: int = 6,
        num_dec_layers: int = 6,
        codebook_size: int = 8192,
        code_dim: int = 256,
        dropout: float = 0.1,
        ff_mult: int = 4,
        commitment_beta: float = 0.25,
        vq_loss_weight: float = 1.0,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        vq_decay: float = 0.99,
        usage_entropy_weight: float = 0.0,
        mse_loss_weight: float = 1.0,
        cos_loss_weight: float = 1.0,
        proto_bypass_weight: float = 1.0,
        quant_noise_std: float = 0.05,
        diversity_loss_weight: float = 0.0,
        vq_use_gumbel: bool = True,
        vq_gumbel_tau_start: float = 1.0,
        vq_gumbel_tau_end: float = 0.1,
        vq_gumbel_warmup_steps: int = 10000,
        vq_code_dropout_rate: float = 0.05,
        vq_normalize_after_ema: bool = True,
        vq_warmup_steps: int = 2000,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.input_proj = nn.Linear(embed_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)
        self.encoder = TransformerEncoderStack(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_enc_layers,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
        )

        self.enc_norm = nn.LayerNorm(d_model)
        self.pre_vq = nn.Sequential(
            nn.Linear(d_model, code_dim),
            nn.LayerNorm(code_dim),
        )
        self.vq = VectorQuantizerEMA(
            codebook_size,
            code_dim,
            commitment_beta=commitment_beta,
            decay=vq_decay,
            restart_unused=True,
            restart_threshold=1.0,
            use_cosine=True,
            use_gumbel=self.hparams.vq_use_gumbel,
            gumbel_tau_start=self.hparams.vq_gumbel_tau_start,
            gumbel_tau_end=self.hparams.vq_gumbel_tau_end,
            gumbel_warmup_steps=self.hparams.vq_gumbel_warmup_steps,
            code_dropout_rate=self.hparams.vq_code_dropout_rate,
            normalize_after_ema=self.hparams.vq_normalize_after_ema,
            quantize_with_soft=False,
            ema_use_soft=False,
            min_step_before_restart=10,

        )
        self.vq_loss_weight = vq_loss_weight

        self.lr = lr
        self.weight_decay = weight_decay

        self.post_vq = nn.Linear(code_dim, d_model)
        self.decoder_pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)
        self.decoder = TransformerEncoderStack(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_dec_layers,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
        )
        self.output_proj = nn.Linear(d_model, embed_dim)
        
        # mask embedding
        self.mask_embedding = nn.Embedding(1, embed_dim)

        # Prototype bypass
        self.code_to_out = nn.Linear(code_dim, embed_dim)

    def forward(self, batch: Dict[str, torch.Tensor], use_st: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward for training/inference.
        Returns:
          - x_hat: reconstructed embeddings [B,L,E]
          - indices: [B,L] long (discrete tokens)
          - vq_loss, codebook_loss, commitment_loss, perplexity
        """
        x = batch["x"]  # [B,L,E]
        mask = batch["mask"]  # [B,L], 1=valid
        pad_mask = (mask == 0)
        manual_mask = (mask == -1)
        if manual_mask.any():
            x[manual_mask] = self.mask_embedding.weight[0]

        # Encoder
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h, pad_mask=pad_mask)  # [B,L,d_model]
        h = self.enc_norm(h)

        # To code space and quantize
        z_e = self.pre_vq(h)  # [B,L,code_dim]

        if self.training and self.hparams.quant_noise_std > 0.0:
            noise = torch.randn_like(z_e) * self.hparams.quant_noise_std
            noise = noise.masked_fill((batch["mask"] == 0).unsqueeze(-1), 0.0)
            z_e = z_e + noise
        z_e = torch.nan_to_num(z_e)

        vq_out = self.vq(z_e, mask=mask)
        z_q = vq_out["z_q_st"] if use_st else vq_out["z_q"]  # [B,L,code_dim]

        # Decoder
        d = self.post_vq(z_q)
        d = self.decoder_pos_enc(d)
        d = self.decoder(d, pad_mask=pad_mask)  # [B,L,d_model]
        base = self.output_proj(d)
        proto = self.code_to_out(z_q)
        x_hat = base + self.hparams.proto_bypass_weight * proto
        x_hat = torch.nan_to_num(x_hat)

        return {
            "x_hat": x_hat,
            "indices": vq_out["indices"],
            "vq_loss": vq_out["vq_loss"],
            "codebook_loss": vq_out["codebook_loss"],
            "commitment_loss": vq_out["commitment_loss"],
            "perplexity": vq_out["perplexity"],
        }
    
    @torch.no_grad()
    def get_encoder_attn_maps(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Return encoder self-attention maps for analysis.
        x = batch["x"]
        mask = batch["mask"]
        pad_mask = (mask == 0)
        h = self.input_proj(x)
        h = self.pos_enc(h)
        attn = []
        for layer in self.encoder.encoder.layers:
            h, a = layer.self_attn(h, h, h, key_padding_mask=pad_mask, need_weights=True)
            attn.append(a)  # [B, num_heads, L, L]
            h = layer.norm1(layer.dropout1(h) + h)
            h2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(h))))
            h = layer.norm2(layer.dropout2(h2) + h)
        return torch.stack(attn, dim=0)  # [num_layers, B, num_heads, L, L]
    
    @torch.no_grad()
    def get_decoder_attn_maps(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Return decoder self-attention maps for analysis.
        x = batch["x"]
        mask = batch["mask"]
        pad_mask = (mask == 0)
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h, pad_mask=pad_mask)
        h = self.enc_norm(h)
        z_e = self.pre_vq(h)
        vq_out = self.vq(z_e, mask=mask)
        z_q = vq_out["z_q_st"]
        d = self.post_vq(z_q)
        d = self.decoder_pos_enc(d)
        attn = []
        for layer in self.decoder.encoder.layers:
            d, a = layer.self_attn(d, d, d, key_padding_mask=pad_mask, need_weights=True)
            attn.append(a)  # [B, num_heads, L, L]
            d = layer.norm1(layer.dropout1(d) + d)
            d2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(d))))
            d = layer.norm2(layer.dropout2(d2) + d)
        return torch.stack(attn, dim=0)  # [num_layers, B, num_heads, L, L]
            

    @staticmethod
    def masked_mse(x_hat: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x_hat,x: [B,L,E]; mask: [B,L] (1=valid)
        m = mask.unsqueeze(-1).float()  # [B,L,1]
        diff2 = (x_hat - x) ** 2 * m
        denom = (m.sum() * x.size(-1)).clamp_min(1.0)
        return diff2.sum() / denom

    @staticmethod
    def masked_cosine_loss(x_hat: torch.Tensor, x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        # Compute cosine loss only on valid positions to avoid padding artifacts.
        B, L, E = x.shape
        m = mask.reshape(-1).bool()              # [B*L]
        if m.sum() == 0:
            return x_hat.new_zeros(())
        xh = x_hat.reshape(-1, E)[m]
        xt = x.reshape(-1, E)[m]
        if xh.numel() == 0:
            return x_hat.new_zeros(())
        xh = F.normalize(xh, dim=-1, eps=eps)
        xt = F.normalize(xt, dim=-1, eps=eps)
        cos = (xh * xt).sum(dim=-1)              # [N_valid]
        return (1.0 - cos).mean()

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        out = self.forward(batch, use_st=True)
        x = batch['label'] if 'label' in batch else batch['x']
        
        if not torch.isfinite(out["x_hat"]).all():
            out["x_hat"] = torch.nan_to_num(out["x_hat"])
        # only compute losses on manually masked positions (-1 in mask)
        recon_mask = (batch["mask"] != 0).float()
        mse = self.masked_mse(out["x_hat"], x, recon_mask)
        cos = self.masked_cosine_loss(out["x_hat"], x, recon_mask)
        recon_loss = self.hparams.mse_loss_weight * mse + self.hparams.cos_loss_weight * cos

        # Warm up the VQ loss to avoid over-constraining the encoder early.
        if self.trainer is not None:
            step = self.global_step
        else:
            step = 0
        w = float(min(1.0, step / max(1, self.hparams.vq_warmup_steps)))
        loss = recon_loss + (w * self.vq_loss_weight) * out["vq_loss"]

        # Usage entropy regularization.
        if self.hparams.usage_entropy_weight > 0.0:
            valid = batch["mask"].bool()
            idx = out["indices"][valid]
            if idx.numel() > 0:
                K = self.hparams.codebook_size
                counts = torch.bincount(idx, minlength=K).float()
                probs = counts / (counts.sum() + 1e-8)
                entropy = -(probs * (probs + 1e-8).log()).sum()
                norm_entropy = entropy / math.log(K)
                usage_loss = (1.0 - norm_entropy)
                loss = loss + self.hparams.usage_entropy_weight * usage_loss
                self.log("train/usage_entropy", norm_entropy.detach(), on_step=True, on_epoch=True)

        # Codebook diversity regularization.
        if self.hparams.diversity_loss_weight > 0.0:
            div_loss = self.codebook_diversity_loss(used_only=True) * self.hparams.diversity_loss_weight
            loss = loss + div_loss
            self.log("train/div_loss", div_loss.detach(), on_step=True, on_epoch=True)

        # Log a diagnostic flag if the loss becomes NaN.
        if torch.isnan(loss):
            self.log("debug/nan_step", torch.tensor(1.0, device=loss.device), on_step=True, prog_bar=True)

        # Track codebook usage statistics.
        valid = batch["mask"].bool()
        idx = out["indices"][valid]
        if idx.numel() > 0:
            K = self.hparams.codebook_size
            counts = torch.bincount(idx, minlength=K).float()
            top1_frac = (counts.max() / counts.sum()).clamp_min(0.0)
            num_used = (counts > 0).sum()
            self.log("train/top1_frac", top1_frac, on_step=True, on_epoch=True)
            self.log("train/num_used", num_used, on_step=True, on_epoch=True)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/recon_loss", recon_loss, on_step=True, on_epoch=True)
        self.log("train/mse", mse, on_step=True, on_epoch=True)
        self.log("train/cos", cos, on_step=True, on_epoch=True)
        self.log("train/vq_loss", out["vq_loss"], on_step=True, on_epoch=True)
        self.log("train/perplexity", out["perplexity"], on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        out = self.forward(batch, use_st=True)
        recon_mask = (batch["mask"] != 0).float()
        x = batch['label'] if 'label' in batch else batch['x']
        mse = self.masked_mse(out["x_hat"], x, recon_mask)
        cos = self.masked_cosine_loss(out["x_hat"], x, recon_mask)
        recon_loss = self.hparams.mse_loss_weight * mse + self.hparams.cos_loss_weight * cos
        loss = recon_loss + self.vq_loss_weight * out["vq_loss"]

        # Measure reconstruction consistency for the quantized path.
        with torch.no_grad():
            x_hat_codes = self.decode_tokens(out["indices"], batch["mask"])
            q_path_mse = self.masked_mse(x_hat_codes, x, batch["mask"])
        self.log("val/q_path_mse", q_path_mse, on_epoch=True)

        if self.hparams.diversity_loss_weight > 0.0:
            self.log("val/div_loss", self.codebook_diversity_loss(used_only=True).detach(), on_epoch=True)

        if self.hparams.usage_entropy_weight > 0.0:
            valid = batch["mask"].bool()
            idx = out["indices"][valid]
            if idx.numel() > 0:
                K = self.hparams.codebook_size
                counts = torch.bincount(idx, minlength=K).float()
                probs = counts / (counts.sum() + 1e-8)
                entropy = -(probs * (probs + 1e-8).log()).sum()
                norm_entropy = entropy / math.log(K)
                self.log("val/usage_entropy", norm_entropy.detach(), on_epoch=True)

        # Track codebook usage statistics.
        valid = batch["mask"].bool()
        idx = out["indices"][valid]
        if idx.numel() > 0:
            K = self.hparams.codebook_size
            counts = torch.bincount(idx, minlength=K).float()
            top1_frac = (counts.max() / counts.sum()).clamp_min(0.0)
            num_used = (counts > 0).sum()
            self.log("val/top1_frac", top1_frac, on_epoch=True)
            self.log("val/num_used", num_used, on_epoch=True)

        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.log("val/recon_loss", recon_loss, on_epoch=True)
        self.log("val/mse", mse, on_epoch=True)
        self.log("val/cos", cos, on_epoch=True)
        self.log("val/vq_loss", out["vq_loss"], on_epoch=True)
        self.log("val/perplexity", out["perplexity"], on_epoch=True)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=(0.9, 0.95))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=200)
        return {"optimizer": optim, "lr_scheduler": scheduler}
    
    @torch.no_grad()
    def get_encoder_embeddings(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Get encoder continuous embeddings before VQ.
        Args:
          x: [B,L,E], mask: [B,L]
        Returns:
          z_e: [B,L,code_dim]
        """
        self.eval()
        pad_mask = (mask == 0)
        manual_mask = (mask == -1)
        enc_pad_mask = pad_mask | manual_mask
        
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h, pad_mask=enc_pad_mask)
        h = self.enc_norm(h)
        z_e = self.pre_vq(h)
        return z_e

    @torch.no_grad()
    def encode_tokens(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Encode to discrete token indices.
        Args:
          x: [B,L,E], mask: [B,L]
        Returns:
          indices: [B,L] long
        """
        self.eval()
        pad_mask = (mask == 0)
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h, pad_mask=pad_mask)
        h = self.enc_norm(h)
        z_e = self.pre_vq(h)
        vq_out = self.vq(z_e, mask=mask)
        return vq_out["indices"]

    @torch.no_grad()
    def decode_tokens(self, indices: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Decode from discrete token indices to embeddings.
        Args:
          indices: [B,L] long
          mask: [B,L]
        Returns:
          x_hat: [B,L,E]
        """
        self.eval()
        z_q = F.embedding(indices, self.vq.codebook.weight)  # [B,L,code_dim]
        z_q = z_q.masked_fill((mask == 0).unsqueeze(-1), 0.0)
        d = self.post_vq(z_q)
        d = self.decoder_pos_enc(d)
        d = self.decoder(d, pad_mask=(mask == 0))
        base = self.output_proj(d)
        proto = self.code_to_out(z_q)
        x_hat = base + self.hparams.proto_bypass_weight * proto
        return x_hat

    @torch.no_grad()
    def lookup_codebook(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Look up codebook embeddings for discrete token indices.
        Args:
          indices: arbitrary-shaped integer tensor of token IDs
        Returns:
          codebook embeddings with an added trailing code_dim axis
        """
        self.eval()
        return F.embedding(indices, self.vq.codebook.weight)

    def codebook_diversity_loss(self, used_only: bool = True, min_used: int = 2) -> torch.Tensor:
        # Normalize codebook vectors with eps and optionally restrict to used codes.
        W = self.vq.codebook.weight              # [K,C]
        if used_only and hasattr(self.vq, "ema_count") and self.vq.ema_count is not None:
            used = (self.vq.ema_count > 1e-6)
            if used.sum() < min_used:
                return W.new_zeros(())
            W = W[used]
        if W.size(0) < 2:
            return W.new_zeros(())
        Wn = F.normalize(W, dim=1, eps=1e-8)
        G = Wn @ Wn.T                            # [k_used,k_used]
        I = torch.eye(G.size(0), device=G.device, dtype=G.dtype)
        off = G - I
        return (off.pow(2).sum() / (G.numel() - G.size(0)))


MicroVQVAEModel = DVAEMaskedTransformer
