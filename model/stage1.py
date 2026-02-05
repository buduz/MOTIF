import os
from typing import Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from model.modules.blocks import (
    BasicTransformerBlock,
    VectorQuantizer,
    ProgressPosEnc,
    ResidualTemporalBlock, 
    DeConv1dBlock
)

import matplotlib.pyplot as plt

# Encoder: Local Attention Transformer + Downsampling
class LocalTransformerEncoder(nn.Module):
    def __init__(
        self,
        state_dim: int = 10,
        model_dim: int = 256,
        latent_dim: int = 64,
        depth_high: int = 2,
        depth_low: int = 0, 
        num_heads: int = 8,
        dropout: float = 0.1,
        local_w: int = 12,
        use_sinusoidal_pe: bool = False,
        max_len: int = 2048,
        downsample_factor: int = 1,
    ):
        super().__init__()
        self.local_w = local_w
        self.downsample_factor = max(1, int(downsample_factor))

        self.in_proj = nn.Linear(state_dim, model_dim)
        self.pos_emb_enc = ProgressPosEnc(d_model=model_dim)

        self.blocks_high = nn.ModuleList([
            BasicTransformerBlock(
                dim=model_dim,
                num_attention_heads=num_heads,
                attention_head_dim=model_dim // num_heads,
                dropout=dropout,
                positional_embeddings="sinusoidal" if use_sinusoidal_pe else None,
                num_positional_embeddings=max_len if use_sinusoidal_pe else None,
            )
            for _ in range(depth_high)
        ])

        if self.downsample_factor > 1:
            assert math.log2(self.downsample_factor).is_integer(), \
                "downsample_factor must be a power of 2"

        if self.downsample_factor == 1:
            strides = [1, 1]
            kernel_sizes = [3, 2]
        else:
            n = int(math.log2(self.downsample_factor))
            strides = [2] * n + [1]
            kernel_sizes = [5] + [3] * n
            if len(strides) == 1:
                kernel_sizes = [3, 2]
                strides = [1, 1]

        self.conv_down = ResidualTemporalBlock(
            inp_channels=model_dim,
            out_channels=model_dim,
            kernel_size=kernel_sizes,
            stride=strides,
            n_groups=8,
            causal=False,
            residual=False,
            pooling_layers=[],
        )

        self.blocks_low = nn.ModuleList([
            BasicTransformerBlock(
                dim=model_dim,
                num_attention_heads=num_heads,
                attention_head_dim=model_dim // num_heads,
                dropout=dropout,
                positional_embeddings="sinusoidal" if use_sinusoidal_pe else None,
                num_positional_embeddings=max_len if use_sinusoidal_pe else None,
            )
            for _ in range(depth_low)
        ])

        self.to_latent = nn.Linear(model_dim, latent_dim)

    def forward(
        self,
        x: torch.Tensor,           # [B, H, state_dim]
        progress: torch.Tensor = None,  # [B, H] in [0,1]
        causal: bool = False,
    ) -> torch.Tensor:
        B, H, _ = x.shape
        h = self.in_proj(x)  # [B, H, D]

        # add progress PE
        if progress is not None:
            if progress.dim() == 3:
                p = progress.view(B, H)
            else:
                p = progress  # [B, H]
            pos_full = self.pos_emb_enc(p)  # [B, H, D]
            h = h + pos_full

        if len(self.blocks_high) > 0:
            band_mask_bool = build_sliding_window_mask(H, self.local_w, causal=causal, device=h.device)
            attn_mask_high = to_attn_mask_for_torch(band_mask_bool)
            for blk in self.blocks_high:
                h = blk(hidden_states=h, attention_mask=attn_mask_high)

        h = self.conv_down(h)
        H_ds = h.size(1)

        if len(self.blocks_low) > 0:
            band_mask_bool_low = build_sliding_window_mask(H_ds, self.local_w, causal=causal, device=h.device)
            attn_mask_low = to_attn_mask_for_torch(band_mask_bool_low)
            for blk in self.blocks_low:
                h = blk(hidden_states=h, attention_mask=attn_mask_low)

        z_e = self.to_latent(h)  # [B, H_ds, latent_dim]
        return z_e


# Decoder: Upsampling + Local Attention Transformer
class LocalTransformerDecoder(nn.Module):
    def __init__(
        self,
        state_dim: int = 10,
        model_dim: int = 256,
        latent_dim: int = 64,
        depth_high: int = 4,
        depth_low: int = 0,
        num_heads: int = 8,
        dropout: float = 0.1,
        local_w: int = 12,
        use_sinusoidal_pe: bool = True,
        max_len: int = 2048,
        downsample_factor: int = 1,
    ):
        super().__init__()
        self.local_w = local_w
        self.downsample_factor = max(1, int(downsample_factor))

        self.from_latent = nn.Linear(latent_dim, model_dim)
        self.pos_emb_dec = ProgressPosEnc(d_model=model_dim)

        self.blocks_low = nn.ModuleList([
            BasicTransformerBlock(
                dim=model_dim,
                num_attention_heads=num_heads,
                attention_head_dim=model_dim // num_heads,
                dropout=dropout,
                positional_embeddings="sinusoidal" if use_sinusoidal_pe else None,
                num_positional_embeddings=max_len if use_sinusoidal_pe else None,
            )
            for _ in range(depth_low)
        ])

        if self.downsample_factor > 1:
            self.deconv = DeConv1dBlock(
                inp_channels=model_dim,
                out_channels=model_dim,
                kernel_size=4,
                stride=self.downsample_factor,
                n_groups=8,
                causal=False,
            )
        else:
            self.deconv = None

        self.blocks_high = nn.ModuleList([
            BasicTransformerBlock(
                dim=model_dim,
                num_attention_heads=num_heads,
                attention_head_dim=model_dim // num_heads,
                dropout=dropout,
                positional_embeddings="sinusoidal" if use_sinusoidal_pe else None,
                num_positional_embeddings=max_len if use_sinusoidal_pe else None,
            )
            for _ in range(depth_high)
        ])

        self.to_state = nn.Linear(model_dim, state_dim)

    def forward(
        self,
        z_q: torch.Tensor,            # [B, H_ds, latent_dim]
        progress: torch.Tensor = None,  # [B, H]
        causal: bool = False,
    ) -> torch.Tensor:
        B, H_ds, _ = z_q.shape
        h = self.from_latent(z_q)  # [B, H_ds, D]

        if len(self.blocks_low) > 0:
            band_mask_bool_low = build_sliding_window_mask(H_ds, self.local_w, causal=causal, device=h.device)
            attn_mask_low = to_attn_mask_for_torch(band_mask_bool_low)
            for blk in self.blocks_low:
                h = blk(hidden_states=h, attention_mask=attn_mask_low)

        if progress is not None:
            if progress.dim() == 3:
                p = progress.view(B, -1)
            else:
                p = progress  # [B, H]
            H = p.size(1)
        else:
            if self.downsample_factor > 1:
                H = H_ds * self.downsample_factor
            else:
                H = H_ds
            p = None

        if self.deconv is not None:
            h_t = h.transpose(1, 2)           # [B, D, H_ds]
            h_up_t = self.deconv(h_t)         # [B, D, H_up]
            h = h_up_t.transpose(1, 2)        # [B, H_up, D]
        H_up = h.size(1)

        if H_up != H:
            h = F.interpolate(
                h.transpose(1, 2), size=H, mode="linear", align_corners=False
            ).transpose(1, 2)
        else:
            H = H_up

        if p is not None:
            pos_dec = self.pos_emb_dec(p)  # [B, H, D]
            # h = h + pos_dec

        if len(self.blocks_high) > 0:
            band_mask_bool_high = build_sliding_window_mask(H, self.local_w, causal=causal, device=h.device)
            attn_mask_high = to_attn_mask_for_torch(band_mask_bool_high)
            for blk in self.blocks_high:
                h = blk(hidden_states=h, attention_mask=attn_mask_high)

        x_hat = self.to_state(h)  # [B, H, state_dim]
        return x_hat

# Action motif Autoencoder
class ActionMotifLearning(nn.Module):
    def __init__(
        self,
        state_dim: int = 10,
        model_dim: int = 256,
        latent_dim: int = 64,
        codebook_size: int = 512,
        window_size: int = 16,
        depth_enc: int = 4,          
        depth_dec: int = 4,          
        num_heads: int = 8,
        dropout: float = 0.1,
        local_w: int = 12,
        use_sinusoidal_pe: bool = True,
        max_len: int = 2048,
        vq_beta: float = 0.25,
        downsample_factor: int = 1,   
        depth_enc_low: int = 0,      
        depth_dec_low: int = 0,      
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.vq_beta = vq_beta
        self.window_size = window_size
        self.downsample_factor = max(1, int(downsample_factor))

        self.encoder = LocalTransformerEncoder(
            state_dim=state_dim,
            model_dim=model_dim,
            latent_dim=latent_dim,
            depth_high=depth_enc,
            depth_low=depth_enc_low,
            num_heads=num_heads,
            dropout=dropout,
            local_w=local_w,
            use_sinusoidal_pe=use_sinusoidal_pe,
            max_len=max_len,
            downsample_factor=self.downsample_factor,
        )

        self.vq = VectorQuantizer(
            num_latents=codebook_size,
            latent_dim=latent_dim,
            code_restart=True,
        )

        self.decoder = LocalTransformerDecoder(
            state_dim=state_dim,
            model_dim=model_dim,
            latent_dim=latent_dim,
            depth_high=depth_dec,
            depth_low=depth_dec_low,
            num_heads=num_heads,
            dropout=dropout,
            local_w=local_w,
            use_sinusoidal_pe=use_sinusoidal_pe,
            max_len=max_len,
            downsample_factor=self.downsample_factor,
        )

    def forward(self, batch: Dict[str, torch.Tensor], causal_attn: bool = False):
        """
        batch: dict(progress=[B,H], rel_state=[B,H,state_dim])
        """
        progress, state_seq = batch["progress"], batch["rel_state"]
        # encoder
        z_e = self.encoder(state_seq, progress, causal=causal_attn)   # [B, H_ds, latent_dim]
        # VQ
        z_q, z, x_vq_in, indices, distance = self.vq(z_e)
        # decoder
        state_seq_hat = self.decoder(z_q, progress, causal=causal_attn)  # [B, H, state_dim]
        return dict(recon=state_seq_hat, z_e=z_e, z_q=z_q, idx=indices, z=z)

    def compute_loss(
        self,
        x_hat: torch.Tensor,
        x: torch.Tensor,
        z_e: torch.Tensor,
        z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        rec_loss = ((x_hat - x) ** 2).mean()
        q_loss = ((z_e.detach() - z) ** 2).mean()
        commit_loss = ((z_e - z.detach()) ** 2).mean()

        loss = rec_loss + q_loss + self.vq_beta * commit_loss
        return {
            "loss": loss,
            "recon_loss": rec_loss,
            "vq_loss": q_loss,
        }

    def compute_code_usage(self, indices) -> float:
        unique, counts = torch.unique(indices, return_counts=True)
        index_counts = torch.zeros(
            self.codebook_size, dtype=torch.long, device=unique.device
        )
        index_counts[unique] = counts
        code_usage = (index_counts != 0).float().mean()
        return code_usage

    def _counts_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        indices: [B, H_ds] or any shape of code indices
        return: counts [K]
        """
        flat = indices.reshape(-1)
        counts = torch.bincount(flat, minlength=self.codebook_size).float()
        return counts

    def compute_perplexity(self, indices: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        """
        perplexity = exp( -sum p log p )
        """
        counts = self._counts_from_indices(indices)
        total = counts.sum().clamp_min(1.0)
        p = counts / total
        entropy = -(p * (p + eps).log()).sum()
        perplexity = entropy.exp()
        return perplexity

    def compute_topk_usage(self, indices: torch.Tensor, ks=(10, 20)) -> Dict[str, torch.Tensor]:
        """
        calculates the fraction of the codebook that is used in the top k most frequent codes
        """
        counts = self._counts_from_indices(indices)
        total = counts.sum().clamp_min(1.0)
        sorted_counts, _ = torch.sort(counts, descending=True)
        out = {}
        for k in ks:
            k = min(k, self.codebook_size)
            frac = sorted_counts[:k].sum() / total
            out[f"top{k}_frac"] = frac
        return out

    def plot_and_reset_code_usage(self, epoch: int, path):
        """
        plots the usage distribution of the codebook and resets the usage
        """
        plot_usage_distribution(self.vq.usage, os.path.join(path, f"train_usage_unsorted_epoch{epoch:03d}.png"))
        plot_usage_distribution(self.vq.usage.sort().values, os.path.join(path, f"train_usage_sorted_epoch{epoch:03d}.png"))
        self.vq.random_restart()
        self.vq.reset_usage()

class GradientReversal(Function):
    """
    GRL Layer: Forward propagation remains unchanged, backward propagation reverses the gradient (multiplied by -alpha).
    Makes the Encoder's update direction "maximize the Discriminator's Loss".
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            # Reverse gradient
            grad_input = - ctx.alpha * grad_output
        return grad_input, None

class EmbodimentDiscriminator(nn.Module):
    """
    MLP attempts to distinguish robot_id from latent z_e
    """
    def __init__(self, input_dim, num_robots=2, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, num_robots)
        )

    def forward(self, z, alpha=1.0):
        # 1. Gradient reversal
        z_rev = GradientReversal.apply(z, alpha)
        # 2. Classification
        logits = self.net(z_rev)
        return logits
    
class LabelRemapper:
    def __init__(self, valid_ids, device='cpu'):
        """
        valid_ids: list, e.g., [0, 1, 3]
        """
        self.device = device
        max_id = max(valid_ids)
        
        self.table = torch.full((max_id + 1,), -1, dtype=torch.long, device=device)
        
        for new_id, old_id in enumerate(valid_ids):
            self.table[old_id] = new_id
            
    def __call__(self, targets):
        return self.table[targets]

def compute_cross_robot_infonce(
    z_e: torch.Tensor,          # [B, H_ds, D]
    languages: list,            # len=B, each is a str
    progress: torch.Tensor,     # [B, 32] or [B, H]
    tau: float = 0.1,
    sigma: float = 0.02,
    hard_delta: float = None,
    eps: float = 1e-8,
):
    """
    Multi-positive InfoNCE:
      positives: same language & close progress (soft weight or hard window)
      negatives: all others in batch

    Returns:
      loss (scalar), stats dict
    """
    device = z_e.device
    B = z_e.size(0)

    # 1) pool latent to one embedding per sample
    emb = z_e.mean(dim=1)                 # [B, D]
    emb = F.normalize(emb, dim=-1)        # cosine space

    # 2) similarity matrix
    sim = emb @ emb.t()                  # [B, B]
    sim = sim / tau

    # mask out self
    eye = torch.eye(B, device=device).bool()
    sim = sim.masked_fill(eye, -1e9)

    # 3) build positive weights W_ij
    # same language matrix
    lang_eq = torch.zeros((B, B), device=device, dtype=torch.bool)
    for i in range(B):
        for j in range(B):
            if i != j and languages[i] == languages[j]:
                lang_eq[i, j] = True

    # progress distance
    if progress.dim() == 2:
        p0 = progress[:, 0]              # [B]
    else:
        # if [B, H, ...] accidentally
        p0 = progress[:, 0]

    dp = (p0[:, None] - p0[None, :]).abs()  # [B, B]

    if hard_delta is not None:
        prog_close = dp < hard_delta
        W = (lang_eq & prog_close).float()
    else:
        # soft gaussian weight
        W = lang_eq.float() * torch.exp(-(dp / sigma) ** 2)

    # ensure no self-positive
    W = W.masked_fill(eye, 0.0)

    # 4) multi-positive InfoNCE
    # numerator_i = sum_j W_ij * exp(sim_ij)
    exp_sim = torch.exp(sim)  # [B,B]
    num = (W * exp_sim).sum(dim=1)  # [B]
    den = exp_sim.sum(dim=1) + eps  # [B]

    # only keep anchors that have any positives
    has_pos = (W.sum(dim=1) > 0)
    if has_pos.any():
        loss = -torch.log((num[has_pos] + eps) / den[has_pos]).mean()
    else:
        loss = torch.tensor(0.0, device=device)

    stats = {
        "infonce_loss": loss.detach(),
        "pos_per_anchor": W.sum(dim=1).mean().detach(),
        "valid_anchor_frac": has_pos.float().mean().detach(),
        "avg_dp_pos": (dp * (W > 0)).sum() / ((W > 0).sum() + eps),
    }
    return loss, stats


# sliding-window attention mask
def build_sliding_window_mask(H: int, w: int, causal: bool = False, device=None):
    """
    Returns a boolean mask [H, H]; True = allow attention; False = mask out.
    - Non-causal: |i - j| < w//2 + (w % 2)
    - Causal: j <= i AND (i - j) < w
    """
    i = torch.arange(H, device=device).unsqueeze(1)  # [H,1]
    j = torch.arange(H, device=device).unsqueeze(0)  # [1,H]
    if causal:
        mask = (j <= i) & ((i - j) < w)
    else:
        left = (w - 1) // 2
        right = w - 1 - left
        mask = ((j - i) >= -left) & ((j - i) <= right)
    return mask  # [H,H] bool (True=keep)


def to_attn_mask_for_torch(mask_bool: torch.Tensor):
    """
    PyTorch MHA Tensor mask: True means 'do not attend'.
    """
    return ~mask_bool


def plot_usage_distribution(usage, saved_path):
    data = usage.cpu().numpy()
    n = 1
    for n in range(1, 10):
        if (2 ** n) ** 2 <= len(data) < (2 ** (n + 1)) ** 2:
            break
    data = data.reshape(2 ** n, -1)
    fig, ax = plt.subplots()
    cax = ax.matshow(data, interpolation="nearest")
    fig.colorbar(cax)
    plt.axis("off")
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(saved_path, bbox_inches="tight", pad_inches=0.0)
    plt.close()


MODEL_REGISTRY = {
    "MOTIFStage1": ActionMotifLearning,
}

def build_model(model_cfg: dict):
    name = model_cfg.get("class_name")
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model class_name: {name}")
    kwargs = {k: v for k, v in model_cfg.items() if k != "class_name"}
    return MODEL_REGISTRY[name](**kwargs)
