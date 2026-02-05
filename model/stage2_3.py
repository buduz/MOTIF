import numpy as np
import json
from pathlib import Path
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import T5EncoderModel, T5Tokenizer
from transformers.feature_extraction_utils import BatchFeature
from einops import rearrange, repeat
from typing import Dict, List

from model.stage1 import ActionMotifLearning
from model.modules.perceiver_resampler_with_mask import PerceiverResampler
from model.modules.flow_matching_action_head import FlowmatchingActionHead

log = logging.getLogger(__name__)

# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class MultimodalMotifPredictor(nn.Module):
    """Multimodal key learning model using text, vision and Perceiver fusion."""

    def __init__(
        self,
        vq_vae_ckpt: str = None,
        model_dim: int = 512,
        perceiver_depth: int = 6,
        perceiver_heads: int = 8,
        perceiver_dim_head: int = 64,
        perceiver_max_tokens: int = 1000,
    ) -> None:
        super().__init__()

        if vq_vae_ckpt is not None:
            vq_vae_cfg_path = Path(vq_vae_ckpt).parent / "config.json"
            with open(vq_vae_cfg_path, "r") as f:
                cfg = json.load(f)
            cfg = {k: v for k, v in cfg['models']['stage1'].items() if k != "class_name"}
            self.vq_vae = ActionMotifLearning(**cfg)
            checkpoint = torch.load(vq_vae_ckpt, map_location="cpu")
            self.vq_vae.load_state_dict(checkpoint["model"], strict=False)  # TODO: discriminator
            for param in self.vq_vae.parameters():
                param.requires_grad = False
            self.vq_vae.vq.code_restart = False
            log.info("Loaded VQVAE from %s", vq_vae_ckpt)

            # Perceiver
            code_dim = cfg["latent_dim"]
            ratio = cfg["window_size"] / cfg["downsample_factor"]

            if ratio <= 0 or not float(ratio).is_integer():
                raise ValueError("window_size / downsample_factor must be a positive integer.")

            num_latents = int(ratio)
            self.perceiver = PerceiverResampler(
                dim=model_dim,
                depth=perceiver_depth,
                dim_head=perceiver_dim_head,
                heads=perceiver_heads,
                max_tokens=perceiver_max_tokens,
                num_latents=num_latents,
            )

            self.to_code_dim = nn.Linear(model_dim, code_dim)

        # Vision encoder (DINO)
        dino_dim = 768
        self.dino_transform = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        self.dino_encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.dino_encoder.requires_grad_(False)
        self.vision_proj = nn.Linear(dino_dim, model_dim)

        # Text encoder (T5)
        t5_dim = 768
        self.text_encoder = T5EncoderModel.from_pretrained("t5-base")
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.text_encoder.requires_grad_(False)
        self.lang_proj = nn.Linear(t5_dim, model_dim)

    # Encode text
    def _encode_text(self, sentences: List[str], device = 'cuda'):
        encoding = self.tokenizer(sentences, return_tensors="pt", padding=True).to(device)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        with torch.no_grad():
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)

        return outputs.last_hidden_state, attention_mask

    # Encode images
    def _encode_image(self, images: torch.Tensor):
        b, v, t, c, h, w = images.shape
        images = rearrange(images, "b v t c h w -> (b t v) c h w")
        images = self.dino_transform(images)

        with torch.no_grad():
            feats = self.dino_encoder.forward_features(images)
            emb = feats["x_norm_patchtokens"]  # [(B*T*V), N, dino_dim]

        emb = rearrange(emb, "(b t v) n d -> b (t v n) d", b=b, t=t, v=v)
        mask = torch.ones(b, emb.shape[1], dtype=torch.int64, device=emb.device)
        return emb, mask

    # Encode observation (language + image)
    def _encode_observation(self, batch: Dict[str, torch.Tensor]):
        lang_emb, lang_mask = self._encode_text(batch["language"], device=batch["images"].device)
        img_emb, img_mask = self._encode_image(batch["images"])

        lang_emb = self.lang_proj(lang_emb)
        img_emb = self.vision_proj(img_emb)

        obs_seq = torch.cat([lang_emb, img_emb], dim=1)
        masks = torch.cat([lang_mask, img_mask], dim=1)
        return obs_seq, masks

    # Forward pass
    def forward(self, batch: Dict[str, torch.Tensor], causal_attn: bool = False):
        obs_seq, masks = self._encode_observation(batch)
        key_token = self.perceiver(obs_seq, masks)

        z_e = self.vq_vae.encoder(batch["rel_state"], batch["progress"], causal=causal_attn)
        key = self.to_code_dim(key_token)

        z, idx_e, dis_e = self.vq_vae.vq.inference_forward(z_e)
        z_k, idx_k, dis_k = self.vq_vae.vq.inference_forward(key)

        return F.mse_loss(z_e, key)

    # Inference: obtain key tokens
    @torch.no_grad()
    def get_key(self, batch: Dict[str, torch.Tensor], causal_attn: bool = False):
        obs_seq, masks = self._encode_observation(batch)
        key_token = self.perceiver(obs_seq, masks)
        return self.to_code_dim(key_token), obs_seq, masks


class MotifRoboticPolicy(nn.Module):
    """Multimodal key learning model using text, vision and Perceiver fusion."""

    def __init__(
        self,
        mkl_cfg: dict = None,
        mkl_ckpt: str = None,
        add_latent_action: bool = False,
        flow_matching_cfg: dict = None,
    ) -> None:
        super().__init__()

        if mkl_cfg is None:
            mkl_cfg = {}

        self.add_latent_action = add_latent_action
        if self.add_latent_action and mkl_ckpt is None:
            raise ValueError("Must provide mkl_ckpt if add_latent_action is True.")

        # MultimodalMotifPredictor (frozen)
        mkl_cfg = {k: v for k, v in mkl_cfg.items() if k != "class_name"}
        self.mkl = MultimodalMotifPredictor(**mkl_cfg)

        if mkl_ckpt is not None:
            checkpoint = torch.load(mkl_ckpt, map_location="cpu")
            missing, unexpected = self.mkl.load_state_dict(checkpoint["model"], strict=False)
            for k in missing:
                if not k.startswith(tuple(["vq_vae", "dino_encoder", "text_encoder"])):
                    raise ValueError(f"Missing key: {k}")
                if len(unexpected) > 0:
                    raise ValueError(f"Unexpected keys: {unexpected}")
            log.info(f"Loaded mkl_ckpt from {mkl_ckpt}")

        # Freeze stage2 modules
        for param in self.mkl.parameters():  
            param.requires_grad = False

        self.flow_matching_head = FlowmatchingActionHead(flow_matching_cfg)
        self.flow_matching_head.set_trainable_parameters(
            tune_projector=True,
            tune_diffusion_model=True,
        )

        dino_dim = 768
        self.vision_proj = nn.Linear(dino_dim, flow_matching_cfg['hidden_size'])
        t5_dim = 768
        self.lang_proj = nn.Linear(t5_dim, flow_matching_cfg['hidden_size'])

    def get_observation_embedding(self, batch: Dict[str, torch.Tensor]):
        lang_emb, lang_mask = self.mkl._encode_text(batch["language"], device=batch["images"].device)
        img_emb, img_mask = self.mkl._encode_image(batch["images"])

        lang_emb = self.lang_proj(lang_emb)
        img_emb = self.vision_proj(img_emb)

        obs_seq = torch.cat([lang_emb, img_emb], dim=1)
        masks = torch.cat([lang_mask, img_mask], dim=1)

        return obs_seq, masks
    
    def forward(self, batch: Dict[str, torch.Tensor], causal_attn: bool = False):
        obs_emb, masks = self.get_observation_embedding(batch)
        if self.add_latent_action:
            key, _, _ = self.mkl.get_key(batch, causal_attn)
            latent_action, _, _ = self.mkl.vq_vae.vq.inference_forward(key)
        else:
            latent_action = None
        action_head_input1 = BatchFeature(
            data={
                "backbone_features": obs_emb, 
                "backbone_attention_mask": ~masks.bool()    # mask is True for padding tokens
                }
        ) 
        action_head_input2 = BatchFeature(
            data={
                "embodiment_id": batch['embodiment_id'], 
                "state": batch['state'],
                "action": batch['action'],
                "action_mask": batch['action_mask'],
                "latent_action": latent_action,
                }
        ) 
        action_head_outputs = self.flow_matching_head(action_head_input1, action_head_input2)
        return action_head_outputs.loss

    @torch.no_grad()
    def get_action(self, batch: Dict[str, torch.Tensor], causal_attn: bool = False):
        obs_emb, masks = self.get_observation_embedding(batch)
        if self.add_latent_action:
            key, _, _ = self.mkl.get_key(batch, causal_attn)
            latent_action, _, _ = self.mkl.vq_vae.vq.inference_forward(key)
        else:
            latent_action = None
        action_head_input1 = BatchFeature(
            data={
                "backbone_features": obs_emb, 
                "backbone_attention_mask": ~masks.bool()    # mask is True for padding tokens
                }
        )
        action_head_input2 = BatchFeature(
            data={
                "embodiment_id": batch['embodiment_id'], 
                "state": batch['state'],
                "latent_action": latent_action,
                }
        ) 
        action_head_outputs = self.flow_matching_head.get_action(action_head_input1, action_head_input2)
        return action_head_outputs


MODEL_REGISTRY = {
    "MOTIFStage2": MultimodalMotifPredictor,
    "MOTIFStage3": MotifRoboticPolicy,
}

def build_model(model_cfg: dict):
    name = model_cfg.get("class_name")
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model class_name: {name}")
    kwargs = {k: v for k, v in model_cfg.items() if k != "class_name"}
    return MODEL_REGISTRY[name](**kwargs)

