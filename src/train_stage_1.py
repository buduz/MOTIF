import os
import time
import json
import logging
import math
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import torch
import torch.nn.functional as F
import wandb

from model.stage1 import build_model, compute_cross_robot_infonce, EmbodimentDiscriminator, LabelRemapper
from data.embodiment_tags import EMBODIMENT_TAG_MAPPING

from train_utils import (
    set_seed,
    build_dataset,
    build_loader,
    build_optim_sched,
    count_params_B,
    get_grad_norm
)

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../config", config_name="train_stage1")
def main(cfg: DictConfig):
    # Print/Log the full current configuration
    log.info("\n" + OmegaConf.to_yaml(cfg, resolve=True))

    set_seed(cfg.seed)
    device = cfg.device

    out_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(out_dir, exist_ok=True)

    code_usage_path = os.path.join(out_dir, "code_usage")
    os.makedirs(code_usage_path, exist_ok=True)

    # Save configuration
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=4)

    # >>> wandb init
    wandb_run = None
    if cfg.get("wandb", {}).get("enable", False):
        wandb_mode = "offline" if cfg.wandb.get("offline", False) else "online"
        wandb_run = wandb.init(
            project=cfg.wandb.get("project", "stage1-train"),
            name=cfg.wandb.get("name", os.path.basename(out_dir)),
            tags=cfg.wandb.get("tags", None),
            notes=cfg.wandb.get("notes", None),
            dir=out_dir,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=wandb_mode,
            resume="allow",
        )
        if cfg.wandb.get("log_code", True):
            wandb.run.log_code(".")

    # Build dataset & loader
    dataset = build_dataset(cfg.dataset)
    loader = build_loader(dataset, cfg.dataset, device)

    # Build model/optimizer/scheduler/loss
    model = build_model(cfg.models.stage1).to(device)

    # Initialize discriminator
    if cfg.get("use_adversarial", False):
        discriminator = EmbodimentDiscriminator(
            input_dim=cfg.models.stage1.latent_dim, 
            num_robots=cfg.get("num_robots", 2)
        ).to(device)
        model.discriminator = discriminator
        valid_ids = [EMBODIMENT_TAG_MAPPING[tag] for tag in cfg.dataset['embodiment_tag']]
        remapper = LabelRemapper(valid_ids, device=device)

    iters_per_epoch = (
        len(loader) if cfg.train.iters_per_epoch == 0 else cfg.train.iters_per_epoch
    )
    cfg.optim = cfg.optim.stage1
    
    optim, sched = build_optim_sched(
        model, cfg.optim, cfg.train.epochs, iters_per_epoch
    )

    # Count model parameters
    count_params_B(model)

    # >>> wandb watch (optional)
    if wandb_run is not None and cfg.wandb.get("watch", False):
        wandb.watch(
            model,
            log=cfg.wandb.get("watch_log", "gradients"),  # "gradients" or "all"
            log_freq=cfg.wandb.get("watch_freq", 100),
        )

    global_iters = 0

    # train loop
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        total_loss, n_seen, iters = 0.0, 0, 0
        optim.zero_grad(set_to_none=True)
        t0 = time.time()

        grad_norm = 0.0 
        epoch_counts = torch.zeros(
            model.codebook_size, device=device, dtype=torch.float
        )

        infonce_epoch_loss_sum = 0.0
        infonce_epoch_n = 0  

        for step, batch in enumerate(loader):
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device, non_blocking=True)

            out = model(batch, causal_attn=False)
            x_hat, z_e, z, indices = (
                out["recon"],
                out["z_e"],
                out["z"],
                out["idx"],
            )

            target = batch["rel_state"]
            outputs = model.compute_loss(x_hat, target, z_e, z)

            # ====== 1. Task/Language InfoNCE======
            infonce_loss = None
            if cfg.add_info_NCE:
                infonce_loss, _ = compute_cross_robot_infonce(
                    z_e=out["z_e"],
                    languages=batch["language"],
                    progress=batch["progress"],
                    tau=0.1,
                    sigma=0.02,
                    hard_delta=cfg.get("hard_delta", None)
                )
                outputs["loss"] += cfg.lambda_infonce * infonce_loss
                outputs.update({"infonce_loss": infonce_loss})

            # ====== 2. Embodiment Adversarial Loss ======
            if cfg.get("use_adversarial", False) and hasattr(model, 'discriminator'):
                p = float(global_iters) / (cfg.train.epochs * len(loader))
                alpha = 2.0 / (1.0 + math.exp(-10 * p)) - 1
                disc_input = z_e.reshape(-1, z_e.size(-1))
                disc_logits = model.discriminator(disc_input, alpha=alpha) # [N, num_robots]
                # Generate labels
                embod_ids = batch["embodiment_id"].long()
                embod_ids = remapper(embod_ids)
                target_ids = embod_ids.unsqueeze(1).repeat(1, z_e.size(1)).view(-1)
                
                # Compute CrossEntropy
                adv_loss = F.cross_entropy(disc_logits, target_ids)
                
                # Weight and log
                lambda_adv = cfg.get("lambda_adv", 0.05)
                outputs["loss"] += lambda_adv * adv_loss
                
                outputs["adv_loss"] = adv_loss.detach()
                
                # Compute and log discriminator accuracy
                acc = (disc_logits.argmax(dim=1) == target_ids).float().mean()
                outputs["adv_acc"] = acc.detach()

            # Compute code usage
            code_usage = model.compute_code_usage(indices)

            perplexity_step = model.compute_perplexity(indices)
            topk_step = model.compute_topk_usage(indices, ks=(10, 20))
            with torch.no_grad():
                epoch_counts += model._counts_from_indices(indices)
            raw_loss = outputs["loss"]
            loss = raw_loss / max(1, cfg.train.grad_accum_steps)
            loss.backward()

            if (step + 1) % cfg.train.grad_accum_steps == 0:
                if cfg.train.grad_clip_norm and cfg.train.grad_clip_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.train.grad_clip_norm
                    ).item()
                else:
                    grad_norm = get_grad_norm(model)

                optim.step()
                optim.zero_grad(set_to_none=True)

                if cfg.optim["T_max_factor"]: 
                    sched.step()
                iters += 1
                global_iters += 1

            bs = batch["state"].shape[0]
            total_loss += raw_loss.item() * bs
            n_seen += bs

            if cfg.add_info_NCE and infonce_loss is not None:
                with torch.no_grad():
                    infonce_epoch_loss_sum += float(infonce_loss) * bs
                    infonce_epoch_n += bs

            if (
                cfg.train.log_interval > 0
                and iters % max(1, cfg.train.log_interval) == 0
            ):
                log_payload = {
                    "epoch": epoch,
                    "iter": iters,
                    "global_iter": global_iters,
                    "lr": optim.param_groups[0]["lr"],
                    "loss": raw_loss.item(),
                    "recon_loss": outputs["recon_loss"].item(),
                    "vq_loss": outputs["vq_loss"].item(),
                    "code_usage": code_usage.item(),
                    "perplexity": perplexity_step.item(),
                    "top10_frac": topk_step["top10_frac"].item(),
                    "top20_frac": topk_step["top20_frac"].item(),
                    "grad_norm": grad_norm,
                }

                if cfg.add_info_NCE and infonce_loss is not None:
                    log_payload["infonce_loss"] = float(infonce_loss)

                if "adv_loss" in outputs:
                    log_payload["adv_loss"] = float(outputs["adv_loss"])
                    log_payload["adv_acc"] = float(outputs["adv_acc"])

                log.info(json.dumps(log_payload, ensure_ascii=False))

                # >>> wandb step log
                if wandb_run is not None:
                    wandb_step_payload = {
                        "train/loss_step": raw_loss.item(),
                        "train/recon_loss_step": outputs["recon_loss"].item(),
                        "train/vq_loss_step": outputs["vq_loss"].item(),
                        "train/code_usage_step": code_usage.item(),
                        "train/perplexity_step": perplexity_step.item(),
                        "train/top10_frac_step": topk_step["top10_frac"].item(),
                        "train/top20_frac_step": topk_step["top20_frac"].item(),
                        "train/grad_norm": grad_norm,
                        "train/lr": optim.param_groups[0]["lr"],
                        "train/epoch": epoch,
                    }

                    if cfg.add_info_NCE and infonce_loss is not None:
                        wandb_step_payload["train/infonce_loss_step"] = float(infonce_loss)

                    if "adv_loss" in outputs:
                        wandb_step_payload["train/adv_loss_step"] = float(outputs["adv_loss"])
                        wandb_step_payload["train/adv_acc_step"] = float(outputs["adv_acc"])

                    wandb.log(wandb_step_payload, step=global_iters)

            if cfg.train.iters_per_epoch > 0 and iters >= cfg.train.iters_per_epoch:
                break

        if not cfg.optim["T_max_factor"]:
            sched.step()

        avg_loss = total_loss / max(1, n_seen)
        epoch_time = time.time() - t0
        with torch.no_grad():
            total_epoch = epoch_counts.sum().clamp_min(1.0)
            p_epoch = epoch_counts / total_epoch
            entropy_epoch = -(p_epoch * (p_epoch + 1e-10).log()).sum()
            perplexity_epoch = entropy_epoch.exp()

            sorted_counts_epoch, _ = torch.sort(epoch_counts, descending=True)
            top10_frac_epoch = sorted_counts_epoch[:10].sum() / total_epoch
            top20_frac_epoch = sorted_counts_epoch[:20].sum() / total_epoch

        epoch_payload = {
            "epoch": epoch,
            "avg_loss": avg_loss,
            "time": epoch_time,
            "lr": optim.param_groups[0]["lr"],
            "grad_norm": grad_norm,
            "perplexity_epoch": perplexity_epoch.item(),
            "top10_frac_epoch": top10_frac_epoch.item(),
            "top20_frac_epoch": top20_frac_epoch.item(),
        }

        if cfg.add_info_NCE and infonce_epoch_n > 0:
            epoch_payload["infonce_loss_epoch"] = infonce_epoch_loss_sum / infonce_epoch_n

        log.info(json.dumps(epoch_payload, ensure_ascii=False))

        # >>> wandb epoch log
        if wandb_run is not None:
            wandb_epoch_payload = {
                "train/loss_epoch": avg_loss,
                "train/epoch_time": epoch_time,
                "train/lr_epoch": optim.param_groups[0]["lr"],
                "train/grad_norm_epoch": grad_norm,
                "train/perplexity_epoch": perplexity_epoch.item(),
                "train/top10_frac_epoch": top10_frac_epoch.item(),
                "train/top20_frac_epoch": top20_frac_epoch.item(),
            }

            if cfg.add_info_NCE and infonce_epoch_n > 0:
                wandb_epoch_payload["train/infonce_loss_epoch"] = infonce_epoch_loss_sum / infonce_epoch_n

            wandb.log(wandb_epoch_payload, step=global_iters)

        # Visualize code usage
        model.plot_and_reset_code_usage(epoch, code_usage_path)

        # Save weights
        if epoch % cfg.train.ckpt_interval == 0 or (epoch == cfg.train.epochs and epoch % cfg.train.ckpt_interval != 0):
            ckpt_path = os.path.join(out_dir, f"epoch_{epoch:04d}.pth")
            state_dict = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optim.state_dict(),
                "scheduler": sched.state_dict(),
                "cfg": OmegaConf.to_container(cfg, resolve=True),
            }

            torch.save(state_dict, ckpt_path)
            log.info(f"=> saved {ckpt_path}")

            if wandb_run is not None and cfg.wandb.get("log_ckpt", False):
                artifact = wandb.Artifact(
                    name=f"ckpt-epoch-{epoch:04d}",
                    type="model",
                    metadata={"epoch": epoch},
                )
                artifact.add_file(ckpt_path)
                wandb.log_artifact(artifact)

    if wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()