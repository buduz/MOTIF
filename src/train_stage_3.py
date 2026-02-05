import os
import time
import json
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import torch
import wandb

from model.stage2_3 import build_model

from train_utils import (
    set_seed,
    build_dataset,
    build_loader,
    build_optim_sched,
    count_params_B,
    get_grad_norm
)

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../config", config_name="train_stage3")
def main(cfg: DictConfig):
    # Print/log the current complete configuration
    log.info("\n" + OmegaConf.to_yaml(cfg, resolve=True))

    set_seed(cfg.seed)
    device = cfg.device

    out_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Save configuration
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=4)

    # Wandb init
    wandb_run = None
    if cfg.get("wandb", {}).get("enable", False):
        wandb_mode = "offline" if cfg.wandb.get("offline", False) else "online"
        wandb_run = wandb.init(
            project=cfg.wandb.get("project", "stage3-train"),
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
    dataset.save_metadata(out_dir)
    loader = build_loader(dataset, cfg.dataset, device)

    # Build model/optimizer/scheduler/loss
    model = build_model(cfg.models.stage3).to(device)
    iters_per_epoch = (
        len(loader) if cfg.train.iters_per_epoch == 0 else cfg.train.iters_per_epoch
    )
    cfg.optim = cfg.optim.stage3
    optim, sched = build_optim_sched(
        model, cfg.optim, cfg.train.epochs, iters_per_epoch
    )

    # Count model parameters
    count_params_B(model)

    if wandb_run is not None and cfg.wandb.get("watch", False):
        wandb.watch(
            model,
            log=cfg.wandb.get("watch_log", "gradients"),
            log_freq=cfg.wandb.get("watch_freq", 100),
        )

    global_iters = 0

    # Train loop
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        total_loss, n_seen, iters = 0.0, 0, 0
        optim.zero_grad(set_to_none=True)
        t0 = time.time()

        grad_norm = 0.0

        for step, batch in enumerate(loader):
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device, non_blocking=True)

            raw_loss = model(batch, causal_attn=False)
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
                    "grad_norm": grad_norm,
                }
                log.info(json.dumps(log_payload, ensure_ascii=False))

                if wandb_run is not None:
                    wandb.log(
                        {
                            "train/loss_step": raw_loss.item(),
                            "train/lr": optim.param_groups[0]["lr"],
                            "train/grad_norm": grad_norm,
                            "train/epoch": epoch,
                        },
                        step=global_iters,
                    )

            if cfg.train.iters_per_epoch > 0 and iters >= cfg.train.iters_per_epoch:
                break

        if not cfg.optim["T_max_factor"]:
            sched.step()

        avg_loss = total_loss / max(1, n_seen)
        epoch_time = time.time() - t0

        epoch_payload = {
            "epoch": epoch,
            "avg_loss": avg_loss,
            "time": epoch_time,
            "lr": optim.param_groups[0]["lr"],
            "grad_norm": grad_norm,
        }
        log.info(json.dumps(epoch_payload, ensure_ascii=False))

        if wandb_run is not None:
            wandb.log(
                {
                    "train/loss_epoch": avg_loss,
                    "train/epoch_time": epoch_time,
                    "train/lr_epoch": optim.param_groups[0]["lr"],
                    "train/grad_norm_epoch": grad_norm,
                },
                step=global_iters,
            )

        # Save weights/checkpoints
        is_after_skip = epoch >= cfg.train.skip_epochs
        is_checkpoint_interval = epoch % cfg.train.ckpt_interval == 0
        is_last_epoch = epoch == cfg.train.epochs
        if is_after_skip and (is_checkpoint_interval or is_last_epoch):
            ckpt_path = os.path.join(out_dir, f"epoch_{epoch:04d}.pth")

            full_state = model.state_dict()
            exclude_prefixes = [
                "mkl.dino_encoder.",
                "mkl.text_encoder."
            ]
            filtered_state = {
                k: v for k, v in full_state.items()
                if not any(k.startswith(prefix) for prefix in exclude_prefixes)
            }
            torch.save(
                {
                    "epoch": epoch,
                    "model": filtered_state,
                    "cfg": OmegaConf.to_container(cfg, resolve=True),
                },
                ckpt_path,
            )
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