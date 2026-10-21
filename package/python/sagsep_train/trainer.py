from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from sagsep.model import PolicyModel, build_policy_network
from sagsep_train.checkpoint import (
    checkpoint_path,
    load_checkpoint,
    restore_training_state,
    save_checkpoint,
)
from sagsep_train.config import TrainingRecipe, recipe_to_dict, write_snapshot
from sagsep_train.data.collate import collator
from sagsep_train.data.dataset import (
    GlobalPermutationSampler,
    ShardIndex,
    ShardOrderSampler,
    TrainingDataset,
    make_dataloader,
    select_val_records,
    write_record_ids,
)
from sagsep_train.data.format import sha256_file
from sagsep_train.data.manifest import read_manifest

__all__ = [
    "SNAPSHOT_FILENAME",
    "VAL_RECORD_IDS_FILENAME",
    "TRAIN_LOG_FILENAME",
    "VAL_LOG_FILENAME",
    "EPOCH_LOG_FILENAME",
    "Trainer",
]

SNAPSHOT_FILENAME = "config_snapshot.yaml"
VAL_RECORD_IDS_FILENAME = "val_record_ids.json"
TRAIN_LOG_FILENAME = "train_log.jsonl"
VAL_LOG_FILENAME = "val_log.jsonl"
EPOCH_LOG_FILENAME = "epoch_log.jsonl"

_VAL_SEED_OFFSET = 1

_EPOCH_SEED_STRIDE = 1_000_003


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        json.dump(record, handle)
        handle.write("\n")


class Trainer:
    def __init__(
        self,
        recipe: TrainingRecipe,
        *,
        run_dir: str | os.PathLike[str],
        recipe_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self.recipe = recipe
        self.recipe_path = Path(recipe_path).resolve() if recipe_path is not None else None
        self._base_dir = self.recipe_path.parent if self.recipe_path else Path.cwd()
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)

        data = recipe.data
        trainer_cfg = recipe.trainer

        self.dataset_dir = self._resolve(data.dataset_dir)
        manifest_path = self.dataset_dir / data.manifest
        if not manifest_path.is_file():
            raise FileNotFoundError(f"dataset manifest not found: {manifest_path}")
        self.manifest = read_manifest(manifest_path)
        self.manifest_sha256 = sha256_file(manifest_path)

        if not data.train_shards:
            raise ValueError("data.train_shards: at least one training shard is required")
        self.train_index = ShardIndex(self.dataset_dir, data.train_shards)
        self.val_index = (
            ShardIndex(self.dataset_dir, data.val_shards) if data.val_shards else None
        )

        self.val_record_ids: list[int] = []
        if self.val_index is not None:
            self.val_record_ids = select_val_records(
                len(self.val_index),
                data.val_limit,
                seed=int(trainer_cfg.seed) + _VAL_SEED_OFFSET,
            )
            write_record_ids(self.run_dir / VAL_RECORD_IDS_FILENAME, self.val_record_ids)

        self.snapshot_path = self.run_dir / SNAPSHOT_FILENAME
        if not self.snapshot_path.exists():
            write_snapshot(
                recipe,
                self.snapshot_path,
                extra={
                    "dataset_manifest_sha256": self.manifest_sha256,
                    "val_record_ids_file": VAL_RECORD_IDS_FILENAME,
                },
            )
            print(f"[trainer] wrote {self.snapshot_path}")

        self.seed = int(trainer_cfg.seed)
        self._seed_everything(self.seed)
        self.deterministic = bool(trainer_cfg.deterministic)
        if self.deterministic:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.net = build_policy_network(recipe.model.model_config_dict())
        probe = PolicyModel.from_network(self.net, device=str(trainer_cfg.device))
        self.device = torch.device(probe.device)
        del probe
        print(f"[trainer] device={self.device} run_dir={self.run_dir}")

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self.collate_fn = collator(
            support_rule=data.support_rule,
            cross_depot=data.partial_connect_cross_depot,
        )
        pin_memory = bool(data.pin_memory) and self.device.type == "cuda"

        self.train_dataset = TrainingDataset(self.train_index)
        if data.shuffle_mode == "global":
            self.train_sampler: Any = GlobalPermutationSampler(
                len(self.train_dataset), seed=self.seed
            )
        else:
            self.train_sampler = ShardOrderSampler(
                self.train_dataset.shard_sizes(), seed=self.seed
            )
        self.train_loader = make_dataloader(
            self.train_dataset,
            batch_size=data.batch_size,
            sampler=self.train_sampler,
            num_workers=data.num_workers,
            drop_last=bool(data.drop_last),
            pin_memory=pin_memory,
            collate_fn=self.collate_fn,
            seed=self.seed,
        )

        self.val_dataset: TrainingDataset | None = None
        self.val_loader = None
        if self.val_index is not None and self.val_record_ids:
            self.val_dataset = TrainingDataset(self.val_index, self.val_record_ids)
            self.val_loader = make_dataloader(
                self.val_dataset,
                batch_size=data.eval_batch_size,
                sampler=list(range(len(self.val_dataset))),
                num_workers=data.num_workers,
                drop_last=False,
                pin_memory=pin_memory,
                collate_fn=self.collate_fn,
                seed=self.seed,
            )

        self.global_step = 0
        self.start_epoch = 0
        self.epochs_completed = 0
        self.best_val_loss: float | None = None
        self.train_log_path = self.run_dir / TRAIN_LOG_FILENAME
        self.val_log_path = self.run_dir / VAL_LOG_FILENAME
        self.epoch_log_path = self.run_dir / EPOCH_LOG_FILENAME
        self._latest_saved_step: int | None = None

        self._wandb = None
        self._init_wandb()

        if trainer_cfg.resume_from:
            self._resume(self._resolve(trainer_cfg.resume_from))

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------

    def _resolve(self, value: str | os.PathLike[str]) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self._base_dir / path).resolve()

    def _seed_everything(self, seed: int) -> None:
        random.seed(int(seed))
        np.random.seed(int(seed) % (2**32))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    def _epoch_seed(self, epoch: int) -> int:
        return (self.seed * _EPOCH_SEED_STRIDE + int(epoch)) % (2**31 - 1)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        cfg = self.recipe.optimizer
        kwargs = {
            "lr": float(cfg.lr),
            "betas": tuple(float(value) for value in cfg.betas),
            "eps": float(cfg.eps),
            "weight_decay": float(cfg.weight_decay),
        }
        name = str(cfg.name).lower()
        if name == "adam":
            return torch.optim.Adam(self.net.parameters(), **kwargs)
        if name == "adamw":
            return torch.optim.AdamW(self.net.parameters(), **kwargs)
        raise ValueError(f"unsupported optimizer {cfg.name!r}")

    def _build_scheduler(self) -> Any | None:
        cfg = self.recipe.scheduler
        name = str(cfg.name).strip().lower()
        if not name or name == "none":
            return None
        if name == "cosineannealingwarmrestarts":
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=int(cfg.T_0),
                T_mult=int(cfg.T_mult),
                eta_min=float(cfg.eta_min),
            )
        raise ValueError(f"unsupported scheduler {cfg.name!r}")

    def _init_wandb(self) -> None:
        if not bool(self.recipe.logging.wandb):
            return
        try:
            import wandb
        except ImportError:
            print("[trainer] wandb not installed; continuing without experiment tracking.")
            return
        log_cfg = self.recipe.logging
        self._wandb = wandb
        wandb.init(
            project=log_cfg.project or None,
            name=log_cfg.run_name or None,
            tags=list(log_cfg.tags) or None,
            config=recipe_to_dict(self.recipe),
            reinit=True,
        )

    def _resume(self, path: Path) -> None:
        payload = load_checkpoint(path)
        self.global_step, self.start_epoch = restore_training_state(
            payload,
            model=self.net,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
        self.epochs_completed = self.start_epoch
        self.net.to(self.device)
        print(
            f"[trainer] resumed from {path} at step {self.global_step}, "
            f"epoch {self.start_epoch} (mid-epoch continuation is not reproduced)"
        )

    # ------------------------------------------------------------------
    # logging
    # ------------------------------------------------------------------

    def _log(self, path: Path, record: Mapping[str, Any]) -> None:
        if bool(self.recipe.logging.jsonl):
            _append_jsonl(path, record)
        if self._wandb is not None:
            self._wandb.log(dict(record), step=self.global_step)

    # ------------------------------------------------------------------
    # loss
    # ------------------------------------------------------------------

    def _batch_loss(self, batch: Any) -> tuple[torch.Tensor, int]:
        logits = self.net.forward_batch(batch.edge_index, batch.node_attr, batch.edge_attr)
        keep = ~batch.depot_mask
        flat = logits.view(-1)[keep]
        target = batch.target[keep]
        return F.binary_cross_entropy_with_logits(flat, target), int(flat.numel())

    # ------------------------------------------------------------------
    # checkpoints
    # ------------------------------------------------------------------

    def _save(self, kind: str, *, extra: Mapping[str, Any] | None = None) -> Path:
        path = save_checkpoint(
            checkpoint_path(self.run_dir, kind),
            model=self.net,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            global_step=self.global_step,
            epoch=int(self.epochs_completed),
            config=recipe_to_dict(self.recipe),
            extra=dict(extra) if extra else None,
        )
        if kind == "latest":
            self._latest_saved_step = self.global_step
        return path

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------

    def evaluate(self, loader: Any = None) -> dict[str, float]:
        loader = self.val_loader if loader is None else loader
        if loader is None:
            raise ValueError("evaluate: no validation loader (data.val_shards is empty)")

        was_training = bool(self.net.training)
        self.net.eval()
        total = 0.0
        batches = 0
        n_logits = 0
        try:
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(self.device)
                    loss, count = self._batch_loss(batch)
                    if count == 0:
                        continue
                    total += float(loss.item())
                    batches += 1
                    n_logits += count
        finally:
            self.net.train(was_training)

        return {
            "loss": total / batches if batches else float("nan"),
            "batches": batches,
            "logits": n_logits,
        }

    # ------------------------------------------------------------------
    # the loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        cfg = self.recipe.trainer
        max_steps = cfg.max_steps
        log_every = max(int(cfg.log_every), 1)
        stopped_on_steps = False

        for epoch in range(int(self.start_epoch), int(cfg.max_epochs)):
            torch.manual_seed(self._epoch_seed(epoch))
            if hasattr(self.train_sampler, "set_epoch"):
                self.train_sampler.set_epoch(epoch)

            epoch_start = time.time()
            epoch_loss = 0.0
            epoch_batches = 0
            window_time = 0.0
            window_steps = 0
            window_graphs = 0
            window_nodes = 0

            for batch in self.train_loader:
                step_t0 = time.perf_counter()
                self.net.train()
                self.optimizer.zero_grad()
                batch = batch.to(self.device)
                loss, _count = self._batch_loss(batch)
                loss.backward()
                if cfg.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.net.parameters(), float(cfg.grad_clip_norm)
                    )
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)

                step_dt = time.perf_counter() - step_t0
                loss_value = float(loss.item())
                epoch_loss += loss_value
                epoch_batches += 1
                window_time += step_dt
                window_steps += 1
                window_graphs += int(batch.n_replicas)
                window_nodes += int(batch.num_nodes)
                self.global_step += 1

                if self.global_step % log_every == 0:
                    lr = float(self.optimizer.param_groups[0]["lr"])
                    record = {
                        "step": self.global_step,
                        "epoch": epoch,
                        "loss": loss_value,
                        "lr": lr,
                        "step_ms": (window_time / window_steps * 1000.0) if window_steps else 0.0,
                        "graphs": window_graphs,
                        "nodes": window_nodes,
                    }
                    self._log(self.train_log_path, record)
                    print(
                        f"[train] step={self.global_step} epoch={epoch} "
                        f"loss={loss_value:.4f} lr={lr:.2e} "
                        f"step={record['step_ms']:.1f}ms graphs={window_graphs}"
                    )
                    window_time = 0.0
                    window_steps = 0
                    window_graphs = 0
                    window_nodes = 0

                if (
                    self.val_loader is not None
                    and int(cfg.eval_every) > 0
                    and self.global_step % int(cfg.eval_every) == 0
                ):
                    self._validate(epoch=epoch, trigger="step")

                if int(cfg.checkpoint_every) > 0 and self.global_step % int(cfg.checkpoint_every) == 0:
                    self._save("latest")

                if max_steps is not None and self.global_step >= int(max_steps):
                    self._save("latest")
                    print(f"[trainer] max_steps={int(max_steps)} reached at step {self.global_step}")
                    stopped_on_steps = True
                    break

            if stopped_on_steps:
                break

            self.epochs_completed = epoch + 1
            duration = time.time() - epoch_start
            record = {
                "epoch": epoch,
                "step": self.global_step,
                "batches": epoch_batches,
                "mean_loss": epoch_loss / epoch_batches if epoch_batches else float("nan"),
                "duration_s": duration,
            }
            self._log(self.epoch_log_path, record)
            print(
                f"[epoch] {epoch} finished in {duration:.1f}s | "
                f"mean loss {record['mean_loss']:.4f} over {epoch_batches} batches"
            )

            if self.val_loader is not None and int(cfg.eval_every) <= 0:
                self._validate(epoch=epoch, trigger="epoch")

        if self._latest_saved_step != self.global_step:
            self._save("latest")
        print("[trainer] training complete.")

    def _validate(self, *, epoch: int, trigger: str) -> dict[str, float]:
        metrics = self.evaluate()
        record = {
            "timestamp": datetime.now().isoformat(),
            "step": self.global_step,
            "epoch": epoch,
            "trigger": trigger,
            **metrics,
        }
        self._log(self.val_log_path, record)
        print(
            f"[val] step={self.global_step} epoch={epoch} "
            f"loss={metrics['loss']:.4f} over {metrics['batches']} batches"
        )
        if self.best_val_loss is None or metrics["loss"] < self.best_val_loss:
            self.best_val_loss = float(metrics["loss"])
            self._save("best", extra={"val_loss": self.best_val_loss})
            print(f"[checkpoint] new best validation loss {self.best_val_loss:.4f}")
        return metrics
