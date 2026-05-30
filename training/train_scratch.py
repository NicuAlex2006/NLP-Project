"""
Training Loop for the LLaMA-Style Tool-Augmented Transformer.

Key changes from original:
  - Cross-platform device router (CUDA / MPS / CPU)
  - Device-aware AMP (bfloat16 on CUDA, float16 autocast on MPS, float32 on CPU)
  - Updated hyperparams per council consensus: lr=3e-4, warmup=2000, wd=0.05, dropout=0.1
  - Token embedding resize for tool-aware special tokens
  - 2048 context window

Usage:
    python training/train_scratch.py
    python training/train_scratch.py --test
"""

import os
import sys
import time
import json
import math
import torch
import torch.nn as nn
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.model import (
    ScratchTransformer, TransformerConfig, count_parameters,
    get_device, get_autocast_dtype,
)
from training.data_utils import create_dataloaders, get_tokenizer


# =============================================================================
# Training Configuration
# =============================================================================

class TrainingConfig:
    vocab_size: int = 50257
    max_seq_len: int = 2048
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 2048
    dropout: float = 0.1

    dataset: str = "combined_all"
    max_train_examples: int = 1_000_000
    max_val_examples: int = 2000
    batch_size: int = 4
    grad_accumulation_steps: int = 8

    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-5
    weight_decay: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.95
    max_grad_norm: float = 1.0

    num_epochs: int = 3
    warmup_steps: int = 2000

    save_every_steps: int = 1000
    eval_every_steps: int = 500
    log_every_steps: int = 50
    checkpoint_dir: str = str(PROJECT_ROOT / "training" / "checkpoints")
    final_model_dir: str = str(PROJECT_ROOT / "training" / "finetuned_model")

    device: str = "auto"
    throttle_sleep: float = 0.1


# =============================================================================
# Learning Rate Scheduler (Cosine with Warmup)
# =============================================================================

def get_lr(step: int, config: TrainingConfig, total_steps: int) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * (step / config.warmup_steps)
    decay_steps = total_steps - config.warmup_steps
    progress = (step - config.warmup_steps) / max(1, decay_steps)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + (config.learning_rate - config.min_learning_rate) * cosine_decay


# =============================================================================
# Validation
# =============================================================================

@torch.no_grad()
def validate(model, val_loader, device, autocast_dtype, throttle_sleep: float = 0.0) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=(device.type != "cpu")):
            output = model(input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += output['loss'].item()
        num_batches += 1

        if num_batches >= 100:
            break

        if throttle_sleep > 0:
            time.sleep(throttle_sleep)

    model.train()
    return total_loss / max(1, num_batches)


# =============================================================================
# Checkpoint Management
# =============================================================================

def save_checkpoint(model, optimizer, step, epoch, loss, config, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'epoch': epoch,
        'loss': loss,
        'model_config': model.config.to_dict(),
        'training_config': {
            k: v for k, v in vars(TrainingConfig).items()
            if not k.startswith('_') and not callable(v)
        },
    }
    torch.save(checkpoint, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(path, model, optimizer=None, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['step'], checkpoint['epoch'], checkpoint['loss']


def save_final_model(model, tokenizer, config: TrainingConfig):
    save_dir = config.final_model_dir
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    with open(os.path.join(save_dir, "config.json"), 'w') as f:
        json.dump(model.config.to_dict(), f, indent=2)
    tokenizer.save_pretrained(save_dir)
    print(f"\nFinal model saved to: {save_dir}")
    print(f"  - model.pt ({os.path.getsize(os.path.join(save_dir, 'model.pt')) / 1e6:.1f} MB)")
    print(f"  - config.json")
    print(f"  - tokenizer files")


# =============================================================================
# Main Training Loop
# =============================================================================

def train():
    config = TrainingConfig()
    device = get_device(config.device)

    on_cuda = device.type == "cuda"
    on_mps = device.type == "mps"
    effective_throttle = 0.0 if on_cuda else config.throttle_sleep
    effective_workers = 4 if on_cuda else 0
    use_amp = device.type != "cpu"
    autocast_dtype = get_autocast_dtype(device)

    print("=" * 70)
    print("TOOL-AUGMENTED TRANSFORMER TRAINING (LLaMA-style)")
    print("=" * 70)
    print(f"\nDevice: {device}")
    if on_cuda:
        try:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
        print(f"AMP dtype: {autocast_dtype} | Workers: {effective_workers} | Throttle: off")
    elif on_mps:
        print(f"AMP dtype: {autocast_dtype} | Workers: 0 | Throttle: {effective_throttle}s/batch")
    else:
        print(f"AMP: off | Workers: 0 | Throttle: {effective_throttle}s/batch")

    print(f"Dataset: {config.dataset}")
    print(f"Batch size: {config.batch_size} x {config.grad_accumulation_steps} "
          f"= {config.batch_size * config.grad_accumulation_steps} effective")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Epochs: {config.num_epochs}")

    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 1: Loading Data")
    print("-" * 70)

    train_loader, val_loader, tokenizer = create_dataloaders(
        dataset_name=config.dataset,
        max_train_examples=config.max_train_examples,
        max_val_examples=config.max_val_examples,
        max_seq_len=config.max_seq_len,
        batch_size=config.batch_size,
        num_workers=effective_workers,
    )

    total_steps = (len(train_loader) // config.grad_accumulation_steps) * config.num_epochs
    print(f"\nTotal training steps: {total_steps}")

    # =========================================================================
    # Step 2: Create Model
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 2: Creating Model")
    print("-" * 70)

    model_config = TransformerConfig(
        vocab_size=config.vocab_size,
        max_seq_len=config.max_seq_len,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        dropout=config.dropout,
    )

    model = ScratchTransformer(model_config)
    model.resize_token_embeddings(len(tokenizer), tokenizer)
    model = model.to(device)

    params = count_parameters(model)
    print(f"Parameters: {params['trainable_millions']:.1f}M trainable")
    print(f"Effective vocab: {len(tokenizer)} (base + {len(tokenizer) - config.vocab_size} special tokens)")

    # =========================================================================
    # Step 3: Optimizer
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 3: Setting Up Optimizer")
    print("-" * 70)

    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'bias' in name or 'ln' in name or 'layernorm' in name or 'weight' in name and isinstance(
                dict(model.named_modules()).get(name.rsplit('.', 1)[0]), type(None)
            ):
                if any(nd in name for nd in ('bias', 'ln_final.weight', 'ln1.weight', 'ln2.weight')):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
            else:
                decay_params.append(param)

    optimizer_groups = [
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]

    optimizer = torch.optim.AdamW(
        optimizer_groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=1e-8,
    )

    print(f"Optimizer: AdamW (lr={config.learning_rate}, wd={config.weight_decay})")
    print(f"  Decay params: {sum(p.numel() for p in decay_params) / 1e6:.1f}M")
    print(f"  No-decay params: {sum(p.numel() for p in no_decay_params) / 1e6:.1f}M")

    # =========================================================================
    # Step 4: Resume from checkpoint
    # =========================================================================
    start_step = 0
    start_epoch = 0
    checkpoint_path = os.path.join(config.checkpoint_dir, "latest.pt")

    if os.path.exists(checkpoint_path):
        print(f"\nResuming from checkpoint: {checkpoint_path}")
        start_step, start_epoch, last_loss = load_checkpoint(
            checkpoint_path, model, optimizer, device
        )
        print(f"  Resumed at step {start_step}, epoch {start_epoch}, loss {last_loss:.4f}")

    # =========================================================================
    # Step 5: Training Loop
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 5: Training")
    print("-" * 70)
    print(f"{'Step':>8} | {'Epoch':>5} | {'Loss':>8} | {'LR':>10} | {'Time':>8} | {'Tok/s':>8}")
    print("-" * 70)

    scaler = torch.amp.GradScaler(enabled=(on_cuda and autocast_dtype != torch.float32))

    model.train()
    global_step = start_step
    best_val_loss = float('inf')
    training_log = []

    for epoch in range(start_epoch, config.num_epochs):
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            step_start = time.time()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
                output = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = output['loss'] / config.grad_accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % config.grad_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.max_grad_norm
                )

                lr = get_lr(global_step, config, total_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                global_step += 1
                step_time = time.time() - step_start

                tokens_per_sec = (
                    config.batch_size * config.grad_accumulation_steps * config.max_seq_len
                ) / max(step_time, 0.001)

                if global_step % config.log_every_steps == 0:
                    actual_loss = loss.item() * config.grad_accumulation_steps
                    print(
                        f"{global_step:>8} | {epoch+1:>5} | "
                        f"{actual_loss:>8.4f} | {lr:>10.2e} | "
                        f"{step_time:>7.2f}s | {tokens_per_sec:>7.0f}"
                    )
                    training_log.append({
                        'step': global_step,
                        'epoch': epoch,
                        'loss': actual_loss,
                        'lr': lr,
                        'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    })

                if global_step % config.eval_every_steps == 0:
                    val_loss = validate(model, val_loader, device, autocast_dtype)
                    print(f"  Validation loss: {val_loss:.4f}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(
                            model, optimizer, global_step, epoch, val_loss, config,
                            os.path.join(config.checkpoint_dir, "best.pt")
                        )
                        print(f"  New best validation loss: {val_loss:.4f}")

                    model.train()

                if global_step % config.save_every_steps == 0:
                    save_checkpoint(
                        model, optimizer, global_step, epoch,
                        loss.item() * config.grad_accumulation_steps, config,
                        checkpoint_path
                    )

            if effective_throttle > 0:
                time.sleep(effective_throttle)

        epoch_time = time.time() - epoch_start
        print(f"\n  Epoch {epoch+1} completed in {epoch_time/60:.1f} minutes")
        print(f"  Steps so far: {global_step}")

        val_loss = validate(model, val_loader, device, autocast_dtype)
        print(f"  End-of-epoch val loss: {val_loss:.4f}")

    # =========================================================================
    # Step 6: Save Final Model
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 6: Saving Final Model")
    print("-" * 70)

    save_final_model(model, tokenizer, config)

    log_path = os.path.join(config.checkpoint_dir, "training_log.json")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f"Training log saved to: {log_path}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Total steps: {global_step}")
    print(f"Model saved at: {config.final_model_dir}")

    return model, tokenizer


# =============================================================================
# Quick Test Mode
# =============================================================================

def quick_test():
    print("=" * 70)
    print("QUICK TEST MODE (10 steps)")
    print("=" * 70)

    device = get_device()
    autocast_dtype = get_autocast_dtype(device)
    print(f"Device: {device}, AMP dtype: {autocast_dtype}")

    model_config = TransformerConfig(
        d_model=256, n_heads=4, n_layers=4, d_ff=512, max_seq_len=256
    )

    tokenizer = get_tokenizer()
    model = ScratchTransformer(model_config)
    model.resize_token_embeddings(len(tokenizer), tokenizer)
    model = model.to(device)

    from training.data_utils import SummarizationDataset, PadCollate
    from torch.utils.data import DataLoader

    articles = ["This is a test article about machine learning. " * 20] * 20
    summaries = ["Machine learning is a field of AI."] * 20

    dataset = SummarizationDataset(articles, summaries, tokenizer, max_len=256)
    collate = PadCollate(tokenizer.pad_token_id)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print("\nRunning 10 training steps...")
    model.train()
    for i, batch in enumerate(loader):
        if i >= 10:
            break

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=(device.type != "cpu")):
            output = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = output['loss']

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"  Step {i+1}: loss = {loss.item():.4f}")

    print("\nTesting generation...")
    prompt = tokenizer.encode("This is a test", return_tensors="pt").to(device)
    generated = model.generate(prompt, max_new_tokens=20)
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"  Generated: {text[:100]}...")

    print("\nQuick test passed! Ready for full training.")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    if "--test" in sys.argv:
        quick_test()
    else:
        train()
