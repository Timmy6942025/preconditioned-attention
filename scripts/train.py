import argparse
import json
import os
import time

import torch
import torch.nn as nn

from preconditioned_attention import (
    ConditionNumberMonitor,
    TinyTransformer,
    create_dataloaders,
)


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    num_epochs: int,
    lr: float,
    device: str = "cpu",
    monitor: ConditionNumberMonitor | None = None,
) -> dict:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "condition_numbers": [],
        "learning_rates": [],
        "epoch_times": [],
    }

    for epoch in range(num_epochs):
        epoch_start = time.time()

        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits, _ = model(batch_x)
            loss = criterion(logits.view(-1, logits.size(-1)), batch_y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits, _ = model(batch_x)
                loss = criterion(logits.view(-1, logits.size(-1)), batch_y.view(-1))
                val_loss += loss.item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        epoch_time = time.time() - epoch_start

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["learning_rates"].append(scheduler.get_last_lr()[0])
        history["epoch_times"].append(epoch_time)

        if monitor:
            avg_cond = monitor.get_average_condition_number()
            history["condition_numbers"].append(avg_cond)
            monitor.clear()

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            cond_str = f", cond={history['condition_numbers'][-1]:.1f}" if history["condition_numbers"] else ""
            print(
                f"  Epoch {epoch + 1:3d}/{num_epochs}: "
                f"train_loss={avg_train:.4f}, val_loss={avg_val:.4f}"
                f"{cond_str} ({epoch_time:.1f}s)"
            )

    return history


def run_training(
    variant: str = "baseline",
    task: str = "copy",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 42,
    output_dir: str = "results",
):
    torch.manual_seed(seed)

    use_preconditioned = variant == "preconditioned"
    use_sigma_reparam = variant == "sigma_reparam"

    print(f"\n{'=' * 60}")
    print(f"Training {variant.upper()} transformer on {task} task")
    print(f"{'=' * 60}")

    train_loader, val_loader = create_dataloaders(
        task=task,
        vocab_size=64,
        seq_len=32,
        train_samples=8000,
        val_samples=2000,
        batch_size=batch_size,
        seed=seed,
    )

    model = TinyTransformer(
        vocab_size=64,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.1,
        max_seq_len=64,
        use_preconditioned=use_preconditioned,
        use_sigma_reparam=use_sigma_reparam,
    )
    print(f"Model parameters: {model.count_parameters():,}")

    monitor = ConditionNumberMonitor()
    for i, layer in enumerate(model.layers):
        monitor.register_hook(layer.self_attn.attn, layer_idx=i, head_idx=0)

    print(f"\nStarting training for {epochs} epochs...")
    history = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=epochs,
        lr=lr,
        monitor=monitor,
    )

    os.makedirs(output_dir, exist_ok=True)

    history_path = os.path.join(output_dir, f"{variant}_{task}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    model_path = os.path.join(output_dir, f"{variant}_{task}_model.pt")
    torch.save(model.state_dict(), model_path)

    total_time = sum(history["epoch_times"])
    print(f"\nTraining complete in {total_time:.1f}s")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    if history["condition_numbers"]:
        avg_cond = sum(history["condition_numbers"]) / len(history["condition_numbers"])
        print(f"Average condition number: {avg_cond:.2f}")
    print(f"History saved to: {history_path}")
    print(f"Model saved to: {model_path}")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train transformer with optional preconditioned attention")
    parser.add_argument(
        "--variant", type=str, default="baseline", choices=["baseline", "preconditioned", "sigma_reparam"]
    )
    parser.add_argument("--task", type=str, default="copy", choices=["copy", "reverse"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    run_training(
        variant=args.variant,
        task=args.task,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        output_dir=args.output_dir,
    )
