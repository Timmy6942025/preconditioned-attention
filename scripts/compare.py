import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compare_results(history_files: dict[str, str], output_dir: str = "results"):
    histories = {}
    for name, path in history_files.items():
        with open(path) as f:
            histories[name] = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {"baseline": "#1f77b4", "preconditioned": "#ff7f0e", "sigma_reparam": "#2ca02c"}

    for name, h in histories.items():
        color = colors.get(name, None)
        axes[0].plot(h["train_loss"], label=name, linewidth=2, color=color)

    axes[0].set_title("Training Loss", fontsize=14)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    has_cond = all(h.get("condition_numbers") for h in histories.values())
    if has_cond:
        for name, h in histories.items():
            color = colors.get(name, None)
            axes[1].plot(h["condition_numbers"], label=name, linewidth=2, color=color)
        axes[1].set_title("Average Condition Number", fontsize=14)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Condition Number")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    epochs = list(range(1, len(next(iter(histories.values()))["train_loss"]) + 1))
    width = 0.8 / len(histories)
    for i, (name, h) in enumerate(histories.items()):
        offset = (i - len(histories) / 2 + 0.5) * width
        color = colors.get(name, None)
        axes[2].bar([e + offset for e in epochs], h["train_loss"], width, label=name, alpha=0.8, color=color)
    axes[2].set_title("Loss Comparison (Bar)", fontsize=14)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison.png", dpi=150)
    print(f"Comparison plot saved to {output_dir}/comparison.png")

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    for name, h in histories.items():
        print(f"\n{name}:")
        print(f"  Final loss:           {h['train_loss'][-1]:.4f}")
        if h.get("condition_numbers"):
            avg = sum(h["condition_numbers"]) / len(h["condition_numbers"])
            print(f"  Avg condition number: {avg:.2f}")
        print(f"  Total time:           {sum(h['epoch_times']):.1f}s")

    if has_cond:
        baseline_cond = histories["baseline"]["condition_numbers"]
        baseline_avg = sum(baseline_cond) / len(baseline_cond)
        for name, h in histories.items():
            if name == "baseline":
                continue
            avg = sum(h["condition_numbers"]) / len(h["condition_numbers"])
            reduction = (1 - avg / baseline_avg) * 100
            print(f"\n{name} condition number reduction vs baseline: {reduction:.1f}%")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="results/baseline_history.json")
    parser.add_argument("--preconditioned", default="results/preconditioned_history.json")
    parser.add_argument("--sigma-reparam", default="results/sigma_reparam_history.json")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    files = {"baseline": args.baseline, "preconditioned": args.preconditioned}
    if os.path.exists(args.sigma_reparam):
        files["sigma_reparam"] = args.sigma_reparam
    compare_results(files, args.output_dir)
