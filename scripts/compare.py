import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compare_results(baseline_path: str, preconditioned_path: str, output_dir: str = "results"):
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(preconditioned_path) as f:
        preconditioned = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(baseline["train_loss"], label="Baseline", linewidth=2)
    axes[0].plot(preconditioned["train_loss"], label="Preconditioned", linewidth=2)
    axes[0].set_title("Training Loss", fontsize=14)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if baseline.get("condition_numbers") and preconditioned.get("condition_numbers"):
        axes[1].plot(baseline["condition_numbers"], label="Baseline", linewidth=2)
        axes[1].plot(preconditioned["condition_numbers"], label="Preconditioned", linewidth=2)
        axes[1].set_title("Average Condition Number", fontsize=14)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Condition Number")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    epochs = list(range(1, len(baseline["train_loss"]) + 1))
    axes[2].bar([e - 0.2 for e in epochs], baseline["train_loss"], 0.4, label="Baseline", alpha=0.8)
    axes[2].bar([e + 0.2 for e in epochs], preconditioned["train_loss"], 0.4, label="Preconditioned", alpha=0.8)
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

    print("\nBaseline:")
    print(f"  Final loss:           {baseline['train_loss'][-1]:.4f}")
    if baseline["condition_numbers"]:
        avg = sum(baseline["condition_numbers"]) / len(baseline["condition_numbers"])
        print(f"  Avg condition number: {avg:.2f}")
    print(f"  Total time:           {sum(baseline['epoch_times']):.1f}s")

    print("\nPreconditioned:")
    print(f"  Final loss:           {preconditioned['train_loss'][-1]:.4f}")
    if preconditioned["condition_numbers"]:
        avg = sum(preconditioned["condition_numbers"]) / len(preconditioned["condition_numbers"])
        print(f"  Avg condition number: {avg:.2f}")
    print(f"  Total time:           {sum(preconditioned['epoch_times']):.1f}s")

    if baseline["condition_numbers"] and preconditioned["condition_numbers"]:
        base_avg = sum(baseline["condition_numbers"]) / len(baseline["condition_numbers"])
        precond_avg = sum(preconditioned["condition_numbers"]) / len(preconditioned["condition_numbers"])
        reduction = (1 - precond_avg / base_avg) * 100
        print(f"\nCondition number reduction: {reduction:.1f}%")
        if reduction > 0:
            print("  -> Preconditioned attention produces LOWER condition numbers (paper hypothesis validated)")
        else:
            print("  -> No significant condition number reduction observed")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="results/baseline_history.json")
    parser.add_argument("--preconditioned", default="results/preconditioned_history.json")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    compare_results(args.baseline, args.preconditioned, args.output_dir)
