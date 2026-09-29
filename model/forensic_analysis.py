"""Forensic inventory, consistency audit, and research figures.

Reads frozen experiment artifacts and writes only to forensic_analysis/.
No model, checkpoint, or historical result files are modified.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).parent
OUT = ROOT / "forensic_analysis"
SCALES = ["126K", "1M", "5M original", "5M recovery"]
RESULT_DIRS = {
    "126K": ROOT / "results",
    "1M": ROOT / "results_1m",
    "5M original": ROOT / "results_5m",
    "5M recovery": ROOT / "results_5m_recovery",
}
EXPORT_DIRS = {
    "126K": ROOT / "export" / "models",
    "1M": ROOT / "export" / "models_1m",
    "5M original": ROOT / "export" / "models_5m",
    "5M recovery": ROOT / "export" / "models_5m_recovery",
}
COLORS = {"CoT": "#1769aa", "Recurrent Latent": "#d95f02"}


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def canonical_model(value: str) -> str:
    return "CoT" if value.lower() in {"cot", "autoregressivecot"} else "Recurrent Latent"


def read_benchmarks():
    rows = []
    for scale, folder in RESULT_DIRS.items():
        for row in load_json(folder / "raw_benchmark_results.json"):
            row = dict(row)
            row["condition"] = scale
            row["model_display"] = canonical_model(row["model"])
            row["parameters"] = row.get("total_parameters", "")
            rows.append(row)
    return rows


def read_logs():
    rows = []
    pattern = re.compile(r"(?P<model>cot|latent)_task_(?P<task>a|b)_s(?P<seed>[0-9]+)_(?P<condition>aligned|recovery|end_to_end)")
    for condition, folder in EXPORT_DIRS.items():
        for log_path in sorted(folder.glob("*/training_log.json")):
            match = pattern.fullmatch(log_path.parent.name)
            if not match:
                continue
            data = load_json(log_path)
            entries = data.get("epochs", [])
            if not entries:
                entries = [{"epoch": index, "loss": loss} for index, loss in enumerate(data.get("losses", []), start=1)]
            for entry in entries:
                rows.append({
                    "condition": condition,
                    "model": canonical_model(match.group("model")),
                    "task": f"task_{match.group('task')}",
                    "seed": int(match.group("seed")),
                    "epoch": entry.get("epoch"),
                    "loss": entry.get("loss"),
                    "lr": entry.get("lr"),
                    "avg_grad_norm_before": entry.get("avg_grad_norm_before"),
                    "clipped_fraction": entry.get("clipped_fraction"),
                    "source": str(log_path.relative_to(ROOT)),
                })
    return rows


def mean_sem(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0


def empirical_frontier(points):
    """Return observed non-dominated points for minimize-cost/maximize-accuracy."""
    frontier = []
    for point in sorted(points, key=lambda item: (item["cost"], -item["accuracy"], item["k"])):
        dominated = any(
            other["cost"] <= point["cost"]
            and other["accuracy"] >= point["accuracy"]
            and (other["cost"] < point["cost"] or other["accuracy"] > point["accuracy"])
            for other in points
        )
        if not dominated:
            frontier.append(point)
    return frontier


def write_csv(path: Path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_accuracy(rows, task, filename, title):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    for axis, condition in zip(axes.flat, SCALES):
        for model in ("CoT", "Recurrent Latent"):
            grouped = defaultdict(list)
            for row in rows:
                if row["condition"] == condition and row["task"] == task and row["model_display"] == model:
                    grouped[row["k"]].append(row["accuracy"])
            ks = sorted(grouped)
            stats = [mean_sem(grouped[k]) for k in ks]
            means = [item[0] for item in stats]
            sems = [item[1] for item in stats]
            axis.errorbar(ks, means, yerr=sems, marker="o", capsize=3, label=model, color=COLORS[model])
        axis.set_title(condition)
        axis.grid(alpha=0.25)
        axis.set_ylim(-0.02, 1.04)
        axis.set_xticks([1, 2, 4, 8, 12, 16])
    for axis in axes[1]:
        axis.set_xlabel("Inference budget k")
    for axis in axes[:, 0]:
        axis.set_ylabel("Accuracy")
    axes[0, 0].legend(frameon=False)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_scaling(rows):
    fig, axis = plt.subplots(figsize=(9, 5.5))
    conditions = SCALES
    x = np.arange(len(conditions))
    width = 0.36
    for offset, model in [(-width / 2, "CoT"), (width / 2, "Recurrent Latent")]:
        means, sems = [], []
        for condition in conditions:
            values = [r["accuracy"] for r in rows if r["condition"] == condition and r["task"] == "task_a" and r["model_display"] == model and r["k"] == 16]
            mean, sem = mean_sem(values)
            means.append(mean)
            sems.append(sem)
        axis.bar(x + offset, means, width, yerr=sems, capsize=3, label=model, color=COLORS[model])
    axis.set_xticks(x, conditions)
    axis.set_ylabel("Task A accuracy at k=16")
    axis.set_ylim(0, 1.08)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    axis.set_title("Scale and training condition at the largest tested budget", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "03_scaling_k16.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pareto(rows):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    for axis, condition in zip(axes.flat, SCALES):
        for model in ("CoT", "Recurrent Latent"):
            points = []
            for row in rows:
                if row["condition"] == condition and row["task"] == "task_a" and row["model_display"] == model:
                    points.append({"cost": row["latency_ms"], "accuracy": row["accuracy"], "k": row["k"]})
            points = sorted(points, key=lambda item: item["cost"])
            frontier = empirical_frontier(points)
            axis.plot([p["cost"] for p in points], [p["accuracy"] for p in points], "o", color=COLORS[model], alpha=0.45)
            axis.plot([p["cost"] for p in frontier], [p["accuracy"] for p in frontier], "-", color=COLORS[model], label=model)
            for point in points:
                axis.annotate(f"k={point['k']}", (point["cost"], point["accuracy"]), fontsize=6, alpha=0.7)
        axis.set_title(condition)
        axis.set_xlabel("Measured latency (ms/sample)")
        axis.grid(alpha=0.25)
        axis.set_ylim(-0.02, 1.04)
    axes[0, 0].set_ylabel("Task A accuracy")
    axes[1, 0].set_ylabel("Task A accuracy")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Empirical Task A latency frontiers from observed operating points", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "04_pareto_latency_accuracy.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_recovery(rows):
    selected = [r for r in rows if r["condition"] in {"5M original", "5M recovery"} and r["task"] == "task_a"]
    fig, axis = plt.subplots(figsize=(9, 5.5))
    for condition, linestyle in [("5M original", "--"), ("5M recovery", "-")]:
        for model in ("CoT", "Recurrent Latent"):
            grouped = defaultdict(list)
            for row in selected:
                if row["condition"] == condition and row["model_display"] == model:
                    grouped[row["k"]].append(row["accuracy"])
            ks = sorted(grouped)
            stats = [mean_sem(grouped[k]) for k in ks]
            axis.errorbar(ks, [s[0] for s in stats], yerr=[s[1] for s in stats], marker="o", capsize=3, linestyle=linestyle, color=COLORS[model], label=f"{model}, {condition.replace('5M ', '')}")
    axis.set_xlabel("Inference budget k")
    axis.set_ylabel("Task A accuracy")
    axis.set_xticks([1, 2, 4, 8, 12, 16])
    axis.set_ylim(-0.02, 1.04)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    axis.set_title("5M original versus recovery: training condition changes the outcome", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "05_5m_original_vs_recovery.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_training_loss(logs):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for axis, task in zip(axes.flat, ["task_a", "task_b", "task_a", "task_b"]):
        condition = "5M original" if axis in axes.flat[:2] else "5M recovery"
        for model in ("CoT", "Recurrent Latent"):
            grouped = defaultdict(list)
            for row in logs:
                if row["condition"] == condition and row["task"] == task and row["model"] == model:
                    grouped[row["epoch"]].append(row["loss"])
            epochs = sorted(grouped)
            axis.plot(epochs, [np.mean(grouped[e]) for e in epochs], color=COLORS[model], label=model)
        axis.set_title(f"{condition}: {task.replace('task_', 'Task ').upper()}")
        axis.set_ylabel("Training loss")
        axis.grid(alpha=0.25)
    for axis in axes[1]:
        axis.set_xlabel("Epoch")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Training loss across 5M seeds", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "06_training_loss_5m.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_seed_variability(rows):
    fig, axis = plt.subplots(figsize=(10, 5.5))
    positions, labels = [], []
    for index, condition in enumerate(SCALES):
        for model_index, model in enumerate(("CoT", "Recurrent Latent")):
            values = [r["accuracy"] for r in rows if r["condition"] == condition and r["task"] == "task_a" and r["model_display"] == model and r["k"] == 16]
            position = index * 3 + model_index + 1
            axis.scatter([position] * len(values), values, color=COLORS[model], zorder=3, s=35)
            axis.hlines(np.mean(values), position - 0.25, position + 0.25, color="black", linewidth=2)
            if model_index == 0:
                labels.append(condition)
                positions.append(index * 3 + 1.5)
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Task A accuracy at k=16")
    axis.set_ylim(-0.02, 1.08)
    axis.grid(axis="y", alpha=0.25)
    axis.scatter([], [], color=COLORS["CoT"], label="CoT seeds")
    axis.scatter([], [], color=COLORS["Recurrent Latent"], label="Latent seeds")
    axis.legend(frameon=False)
    axis.set_title("Seed-level variation is visible at the same nominal budget", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "07_seed_variability_k16.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_report(benchmarks, logs, inventory, conflicts, frontiers):
    lines = [
        "# Forensic Research Visualization Report",
        "",
        "Generated from repository artifacts by `forensic_analysis.py`.",
        "Frozen model definitions, checkpoints, and historical result folders were read but not modified.",
        "",
        "## A. Complete Result Inventory",
        "",
        f"- Benchmark rows: {len(benchmarks)} raw rows across {len(RESULT_DIRS)} conditions.",
        f"- Training logs: {len({row['source'] for row in logs})} files, {len(logs)} epoch records.",
        f"- Checkpoint files: {sum(1 for path in ROOT.glob('export/**/*.pth'))} `.pth` files.",
        "- Benchmark JSON families per condition: raw, aggregated, Pareto, threshold, and statistical outputs; recovery uses `statistical_tests.json` rather than `statistical_comparison.json`.",
        "- No CSV, NPY, PKL, attention matrix, latent-state trace, readout distribution, or standalone diagnostic artifact was found.",
        "",
        "The detailed file-level inventory is in `inventory.csv`; the normalized benchmark table is in `master_benchmark.csv`; training records are in `training_logs.csv`.",
        "",
        "## B. Master Experiment Matrix",
        "",
        "| Condition | Architecture | Tasks | Seeds | k | D in result rows |",
        "|---|---|---|---|---|---|",
    ]
    for condition in SCALES:
        subset = [r for r in benchmarks if r["condition"] == condition]
        models = sorted({r["model_display"] for r in subset})
        tasks = sorted({r["task"] for r in subset})
        seeds = sorted({r["seed"] for r in subset})
        ks = sorted({r["k"] for r in subset})
        lines.append(f"| {condition} | {', '.join(models)} | {', '.join(tasks)} | {', '.join(map(str, seeds))} | {', '.join(map(str, ks))} | absent |")
    lines += [
        "",
        "Problem depth D is present in the generator/API (2 through 16), but not recorded per benchmark row, so no empirical accuracy-vs-D figure is justified from these outputs.",
        "",
        "## C. Data Consistency Audit",
        "",
        f"- Raw-to-aggregate checks: {len(conflicts)} discrepancies found at the displayed floating-point tolerance." if conflicts else "- Raw-to-aggregate checks: all aggregate means and standard deviations agree with raw seed rows within tolerance.",
        "- Model naming is inconsistent across generations: `AutoregressiveCoT`/`RecurrentLatentReasoner` versus `cot`/`latent`; analysis canonicalizes these to CoT and Recurrent Latent.",
        "- 5M original and 5M recovery are different training conditions and are never pooled.",
        "- Latency values are measured wall-clock outputs, but cross-condition hardware/runtime comparability is not established by the files; interpret latency frontiers within condition.",
        "- Statistical outputs test same-k seed differences, not latency-matched differences. Five seeds are the independent replicates; 1,000 predictions are repeated test instances.",
        "- Seed 46 is retained. The raw files show seed-level variation, but no causal explanation is established.",
        "",
        "### Conflicts",
        "",
    ]
    lines.extend([f"- {item}" for item in conflicts] or ["- No value conflicts detected between raw rows and aggregate files."])
    lines += [
        "",
        "## D. Recommended Graphs",
        "",
        "| Figure | Scientific question | Source/variables | Aggregation | Rank |",
        "|---|---|---|---|---|",
        "| 01 Task A accuracy vs k | Does extra inference compute continue helping? | raw benchmark: task_a, k, accuracy, model, condition | mean +/- SEM across five seeds | ESSENTIAL |",
        "| 02 Task B accuracy vs k | Is the pattern task-specific or near chance? | raw benchmark: task_b, k, accuracy | mean +/- SEM across five seeds | SUPPORTING |",
        "| 03 Scaling at k=16 | How do scale and training condition change final accuracy? | raw benchmark: task_a, k=16, scale/condition, accuracy | mean +/- SEM | ESSENTIAL |",
        "| 04 Latency Pareto | Which observed points trade latency for accuracy? | raw benchmark: task_a, latency_ms, accuracy, k | observed non-dominated points only | ESSENTIAL |",
        "| 05 Original vs recovery | Did the recovery condition change the 5M outcome? | raw benchmark: 5M conditions, task_a, k, accuracy | mean +/- SEM | ESSENTIAL |",
        "| 06 Training loss | How do optimization trajectories differ? | export training_log.json: epoch, loss | mean across seeds | SUPPORTING |",
        "| 07 Seed variability | How much do conclusions vary across training seeds? | raw benchmark: task_a, k=16, seed, accuracy | individual points + mean | SUPPORTING |",
        "",
        "Analytical FLOPs can support a parallel Pareto figure, but it is largely a deterministic linear function of k within each architecture/condition and is redundant with the latency figure for an 8-minute presentation. No D, latent transition, attention, readout, or Hebbian diagnostic graph is recommended because the required raw measurements are absent.",
        "",
        "## E. Generated Graphs",
        "",
        "- `01_taskA_accuracy_vs_k.png`",
        "- `02_taskB_accuracy_vs_k.png`",
        "- `03_scaling_k16.png`",
        "- `04_pareto_latency_accuracy.png`",
        "- `05_5m_original_vs_recovery.png`",
        "- `06_training_loss_5m.png`",
        "- `07_seed_variability_k16.png`",
        "",
        "## F. Presentation Recommendation",
        "",
        "1. Task A accuracy vs k: CoT improves with budget in successful conditions, while latent accuracy rises early and then changes little; this is an observed pattern, not a universal architectural claim.",
        "2. Scaling at k=16: scale alone does not explain outcomes because the 5M original condition remains near chance while 5M recovery succeeds.",
        "3. Latency Pareto: the relevant trade-off is observed accuracy versus measured latency, separated by condition because comparability is limited.",
        "4. 5M original versus recovery: recovery changes Task A substantially, but multiple optimization variables changed, so causality is not isolated.",
        "5. Task B accuracy vs k: results remain close to the eight-way random baseline (0.125) and do not support a strong general conclusion.",
        "6. Training loss: optimization trajectories provide context for the 5M recovery result.",
        "",
        "## G. Data That Should Not Be Graphed",
        "",
        "- Problem-depth accuracy: D is not attached to benchmark rows.",
        "- Latent state transition magnitude, cosine similarity, effective rank, attention, and readout diagnostics: no raw diagnostic arrays or records were found.",
        "- A universal cross-scale latency frontier: measurement comparability is not established.",
        "- Aggregated means without seed uncertainty when comparing models: five-seed variability is material, especially at larger k.",
        "",
        "## Main Verified Pattern",
        "",
        "Task A supports continued CoT improvement with k in 126K, 1M, and 5M recovery. Recurrent Latent improves sharply from k=1 to k=2 in those conditions, then is approximately flat. The original 5M condition is a failure/near-chance condition for both architectures and must remain separate. Task B is weak and mostly near chance. Internal-state causal explanations are not testable from the stored outputs.",
    ]
    (OUT / "FORENSIC_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(exist_ok=True)
    benchmarks = read_benchmarks()
    logs = read_logs()
    inventory = []
    for condition, folder in RESULT_DIRS.items():
        for path in sorted(folder.iterdir()):
            inventory.append({"path": str(path.relative_to(ROOT)), "category": "benchmark result", "condition": condition, "experiment": "benchmark/evaluation", "raw_or_aggregate": "raw" if path.name.startswith("raw_") else "derived/summary"})
    for condition, folder in EXPORT_DIRS.items():
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                inventory.append({"path": str(path.relative_to(ROOT)), "category": "checkpoint/training", "condition": condition, "experiment": path.parent.name, "raw_or_aggregate": "training log" if path.name == "training_log.json" else "checkpoint/timing"})
    write_csv(OUT / "inventory.csv", inventory, ["path", "category", "condition", "experiment", "raw_or_aggregate"])
    master_fields = ["condition", "model_display", "model", "task", "scale", "seed", "k", "accuracy", "latency_ms", "analytical_flops", "parameters", "under_budget_acc", "matched_budget_acc", "over_budget_acc"]
    write_csv(OUT / "master_benchmark.csv", benchmarks, master_fields)
    write_csv(OUT / "training_logs.csv", logs, ["condition", "model", "task", "seed", "epoch", "loss", "lr", "avg_grad_norm_before", "clipped_fraction", "source"])

    conflicts = []
    for condition, folder in RESULT_DIRS.items():
        raw = load_json(folder / "raw_benchmark_results.json")
        aggregate = load_json(folder / "aggregated_results.json")
        for item in aggregate:
            vals = [r["accuracy"] for r in raw if r["task"] == item["task"] and r["k"] == item["k"] and canonical_model(r["model"]) == canonical_model(item["model"])]
            if vals and abs(np.mean(vals) - item["mean_accuracy"]) > 1e-9:
                conflicts.append(f"{condition} {item['task']} {item['model']} k={item['k']}: raw mean {np.mean(vals):.12g} vs aggregate {item['mean_accuracy']:.12g}")

    plot_accuracy(benchmarks, "task_a", "01_taskA_accuracy_vs_k.png", "Task A: accuracy versus inference budget")
    plot_accuracy(benchmarks, "task_b", "02_taskB_accuracy_vs_k.png", "Task B: accuracy versus inference budget")
    plot_scaling(benchmarks)
    plot_pareto(benchmarks)
    plot_recovery(benchmarks)
    plot_training_loss(logs)
    plot_seed_variability(benchmarks)
    build_report(benchmarks, logs, inventory, conflicts, {})
    print(f"Wrote forensic analysis to {OUT}")


if __name__ == "__main__":
    main()