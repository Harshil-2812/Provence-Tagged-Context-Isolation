"""
Publication-style plots for research results.
All plots are generated from actual experiment data — no hardcoded values.
"""

import os
import json
import logging
from typing import List, Dict, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logger = logging.getLogger(__name__)

# Publication-style settings
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Color scheme for defenses
DEFENSE_COLORS = {
    "no_defense": "#d62728",        # Red
    "delimiter_defense": "#ff7f0e",  # Orange
    "full_defense": "#2ca02c",       # Green
    "provenance_only": "#1f77b4",    # Blue
    "isolation_only": "#9467bd",     # Purple
    "validation_only": "#8c564b",    # Brown
    "provenance_isolation": "#e377c2", # Pink
}

DEFENSE_LABELS = {
    "no_defense": "No Defense",
    "delimiter_defense": "Delimiter Defense",
    "full_defense": "Provenance Defense (Full)",
    "provenance_only": "Provenance Tagging Only",
    "isolation_only": "Isolation Only",
    "validation_only": "Validation Only",
    "provenance_isolation": "Provenance + Isolation",
}


def _get_label(defense: str) -> str:
    return DEFENSE_LABELS.get(defense, defense)


def _get_color(defense: str) -> str:
    return DEFENSE_COLORS.get(defense, "#7f7f7f")


def plot_attack_success_comparison(metrics: List[Dict], output_dir: str) -> str:
    """Bar chart: Attack Success Rate per defense condition."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    defenses = [m["defense"] for m in metrics]
    asr = [m["attack_success_rate"] * 100 for m in metrics]
    colors = [_get_color(d) for d in defenses]
    labels = [_get_label(d) for d in defenses]
    
    x = np.arange(len(defenses))
    bars = ax.bar(x, asr, color=colors, width=0.6, edgecolor="white", linewidth=0.5)
    
    # Value labels on bars
    for bar, val in zip(bars, asr):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title("Attack Success Rate by Defense Condition\n"
                 "(Lower is Better — Fewer Successful Injections)")
    ax.set_ylim(0, 110)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.text(len(defenses) - 0.5, 51, "50% baseline", ha="right", color="gray", fontsize=8)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "attack_success_comparison.png")
    plt.savefig(filepath)
    plt.close()
    logger.info(f"Saved: {filepath}")
    return filepath


def plot_task_success_comparison(metrics: List[Dict], output_dir: str) -> str:
    """Bar chart: Task Success Rate per defense condition."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    defenses = [m["defense"] for m in metrics]
    tsr = [m["task_success_rate"] * 100 for m in metrics]
    colors = [_get_color(d) for d in defenses]
    labels = [_get_label(d) for d in defenses]
    
    x = np.arange(len(defenses))
    bars = ax.bar(x, tsr, color=colors, width=0.6, edgecolor="white", linewidth=0.5)
    
    for bar, val in zip(bars, tsr):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Task Success Rate (%)")
    ax.set_title("Task Success Rate by Defense Condition\n"
                 "(Higher is Better — Utility Preservation)")
    ax.set_ylim(0, 110)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "task_success_comparison.png")
    plt.savefig(filepath)
    plt.close()
    logger.info(f"Saved: {filepath}")
    return filepath


def plot_false_blocking_comparison(metrics: List[Dict], output_dir: str) -> str:
    """Bar chart: False Blocking Rate per defense condition."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    defenses = [m["defense"] for m in metrics]
    fbr = [m["false_blocking_rate"] * 100 for m in metrics]
    colors = [_get_color(d) for d in defenses]
    labels = [_get_label(d) for d in defenses]
    
    x = np.arange(len(defenses))
    bars = ax.bar(x, fbr, color=colors, width=0.6, edgecolor="white", linewidth=0.5)
    
    for bar, val in zip(bars, fbr):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("False Blocking Rate (%)")
    ax.set_title("False Blocking Rate by Defense Condition\n"
                 "(Lower is Better — Fewer Legitimate Tasks Blocked)")
    ax.set_ylim(0, max(fbr) * 1.3 + 5 if fbr else 50)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "false_blocking_comparison.png")
    plt.savefig(filepath)
    plt.close()
    logger.info(f"Saved: {filepath}")
    return filepath


def plot_security_utility_tradeoff(metrics: List[Dict], output_dir: str) -> str:
    """
    Scatter plot: Security-Utility Trade-off.
    X-axis: Task Success Rate (utility)
    Y-axis: Attack Success Rate (security risk)
    
    Ideal: high TSR (right) + low ASR (bottom) → better defense
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for m in metrics:
        tsr = m["task_success_rate"] * 100
        asr = m["attack_success_rate"] * 100
        defense = m["defense"]
        color = _get_color(defense)
        label = _get_label(defense)
        
        ax.scatter(tsr, asr, c=color, s=180, zorder=5, edgecolors="white", linewidth=1.5)
        ax.annotate(label, (tsr, asr),
                    textcoords="offset points", xytext=(8, 5),
                    fontsize=8, color=color, fontweight="bold")
    
    # Ideal region annotation
    ax.axhline(y=25, color="green", linestyle="--", alpha=0.3, linewidth=1)
    ax.axvline(x=75, color="green", linestyle="--", alpha=0.3, linewidth=1)
    ax.fill_between([75, 105], [0, 0], [25, 25], alpha=0.05, color="green")
    ax.text(80, 5, "Ideal Region\n(Low ASR + High TSR)", fontsize=8, color="green", alpha=0.7)
    
    ax.set_xlabel("Task Success Rate (%) →  Higher is Better")
    ax.set_ylabel("Attack Success Rate (%) →  Lower is Better")
    ax.set_title("Security-Utility Trade-off\nby Defense Condition")
    ax.set_xlim(-5, 110)
    ax.set_ylim(-5, 110)
    ax.invert_yaxis()  # Lower ASR = better, so invert Y for visual clarity
    ax.set_ylim(110, -5)
    
    # Add direction arrows
    ax.annotate("", xy=(105, 50), xytext=(85, 50),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
    ax.text(88, 55, "Better\nUtility", fontsize=8, color="gray")
    
    ax.annotate("", xy=(50, -3), xytext=(50, 15),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
    ax.text(52, 5, "Better\nSecurity", fontsize=8, color="gray")
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "security_utility_tradeoff.png")
    plt.savefig(filepath)
    plt.close()
    logger.info(f"Saved: {filepath}")
    return filepath


def plot_combined_dashboard(metrics: List[Dict], output_dir: str) -> str:
    """Combined 2x2 dashboard of all key metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Provenance-Tagged Context Isolation — Experimental Results\n"
        "Security-Utility Evaluation across Defense Conditions",
        fontsize=13, fontweight="bold", y=1.02
    )
    
    defenses = [m["defense"] for m in metrics]
    labels = [_get_label(d) for d in defenses]
    colors = [_get_color(d) for d in defenses]
    x = np.arange(len(defenses))
    
    def add_bar(ax, values, title, ylabel, invert=False):
        bars = ax.bar(x, values, color=colors, width=0.6, edgecolor="white")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.0f}%", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10, pad=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(0, 115)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    add_bar(axes[0,0], [m["attack_success_rate"]*100 for m in metrics],
            "Attack Success Rate\n(Lower = Better Security)", "ASR (%)")
    
    add_bar(axes[0,1], [m["attack_blocking_rate"]*100 for m in metrics],
            "Attack Blocking Rate\n(Higher = Better Security)", "ABR (%)")
    
    add_bar(axes[1,0], [m["task_success_rate"]*100 for m in metrics],
            "Task Success Rate\n(Higher = Better Utility)", "TSR (%)")
    
    add_bar(axes[1,1], [m["false_blocking_rate"]*100 for m in metrics],
            "False Blocking Rate\n(Lower = Less Utility Impact)", "FBR (%)")
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "dashboard.png")
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {filepath}")
    return filepath


def generate_all_plots(metrics: List[Dict], output_dir: str) -> List[str]:
    """Generate all publication-style plots."""
    os.makedirs(output_dir, exist_ok=True)
    generated = []
    
    try:
        generated.append(plot_attack_success_comparison(metrics, output_dir))
    except Exception as e:
        logger.error(f"Failed to generate attack success plot: {e}")
    
    try:
        generated.append(plot_task_success_comparison(metrics, output_dir))
    except Exception as e:
        logger.error(f"Failed to generate task success plot: {e}")
    
    try:
        generated.append(plot_false_blocking_comparison(metrics, output_dir))
    except Exception as e:
        logger.error(f"Failed to generate false blocking plot: {e}")
    
    try:
        generated.append(plot_security_utility_tradeoff(metrics, output_dir))
    except Exception as e:
        logger.error(f"Failed to generate tradeoff plot: {e}")
    
    try:
        generated.append(plot_combined_dashboard(metrics, output_dir))
    except Exception as e:
        logger.error(f"Failed to generate dashboard: {e}")
    
    return [f for f in generated if f]
