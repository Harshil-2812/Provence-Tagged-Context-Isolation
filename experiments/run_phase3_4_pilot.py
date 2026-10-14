"""
Phase 3.4 Mini-Pilot Runner.

Executes exactly the six required pilot trials:
1. ATK001, no_defense, seed=42
2. ATK001, delimiter_defense, seed=42
3. ATK001, full_defense, seed=42
4. TASK001, no_defense, seed=123
5. TASK001, delimiter_defense, seed=123
6. TASK001, full_defense, seed=123

Target directory: results/phase3_4_pilot/
Hard wall-clock deadline per trial: 180 seconds.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime

# Ensure project root is on path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from agent.agent import Agent, LLMClient
from attacks.indirect_injection import load_attack_cases, get_attack_by_id
from evaluation.experiment_runner import (
    load_task_cases,
    run_attack_experiment,
    run_task_experiment,
)
from evaluation.results import ResultsStore
from evaluation.metrics import compute_all_metrics, format_metrics_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("phase3_4_pilot")


def run_phase3_4_pilot(results_dir: str = "results/phase3_4_pilot"):
    os.makedirs(results_dir, exist_ok=True)
    
    # Target trial matrix:
    # 1. ATK001, no_defense, seed=42
    # 2. ATK001, delimiter_defense, seed=42
    # 3. ATK001, full_defense, seed=42
    # 4. TASK001, no_defense, seed=123
    # 5. TASK001, delimiter_defense, seed=123
    # 6. TASK001, full_defense, seed=123
    
    trials_spec = [
        {"type": "attack", "case_id": "ATK001", "defense": "no_defense", "seed": 42},
        {"type": "attack", "case_id": "ATK001", "defense": "delimiter_defense", "seed": 42},
        {"type": "attack", "case_id": "ATK001", "defense": "full_defense", "seed": 42},
        {"type": "task", "case_id": "TASK001", "defense": "no_defense", "seed": 123},
        {"type": "task", "case_id": "TASK001", "defense": "delimiter_defense", "seed": 123},
        {"type": "task", "case_id": "TASK001", "defense": "full_defense", "seed": 123},
    ]

    print("=" * 70)
    print("PHASE 3.4 MINI-PILOT EXECUTION")
    print("=" * 70)
    print(f"Results Directory: {results_dir}")
    print(f"Provider: local (Ollama)")
    print(f"Model: qwen3:4b")
    print(f"Hard Deadline: 180s per call")
    print(f"Trial Count: {len(trials_spec)}")
    print("=" * 70)

    # Initialize client
    llm = LLMClient(provider="local", model="qwen3:4b")
    
    # Load cases
    attack_cases = load_attack_cases()
    atk001 = get_attack_by_id("ATK001", attack_cases)
    if not atk001:
        raise ValueError("Could not find ATK001 in attack cases")

    task_cases = load_task_cases()
    task001 = next((t for t in task_cases if t.get("task_id") == "TASK001"), None)
    if not task001:
        raise ValueError("Could not find TASK001 in task cases")

    store = ResultsStore(results_dir=results_dir)
    all_results = []

    start_all = time.time()

    for idx, spec in enumerate(trials_spec, 1):
        case_id = spec["case_id"]
        defense = spec["defense"]
        seed = spec["seed"]
        trial_type = spec["type"]

        print(f"\n[Trial {idx}/6] Running {case_id} under {defense} (seed={seed})...")
        t0 = time.time()

        agent = Agent(llm_client=llm, defense_mode=defense)

        if trial_type == "attack":
            res = run_attack_experiment(agent, atk001, defense, seed=seed)
        else:
            res = run_task_experiment(agent, task001, defense, seed=seed)

        elapsed = time.time() - t0
        print(f"  -> Completed in {elapsed:.2f}s | Status: {res.get('status')} | Decision: {res.get('final_decision')}")
        if trial_type == "attack":
            print(f"     LLM Compliance: {res.get('llm_compliance')} | E2E Success: {res.get('end_to_end_success')}")
        else:
            print(f"     Content Correct: {res.get('content_correct')} | Task Success: {res.get('task_success')} | False Block: {res.get('false_block')}")

        all_results.append(res)
        store.add(res)

    total_time = time.time() - start_all
    print("\n" + "=" * 70)
    print(f"ALL 6 TRIALS COMPLETED IN {total_time:.2f}s")
    print("=" * 70)

    # Save raw records
    raw_json = store.save_json("raw_results.json")
    raw_csv = store.save_csv("raw_results.csv")
    print(f"Saved: {raw_json}")
    print(f"Saved: {raw_csv}")

    # Compute metrics per defense mode
    defenses = ["no_defense", "delimiter_defense", "full_defense"]
    all_metrics = []
    for d in defenses:
        d_res = [r for r in all_results if r.get("defense") == d]
        m = compute_all_metrics(d_res, defense_name=d)
        all_metrics.append(m)

    metrics_json = store.save_metrics_json(all_metrics, "metrics.json")
    summary_csv = store.save_summary_csv(all_metrics, "summary.csv")
    print(f"Saved: {metrics_json}")
    print(f"Saved: {summary_csv}")

    # Save manifest
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "experiment_phase": "phase3_4_pilot",
        "provider": "local",
        "model": "qwen3:4b",
        "timeout_seconds": 180.0,
        "total_trials": len(all_results),
        "trials_spec": trials_spec,
        "completed_time_seconds": total_time,
        "python_version": sys.version,
    }
    manifest_path = os.path.join(results_dir, "config_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved: {manifest_path}")

    print("\n" + "=" * 70)
    print("METRICS SUMMARY TABLE")
    print("=" * 70)
    print(format_metrics_table(all_metrics))

    return all_results, all_metrics


if __name__ == "__main__":
    run_phase3_4_pilot()
