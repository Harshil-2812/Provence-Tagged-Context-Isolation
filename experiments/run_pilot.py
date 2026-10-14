"""
Phase 2 Pilot Experiment Runner.
Evaluates a targeted 6-case pilot across the 3 defense conditions:
- 2 Attacks: ATK001 (Goal Hijacking), ATK004 (Honda Civic ranking / Deliberation)
- 4 Benign Tasks: TASK004 (Python quote), TASK005 (Security policy summary), 
                  TASK006 (User001 DB quote), TASK009 (Concise World Population)

Results are saved in a fresh directory: results/phase2_pilot/
"""

import os
import sys
import json
import csv
import time
import logging

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agent import Agent, LLMClient
from attacks.indirect_injection import get_attack_by_id, load_attack_cases
from evaluation.experiment_runner import (
    load_task_cases, run_attack_experiment, run_task_experiment, DEFENSE_MODES
)
from evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


def run_phase2_pilot(output_dir: str = "results/phase2_pilot", provider: str = "local", model: str = "qwen3:4b"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Pilot cases
    pilot_attack_ids = ["ATK001", "ATK004"]
    pilot_task_ids = ["TASK004", "TASK005", "TASK006", "TASK009"]
    
    all_attacks = load_attack_cases()
    attacks = [a for a in all_attacks if a.attack_id in pilot_attack_ids]
    
    all_tasks = load_task_cases()
    tasks = [t for t in all_tasks if t.get("task_id") in pilot_task_ids]
    
    print(f"\n=======================================================")
    print(f"Starting Phase 2 Reliability & False-Blocking Pilot")
    print(f"Provider: {provider} | Model: {model}")
    print(f"Attacks: {pilot_attack_ids} (N={len(attacks)})")
    print(f"Tasks:   {pilot_task_ids} (N={len(tasks)})")
    print(f"Defenses: {DEFENSE_MODES}")
    print(f"Output Directory: {output_dir}")
    print(f"=======================================================\n")
    
    raw_results = []
    
    for defense in DEFENSE_MODES:
        print(f"\n--- Running Defense: {defense} ---")
        llm = LLMClient(provider=provider, model=model)
        agent = Agent(llm_client=llm, defense_mode=defense)
        
        # 1. Run attacks
        for atk in attacks:
            t0 = time.time()
            print(f"  [ATTACK] {atk.attack_id} ({atk.attack_category})... ", end="", flush=True)
            res = run_attack_experiment(agent, atk, defense)
            dur = time.time() - t0
            print(f"decision={res['final_decision']}, compliance={res['llm_compliance']}, status={res['status']} ({dur:.1f}s)")
            raw_results.append(res)
            
        # 2. Run tasks
        for tsk in tasks:
            t0 = time.time()
            print(f"  [TASK]   {tsk['task_id']} ({tsk['category']})... ", end="", flush=True)
            res = run_task_experiment(agent, tsk, defense)
            dur = time.time() - t0
            print(f"decision={res['final_decision']}, content_ok={res.get('content_correct')}, success={res['task_success']}, status={res['status']} ({dur:.1f}s)")
            raw_results.append(res)
            
    # Save raw results
    raw_json_path = os.path.join(output_dir, "raw_results.json")
    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2, default=str)
        
    raw_csv_path = os.path.join(output_dir, "raw_results.csv")
    if raw_results:
        all_keys = list(dict.fromkeys([k for r in raw_results for k in r.keys()]))
        with open(raw_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            for r in raw_results:
                writer.writerow({k: (str(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()})

    # Compute metrics per defense
    metrics_per_defense = []
    for defense in DEFENSE_MODES:
        defense_results = [r for r in raw_results if r["defense"] == defense]
        m = compute_all_metrics(defense_results, defense_name=defense)
        metrics_per_defense.append(m)
        
    metrics_json_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_per_defense, f, indent=2)

    # Write summary CSV
    summary_csv_path = os.path.join(output_dir, "summary.csv")
    summary_fields = [
        "defense", "total_runs", "attack_runs", "task_runs", "valid_trials",
        "llm_compliance_rate", "end_to_end_attack_success_rate", "attack_blocking_rate",
        "task_success_rate", "operational_task_success_rate", "false_blocking_rate",
        "content_correct_rate", "mean_latency_ms", "median_latency_ms"
    ]
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        for m in metrics_per_defense:
            writer.writerow({k: m.get(k) for k in summary_fields})

    print("\n=======================================================")
    print("PILOT COMPLETED SUCCESSFULLY")
    print(f"Results written to: {output_dir}")
    print("=======================================================\n")
    
    return metrics_per_defense, raw_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="local")
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--output-dir", default="results/phase2_pilot")
    args = parser.parse_args()
    
    run_phase2_pilot(output_dir=args.output_dir, provider=args.provider, model=args.model)
