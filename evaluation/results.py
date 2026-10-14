"""
Results storage and management.
"""

import os
import json
import csv
import logging
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ResultsStore:
    """Manages storage and retrieval of experiment results."""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self._results: List[Dict] = []
    
    def add(self, result: Dict) -> None:
        """Add a single result record."""
        result["timestamp"] = datetime.now().isoformat()
        self._results.append(result)
    
    def add_many(self, results: List[Dict]) -> None:
        for r in results:
            self.add(r)
    
    @property
    def results(self) -> List[Dict]:
        return list(self._results)
    
    def save_csv(self, filename: str = "raw_results.csv") -> str:
        """Save results to CSV file."""
        if not self._results:
            logger.warning("No results to save")
            return ""
        
        filepath = os.path.join(self.results_dir, filename)
        
        # Flatten nested dicts for CSV
        flat_results = [self._flatten(r) for r in self._results]
        
        if flat_results:
            fieldnames = sorted(set().union(*[r.keys() for r in flat_results]))
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(flat_results)
        
        logger.info(f"Saved {len(self._results)} results to {filepath}")
        return filepath
    
    def save_json(self, filename: str = "raw_results.json") -> str:
        """Save results to JSON file."""
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self._results, f, indent=2, default=str)
        logger.info(f"Saved results JSON to {filepath}")
        return filepath
    
    def save_metrics_json(self, metrics: List[Dict], filename: str = "metrics.json") -> str:
        """Save computed metrics to JSON."""
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        return filepath
    
    def save_summary_csv(self, metrics: List[Dict], filename: str = "summary.csv") -> str:
        """Save metrics summary to CSV."""
        if not metrics:
            return ""
        
        filepath = os.path.join(self.results_dir, filename)
        
        flat_metrics = []
        for m in metrics:
            flat = {k: v for k, v in m.items() if not isinstance(v, dict)}
            if "token_usage" in m:
                for k, v in m["token_usage"].items():
                    flat[f"token_{k}"] = v
            flat_metrics.append(flat)
        
        if flat_metrics:
            fieldnames = sorted(set().union(*[r.keys() for r in flat_metrics]))
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(flat_metrics)
        
        return filepath
    
    def _flatten(self, d: Dict, parent_key: str = "", sep: str = "_") -> Dict:
        """Flatten nested dict for CSV export."""
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(self._flatten(v, new_key, sep))
            elif isinstance(v, list):
                items[new_key] = str(v)
            else:
                items[new_key] = v
        return items
    
    def get_by_defense(self, defense: str) -> List[Dict]:
        return [r for r in self._results if r.get("defense") == defense]
    
    def clear(self) -> None:
        self._results.clear()
    
    def __len__(self) -> int:
        return len(self._results)
