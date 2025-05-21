import json
from pathlib import Path
from typing import Any, Dict, Optional

from core.configs import ExperimentConfig


def load_existing_results(output_dir: str, experiment_name: str) -> Optional[Dict[str, Any]]:
    """Load existing results if present and rerun not requested."""
    result_path = Path(output_dir) / f"results_{experiment_name}.json"
    if result_path.exists():
        with open(result_path) as f:
            return json.load(f)
    return None


def save_experiment_results(
    output_dir: str, 
    experiment_name: str, 
    results: Dict[str, Any],
    experiment_config: ExperimentConfig,
    model_name: str,
    tasks: list[str]
) -> None:
    """Save experiment results to JSON file with additional metadata."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Convert experiment config to dict
    config_dict = {
        "name": experiment_config.name,
        "quantization_type": str(experiment_config.quantization_type),
        "reduction_method": experiment_config.reduction_method,
        "calibration_dataset": experiment_config.calibration_dataset,
        "reduction_fit_dataset": experiment_config.reduction_fit_dataset,
    }
    
    # Add reduction config if present
    if experiment_config.pca_config:
        config_dict["pca_config"] = experiment_config.pca_config.__dict__
    if experiment_config.kpca_config:
        config_dict["kpca_config"] = experiment_config.kpca_config.__dict__
    if experiment_config.umap_config:
        config_dict["umap_config"] = experiment_config.umap_config.__dict__
    if experiment_config.rp_config:
        config_dict["rp_config"] = experiment_config.rp_config.__dict__
    if experiment_config.ae_config:
        config_dict["ae_config"] = experiment_config.ae_config.__dict__
    
    # Create complete results dictionary
    complete_results = {
        "model_name": model_name,
        "tasks": tasks,
        "experiment_config": config_dict,
        "scores": results["scores"],
        "time": results["time"]
    }
    
    with open(f"{output_dir}/results_{experiment_name}.json", "w") as f:
        json.dump(complete_results, f, indent=2)
