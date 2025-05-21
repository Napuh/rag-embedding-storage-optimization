import argparse
import time
import warnings
from typing import Any, Dict

import mteb

from core.configs import ExperimentConfig
from core.engine import EmbeddingEngine
from utils.config_utils import create_experiment_configs, load_config
from utils.experiment_utils import (load_existing_results,
                                    save_experiment_results)

warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)


def precompute_calibration_embeddings(model: EmbeddingEngine, experiment_configs: list[ExperimentConfig], batch_size: int):
    
    # Get unique calibration datasets
    datasets = {exp.calibration_dataset for exp in experiment_configs if exp.calibration_dataset}
    if not datasets:
        return
    
    print(f"Precomputing calibration embeddings in cache for: {', '.join(datasets)}")
    for dataset in datasets:
        print(f"Precomputing embeddings for dataset: {dataset}")
        model.set_benchmark(dataset)
        mteb_tasks_pre = mteb.get_tasks(tasks=[dataset], languages=["eng"])
        evaluation_pre = mteb.MTEB(tasks=mteb_tasks_pre)
        
        # Force embedding calculation without caring about results
        evaluation_pre.run(
            model,
            output_folder=None,
            encode_kwargs={"batch_size": batch_size},
            verbosity=0,
        )


def run_experiments(
    model_name: str,
    tasks: list[str],
    experiment_configs: list[ExperimentConfig],
    batch_size: int = 32,
    output_dir: str = "",
    rerun_existing: bool = False,
    cache_location: str = ":memory:",
) -> Dict[str, Any]:

    if not output_dir:
        output_dir = f"results/{model_name}"

    results = {}
    model = EmbeddingEngine(model_name=model_name, cache_location=cache_location)

    # Precompute calibration embeddings
    precompute_calibration_embeddings(model, experiment_configs, batch_size)

    for experiment in experiment_configs:
        # Load existing results if present
        existing_results = load_existing_results(output_dir, experiment.name)
        if existing_results and not rerun_existing:
            results[experiment.name] = existing_results
            ndcg_scores = results[experiment.name]["scores"]
            start_time = time.time() - results[experiment.name]["time"]
        else:
            ndcg_scores = {}
            start_time = time.time()

        print(f"Running experiment {experiment.name}")

        # Set quantization type
        model.set_quant_type(experiment.quantization_type)
        
        # Set reduction config and fit once per experiment
        model.set_reduction_config(experiment)
        
        # Set calibration and reduction fit datasets
        model.set_calibration_dataset(experiment.calibration_dataset)
        model.set_reduction_fit_dataset(experiment.reduction_fit_dataset)
        
        # Fit the reduction method once per experiment
        if experiment.reduction_method:
            print(f"Fitting {experiment.reduction_method} on {experiment.reduction_fit_dataset}...")
            model.fit_reduction()

        for benchmark in tasks:
            # Skip if benchmark already evaluated
            if (
                any(key.startswith(f"{benchmark}-") for key in ndcg_scores)
                and not rerun_existing
            ):
                print(f"Skipping benchmark {benchmark} - already evaluated")
                continue

            print(f"Running benchmark: {benchmark}")

            model.set_benchmark(benchmark)

            mteb_tasks = mteb.get_tasks(tasks=[benchmark], languages=["eng"])

            evaluation = mteb.MTEB(tasks=mteb_tasks)

            mteb_eval_result = evaluation.run(
                model,
                output_folder=None,
                encode_kwargs={"batch_size": batch_size},
                verbosity=0,
            )

            # Extract only ndcg_at_10 scores
            for task in mteb_eval_result:
                for test_set in task.scores:
                    ndcg_scores["-".join([task.task_name, test_set])] = task.scores[
                        test_set
                    ][0]["ndcg_at_10"]

            # Update results after each benchmark
            results[experiment.name] = {
                "scores": ndcg_scores.copy(),
                "time": time.time() - start_time,
            }

            save_experiment_results(
                output_dir, 
                experiment.name, 
                results[experiment.name],
                experiment,
                model_name,
                tasks
            )

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment.yml",
        help="Path to config file",
    )
    parser.add_argument(
        "--cache-location", type=str, default=":memory:", help="Cache location"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for encoding"
    )
    parser.add_argument(
        "--rerun-existing", action="store_true", help="Rerun existing experiments"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    experiment_configs = create_experiment_configs(config["experiments"])

    results = run_experiments(
        model_name=config["model_name"],
        tasks=config["tasks"],
        experiment_configs=experiment_configs,
        cache_location=args.cache_location,
        batch_size=args.batch_size,
        rerun_existing=args.rerun_existing,
    )
