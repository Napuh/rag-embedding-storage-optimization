from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml

from core.configs import (AEConfig, ExperimentConfig, KPCAConfig, PCAConfig,
                          QuantizationType, RPConfig, UMAPConfig)


def load_config(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_experiment_configs(
    experiments_config: list, experiment_type: Literal["qdrant", "mteb"] = "mteb"
) -> list[ExperimentConfig]:
    # Filter for Qdrant-supported quantization types if using qdrant
    supported_types = {
        QuantizationType.FLOAT32,
        QuantizationType.FLOAT16,
        QuantizationType.INT8,
        QuantizationType.BINARY,
    }

    experiment_configs = []
    for exp in experiments_config:
        quant_type = QuantizationType[exp["quantization_type"]]

        reduction_method: Optional[Literal["PCA", "UMAP"]] = exp.get("reduction_method")
        pca_conf_dict = exp.get("pca_config")
        kpca_conf_dict = exp.get("kpca_config")
        umap_conf_dict = exp.get("umap_config")
        rp_conf_dict = exp.get("rp_config")
        ae_config_dict = exp.get("ae_config")

        pca_config_obj = PCAConfig(**pca_conf_dict) if pca_conf_dict and reduction_method == "PCA" else None
        umap_config_obj = UMAPConfig(**umap_conf_dict) if umap_conf_dict and reduction_method == "UMAP" else None
        rp_config_obj = RPConfig(**rp_conf_dict) if rp_conf_dict and reduction_method == "RP" else None
        ae_config_obj = AEConfig(**ae_config_dict) if ae_config_dict and reduction_method == "AE" else None
        kpca_config_obj = KPCAConfig(**kpca_conf_dict) if kpca_conf_dict and reduction_method == "KPCA" else None

        # Basic validation
        if reduction_method and not (pca_config_obj or kpca_config_obj or umap_config_obj or rp_config_obj or ae_config_obj):
            raise ValueError(f"Reduction method '{reduction_method}' specified but no corresponding config found for experiment '{exp['name']}'")
        if (pca_config_obj or kpca_conf_dict or umap_config_obj or rp_config_obj or ae_config_obj) and not reduction_method:
             raise ValueError(f"Reduction config found but no reduction_method specified for experiment '{exp['name']}'")
        # Check for multiple configs
        num_configs = sum(c is not None for c in [pca_config_obj, kpca_config_obj, umap_config_obj, rp_config_obj, ae_config_obj])
        if num_configs > 1:
             raise ValueError(f"Multiple reduction configs specified for experiment '{exp['name']}'")

        # Only add experiments with supported quantization types for qdrant
        if experiment_type == "mteb" or quant_type in supported_types:
            experiment_configs.append(
                ExperimentConfig(
                    name=exp["name"],
                    quantization_type=quant_type,
                    reduction_method=reduction_method,
                    pca_config=pca_config_obj,
                    kpca_config=kpca_config_obj,
                    umap_config=umap_config_obj,
                    rp_config=rp_config_obj,
                    ae_config=ae_config_obj,
                    calibration_dataset=exp.get("calibration_dataset", "MLQuestions"),
                    reduction_fit_dataset=exp.get("reduction_fit_dataset", "MLQuestions"),
                )
            )
    return experiment_configs
