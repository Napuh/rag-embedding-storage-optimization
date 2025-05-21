from typing import Optional

from core.configs import ExperimentConfig
from core.reduction.ae_reducer import AEReducer
from core.reduction.base import DimensionalityReducer
from core.reduction.kpca_reducer import KPCAReducer
from core.reduction.pca_reducer import PCAReducer
from core.reduction.rp_reducer import RPReducer
from core.reduction.umap_reducer import UMAPReducer


def get_reducer(
    experiment: ExperimentConfig,
    model_name: str,
    benchmark: str,
    device: str,
) -> Optional[DimensionalityReducer]:
    """Factory que devuelve el reducer adecuado según un ExperimentConfig."""
    method = experiment.reduction_method
    if method == "PCA" and experiment.pca_config:
        return PCAReducer(experiment.pca_config)
    if method == "KPCA" and experiment.kpca_config:
        return KPCAReducer(experiment.kpca_config)
    if method == "UMAP" and experiment.umap_config:
        return UMAPReducer(experiment.umap_config)
    if method == "RP" and experiment.rp_config:
        return RPReducer(experiment.rp_config)
    if method == "AE" and experiment.ae_config:
        return AEReducer(experiment.ae_config, model_name, benchmark, device)
    return None 