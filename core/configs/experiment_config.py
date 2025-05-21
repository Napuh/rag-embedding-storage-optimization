from dataclasses import dataclass
from typing import Literal, Optional

from .ae_config import AEConfig
from .kpca_config import KPCAConfig
from .pca_config import PCAConfig
from .quantization_type import QuantizationType
from .rp_config import RPConfig
from .umap_config import UMAPConfig


@dataclass
class ExperimentConfig:
    name: str
    quantization_type: QuantizationType | None
    reduction_method: Optional[Literal["PCA", "KPCA", "UMAP", "RP", "AE"]] = None
    pca_config: Optional[PCAConfig] = None
    kpca_config: Optional[KPCAConfig] = None
    umap_config: Optional[UMAPConfig] = None
    rp_config: Optional[RPConfig] = None
    ae_config: Optional[AEConfig] = None
    calibration_dataset: Optional[str] = "MLQuestions"
    reduction_fit_dataset: Optional[str] = "MLQuestions"
