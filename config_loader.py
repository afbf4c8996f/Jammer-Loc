import os
import yaml
import logging
from pathlib import Path
from typing import Optional, List, Set

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal

logger = logging.getLogger(__name__)

# ─── Valid name pools ─────────────────────────────────────────────────────
# ML classifiers used for classification
VALID_ML_CLASSIFIERS: Set[str] = {
    "xgboost", "random_forest", "knn"  # 'svm' dropped for speed reasons
}
# ML regressors used for regression
VALID_ML_REGRESSORS: Set[str] = {
    "xgboost", "random_forest"
}
# DL classifiers
VALID_DL_CLASSIFIERS: Set[str] = {
    "simple_nn", "convmixer1d", "transformer_tabular"
}
# DL regressors
VALID_DL_REGRESSORS: Set[str] = VALID_DL_CLASSIFIERS 

# ─── Model sub-config ────────────────────────────────────────────────────
class ModelConfig(BaseModel):
    # --- hyperparameters & routing ------------------------------
    name: str = Field(..., description="Model architecture to use")
    input_size: Optional[int] = Field(
        None, gt=0,
        description="Number of input features (required for DL classification)"
    )
    hidden_size: int = Field(..., gt=0, description="Hidden dimension size")
    num_classes: Optional[int] = Field(
        None, gt=0,
        description="Number of classes (classification only)"
    )
    scale_regression: bool = Field(
        True, description="If False, skip scaling for regression pipelines"
    )

    task: Literal["classification","regression"] = Field(
        "classification", description="'classification' or 'regression'"
    )    

    framework: Literal["dl","ml"] = Field(
        "dl", description="'dl' or 'ml' (classification only)"
    )

        # ML regressors used for regression
    regressors: List[str]  = Field(
        default_factory=lambda: ["xgboost"],
        description="Which ML regressors to run (xgboost, random_forest) or ['all']"
    )

    input_mode: str = Field("actual", description="'actual' or 'rate'")

    # ML classification algorithms (can also be used for regression; 'all' expands)
    algorithms: List[str]   = Field(default_factory=lambda: ["xgboost"])
    # DL classification models
    dl_models: List[str]    = Field(default_factory=lambda: ["simple_nn"])
    # DL regression models
    dl_regressors: List[str]= Field(default_factory=lambda: ["all"])

    use_receiver_id: bool      = False
    one_hot_receiver_id: bool  = False
    encode_event_rto: bool     = False
    threshold_cm: int          = Field(50, description="Localization threshold in cm")

    # === Hyperparameters for DL architectures ===
    dropout: float = Field(
        0.1, description="Dropout probability for DL models"
    )
    num_blocks: Optional[int] = Field(
        None, description="Number of blocks for SimpleNN and ConvMixer1D"
    )
    kernel_size: Optional[int] = Field(
        None, description="Kernel size for ConvMixer1D"
    )
    num_layers: Optional[int] = Field(
        None, description="Number of layers for TabularTransformer"
    )
    nhead: Optional[int] = Field(
        None, description="Number of attention heads for TabularTransformer"
    )
    dim_feedforward: Optional[int] = Field(
        None, description="Dimension of feedforward layer in Transformer"
    )

    # === Self-supervised pretraining parameters ===
    mask_ratio: float = Field(
        0.3, description="Fraction of CIR taps to mask during pretraining/adaptation"
    )
    pre_nhead: int = Field(
        4, description="Number of attention heads for the masked AE encoder"
    )
    pre_layers: int = Field(
        2, description="Number of Transformer encoder layers for masked AE"
    )

    fine_tune_epochs: int = Field(200)


    # === Hyperparameters for ML architectures ===
    xgb_n_estimators: int = Field(100, gt=1, description="XGBoost n_estimators")
    rf_n_estimators: int = Field(100, gt=1, description="RandomForest n_estimators")
    max_depth: Optional[int] = Field(None, gt=0, description="Tree max_depth")
    learning_rate: float = Field(0.001, gt=0, description="Learning rate for XGB")
    subsample: float = Field(1.0, ge=0, le=1.0, description="XGB subsample")
    colsample_bytree: float = Field(1.0, ge=0, le=1.0, description="XGB colsample_bytree")
    reg_alpha: float = Field(0.0, ge=0, description="XGB reg_alpha")
    reg_lambda: float = Field(1.0, ge=0, description="XGB reg_lambda")
    early_stopping_rounds: int = Field(10, ge=0, description="XGB early stopping")
    min_samples_leaf: int = Field(1, gt=0, description="RF min_samples_leaf")
    max_features: float = Field(1.0, ge=0, le=1.0, description="RF max_features")
    knn_n_neighbors: int = Field(5, gt=1, description="KNN n_neighbors")
    knn_weights: str = Field("uniform", description="KNN weight scheme")

    # ─── Validators for the “all” shorthand ─────────────────────────────
    @field_validator("algorithms", mode="before")
    def _validate_ml_algs(cls, v):
        if v == ["all"]:
            return v
        bad = set(v) - VALID_ML_CLASSIFIERS
        if bad:
            raise ValueError(
                f"Unknown ML classifier(s) {bad}. Valid: {VALID_ML_CLASSIFIERS} or ['all']"
            )
        return v

    @field_validator("dl_models", mode="before")
    def _validate_dl_cls(cls, v):
        if v == ["all"]:
            return v
        bad = set(v) - VALID_DL_CLASSIFIERS
        if bad:
            raise ValueError(
                f"Unknown DL classifier(s) {bad}. Valid: {VALID_DL_CLASSIFIERS} or ['all']"
            )
        return v

    @field_validator("dl_regressors", mode="before")
    def _validate_dl_reg(cls, v):
        if v == ["all"]:
            return v
        bad = set(v) - VALID_DL_REGRESSORS
        if bad:
            raise ValueError(
                f"Unknown DL regressor(s) {bad}. Valid: {VALID_DL_REGRESSORS} or ['all']"
            )
        return v
    

    @field_validator("regressors", mode="before")
    def _validate_ml_regs(cls, v):
        if v == ["all"]:
            return v
        bad = set(v) - VALID_ML_REGRESSORS
        if bad:
            raise ValueError(
                f"Unknown ML regressor(s) {bad}. Valid: {VALID_ML_REGRESSORS} or ['all']"
            )
        return v

    # ─── After-validation for DL classification requirements ─────────────
    '''
    @model_validator(mode="after")
    def _check_required_for_dl_classification(cls, m: "ModelConfig"):
        
        if m.task == "classification" and m.framework == "dl":
            if m.input_size is None or m.input_size <= 0:
                raise ValueError("model.input_size must be set (>0) for DL classification")
            if m.num_classes is None or m.num_classes <= 0:
                raise ValueError("model.num_classes must be set (>0) for DL classification")
        return m
    '''
    @model_validator(mode="after")
    def _check_required_for_dl_classification(cls, m: "ModelConfig"):
        # no-op: we’ll infer input_size & num_classes in main.py after loading data
        return m

# ─── Other sub-configs ───────────────────────────────────────
class TrainingConfig(BaseModel):
    num_epochs: int    = Field(..., gt=0)
    learning_rate: float = Field(..., gt=0)
    batch_size: int    = Field(..., gt=0)
    patience: int      = Field(..., ge=0)
    scheduler_monitor: Literal["mse", "median"] = Field(
        "mse",
        description="Which validation metric to feed into ReduceLROnPlateau: 'mse' or 'median'"
    )

class DataConfig(BaseModel):
    path: str
    coords_xlsx: Optional[str]      = None
    external_test_path: Optional[str]= None
    exclude_positions: List[str]     = Field(default_factory=list)
    parse_cir: bool                  =  Field(False,description="Whether to parse raw cir_cmplx arrays at CSV load time")
    internal_parquet: Optional[str]  = None
    external_parquet: Optional[str]  = None



class OutputConfig(BaseModel):
    directory: str = Field(...)

class LoggingConfig(BaseModel):
    level: str = Field(...)

# ─── Top-level config ────────────────────────────────────────────────────
class Config(BaseModel):
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    output: OutputConfig
    logging: LoggingConfig

    @field_validator("logging")
    def validate_logging_level(cls, v):
        valid = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.level.upper() not in valid:
            raise ValueError(f"Invalid logging level: {v.level}. Choose from {valid}.")
        return v


# ─── YAML loader ---------------------------------------------------------
def load_config(path: str) -> Config:
    try:
        with open(path, "r") as fh:
            cfg_dict = yaml.safe_load(fh)
        return Config(**cfg_dict)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise
