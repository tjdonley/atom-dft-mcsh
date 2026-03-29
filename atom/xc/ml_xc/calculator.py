"""
Calculator class to hold a model plus its pipeline metadata.
"""

from __future__ import annotations


import json
import pickle
import numpy as np
import importlib

from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .base import XCBaseModel
from ...data.data_processing import DataProcessor
from ...data.data_manager import VxcDataLoader, ExcDataLoader

DataLoaderType = Union[VxcDataLoader, ExcDataLoader]

try:
    from sklearn.preprocessing import StandardScaler, RobustScaler
    SKLEARN_AVAILABLE = True
    ScalerType = Union[StandardScaler, RobustScaler]
except ImportError:
    SKLEARN_AVAILABLE = False
    ScalerType = Any


# Error messages
LINTHRESH_TARGETS_NOT_FOUND_ERROR = \
    "linthresh_targets is required when use_symlog_targets is True."
LINTHRESH_FEATURES_NOT_FOUND_ERROR = \
    "linthresh_features is required when use_symlog_features is True."

CALLING_PREDICT_EXC_FOR_MODEL_KIND_POTENTIAL_ERROR = \
    "Model kind is potential, but predict_exc is called."
CALLING_COMPUTE_ENERGY_FOR_MODEL_KIND_POTENTIAL_ERROR = \
    "Model kind is potential, but compute_energy is called."


@dataclass
class MLXCCalculator:
    """
    Backend-agnostic wrapper for model + preprocessing metadata.
    """

    model                  : XCBaseModel
    features_list          : List[str]
    target_functional      : str
    target_component       : str
    target_mode            : str
    reference_functional   : Optional[str]
    scale_features         : bool
    scale_targets          : bool
    scaler_type_features   : str
    scaler_type_targets    : str
    scaler_kwargs_features : Dict[str, Any] = field(default_factory=dict)
    scaler_kwargs_targets  : Dict[str, Any] = field(default_factory=dict)
    scaler_X               : ScalerType = None
    scaler_y               : ScalerType = None
    use_symlog_features    : bool = True
    use_symlog_targets     : bool = True
    linthresh_features     : Optional[float] = None
    linthresh_targets      : Optional[float] = None



    def print_info(self) -> None:
        """
        Print information about the calculator.
        """
        print("=" * 75)
        print("Machine Learning XC Calculator".center(75))
        print("=" * 75)
        print(f"\t model                  : {self.model.model_name}")
        print(f"\t features_list          : {self.features_list}")
        print(f"\t target_functional      : {self.target_functional}")
        print(f"\t target_component       : {self.target_component}")
        print(f"\t target_mode            : {self.target_mode}")
        print(f"\t reference_functional   : {self.reference_functional}")
        print(f"\t scale_features         : {self.scale_features}")
        print(f"\t scale_targets          : {self.scale_targets}")
        print(f"\t scaler_type_features   : {self.scaler_type_features}")
        print(f"\t scaler_type_targets    : {self.scaler_type_targets}")
        print(f"\t scaler_X               : {self.scaler_X}")
        print(f"\t scaler_y               : {self.scaler_y}")
        print(f"\t use_symlog_features    : {self.use_symlog_features}")
        print(f"\t use_symlog_targets     : {self.use_symlog_targets}")
        print(f"\t linthresh_features     : {self.linthresh_features}")
        print(f"\t linthresh_targets      : {self.linthresh_targets}")
        print()


    @classmethod
    def from_dataloader(
        cls,
        model       : XCBaseModel,
        data_loader : DataLoaderType,
    ) -> "MLXCCalculator":

        return cls(
            model                  = model,

            # parameters for documentation
            features_list          = data_loader.features_list,
            target_functional      = data_loader.target_functional,
            target_component       = data_loader.target_component,
            target_mode            = data_loader.target_mode,
            reference_functional   = data_loader.reference_functional,

            # parameters for scaling
            scale_features         = data_loader.scale_features,
            scale_targets          = data_loader.scale_targets,
            scaler_type_features   = data_loader.scaler_type_features,
            scaler_type_targets    = data_loader.scaler_type_targets,
            scaler_kwargs_features = data_loader.scaler_kwargs_features,
            scaler_kwargs_targets  = data_loader.scaler_kwargs_targets,
            scaler_X               = data_loader.scaler_X,
            scaler_y               = data_loader.scaler_y,

            # parameters for symlog transformation
            use_symlog_features    = data_loader.use_symlog_features,
            use_symlog_targets     = data_loader.use_symlog_targets,
            linthresh_features     = data_loader.linthresh_features,
            linthresh_targets      = data_loader.linthresh_targets,
        )


    @classmethod
    def load(
        cls,
        model_dir    : Union[str, Path],
        model_name   : str,
        model_cls    : Optional[type] = None,
        model_kind   : Optional[str] = None,
        features_list: Optional[List[str]] = None,
        device       : Optional[str] = "cpu",
    ) -> "MLXCCalculator":
        model_dir = Path(model_dir)
        config_path = model_dir / f"{model_name}_config.json"
        scalers_path = model_dir / f"{model_name}_scalers.pkl"
        metadata_path = model_dir / f"{model_name}_metadata.json"

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not scalers_path.exists():
            raise FileNotFoundError(f"Scalers file not found: {scalers_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with config_path.open("r", encoding="utf-8") as f:
            model_config = json.load(f)
        if isinstance(model_config, dict) and "model_config" in model_config: # This is a fix for the old model config format
            model_config = model_config["model_config"]
        with scalers_path.open("rb") as f:
            scalers = pickle.load(f)
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        if model_cls is None:
            model_class_path = metadata.get("model_class")
            if model_class_path is None:
                raise ValueError("'model_cls' is required when metadata does not include model_class.")
            module_path, class_name = model_class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            model_cls = getattr(module, class_name)

        if model_kind is None:
            model_kind = metadata.get("model_kind")
        if features_list is None:
            features_list = metadata.get("features_list", [])

        from .torch_backend import TorchXCModel
        model = TorchXCModel(
            model_kind        = model_kind,
            features_list     = features_list,
            model_cls         = model_cls,
            model_init_kwargs = model_config,
            model_name        = model_name,
            model_dir         = str(model_dir),
            device            = device,
        )

        model_file = model_dir / f"{model_name}_model.{model.weights_ext}"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        import torch
        state_dict = torch.load(model_file, map_location=device)
        model.model.load_state_dict(state_dict)

        return cls(
            model                  = model,
            features_list          = metadata.get("features_list", []),
            target_functional      = metadata.get("target_functional", ""),
            target_component       = metadata.get("target_component", ""),
            target_mode            = metadata.get("target_mode", ""),
            reference_functional   = metadata.get("reference_functional", None),
            scale_features         = metadata.get("scale_features", True),
            scale_targets          = metadata.get("scale_targets", True),
            scaler_type_features   = metadata.get("scaler_type_features", "robust"),
            scaler_type_targets    = metadata.get("scaler_type_targets", "standard"),
            scaler_kwargs_features = metadata.get("scaler_kwargs_features", {}),
            scaler_kwargs_targets  = metadata.get("scaler_kwargs_targets", {}),
            scaler_X               = scalers.get("scaler_X"),
            scaler_y               = scalers.get("scaler_y"),
            use_symlog_features    = metadata.get("use_symlog_features", True),
            use_symlog_targets     = metadata.get("use_symlog_targets", True),
            linthresh_features     = metadata.get("linthresh_features", None),
            linthresh_targets      = metadata.get("linthresh_targets", None),
        )


    def save(
        self, 
        model_dir: Optional[Union[str, Path]] = None,
        overwrite: bool = False
    ) -> None:

        """
        Save the calculator to a directory.
        
        Parameters
        ----------
        model_dir : Optional[Union[str, Path]]
            Directory to save the calculator.
        """

        model_dir = model_dir if model_dir is not None else self.model.model_dir
        model_dir = Path(model_dir)

        if not overwrite:
            existing = [p for p in self._resolve_model_paths(model_dir) if p.exists()]
            if existing:
                print("Saving MLXC calculator detected existing files.")
                print("The following files already exist:")
                for path in existing:
                    print(f" - {path}")
                choice = input("Overwrite existing files? [y/N]: ").strip().lower()
                if choice not in ("y", "yes"):
                    print("Canceled save.")
                    return

        model_dir.mkdir(parents=True, exist_ok=True)

        # save the model
        self.model.save_model(model_dir, overwrite=True)


        scalers_path = self._resolve_scalers_path(model_dir)
        scalers_data = {
            "scaler_X": self.scaler_X,
            "scaler_y": self.scaler_y,
        }
        with scalers_path.open("wb") as f:
            pickle.dump(scalers_data, f)

        metadata_path = self._resolve_metadata_path(model_dir)
        metadata_json = self._jsonable(self._to_metadata())
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata_json, f, indent=4)



    def inverse_transform_predictions(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Inverse transform predictions back to physical space.
        """
        y_pred = np.asarray(y_pred).copy()

        # Step 1: Inverse scaling (if scaler was used)
        if self.scaler_y is not None:
            y_pred = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
        # Step 2: Inverse symlog (if symlog was used)
        if self.use_symlog_targets:
            assert self.linthresh_targets is not None, \
                LINTHRESH_TARGETS_NOT_FOUND_ERROR
            y_pred = DataProcessor.symexp(y_pred, linthresh=self.linthresh_targets)

        return y_pred



    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features to the transformed space.
        """
        X = np.asarray(X).copy()
        

        # Step 1: Transform features (if symlog was used)
        if self.use_symlog_features:
            assert self.linthresh_features is not None, \
                LINTHRESH_FEATURES_NOT_FOUND_ERROR
            X = DataProcessor.symlog(X, linthresh=self.linthresh_features)

        # Step 2: Transform features (if scaler was used)
        if self.scaler_X is not None:
            X = self.scaler_X.transform(X)
        

        return X


    
    def predict_vxc(self, features: np.ndarray) -> np.ndarray:
        """
        Forward pass of the calculator.
        """
        if self.model_kind == "potential":
            features = self.transform_features(features)
            predictions = self.model.forward(features)
            predictions = self.inverse_transform_predictions(predictions)
            return predictions
        else:
            # TODO: Implement energy prediction
            raise ValueError(f"Model kind {self.model_kind} not supported")


    def predict_exc(self, features: np.ndarray) -> np.ndarray:
        """
        Forward pass of the calculator.
        """
        if self.model_kind == "potential":
            raise ValueError(CALLING_PREDICT_EXC_FOR_MODEL_KIND_POTENTIAL_ERROR)
        
        features    = self.transform_features(features)
        predictions = self.model.forward(features)
        predictions = self.inverse_transform_predictions(predictions)

        return predictions


    def compute_energy(self, density_data: Any) -> float:
        """
        Compute energy.
        """
        if self.model.model_kind == "potential":
            raise ValueError(CALLING_COMPUTE_ENERGY_FOR_MODEL_KIND_POTENTIAL_ERROR)

        raise NotImplementedError("Energy computation is not implemented for ML XC models")
        
        return self.model.forward(density_data)



    def _to_metadata(self) -> Dict[str, Any]:
        model_class = None
        if getattr(self.model, "model_cls", None) is not None:
            model_class = f"{self.model.model_cls.__module__}.{self.model.model_cls.__name__}"
        return {
            # parameters for documentation
            "features_list"          : self.features_list,
            "target_functional"      : self.target_functional,
            "target_component"       : self.target_component,
            "target_mode"            : self.target_mode,
            "reference_functional"   : self.reference_functional,
            "model_kind"             : getattr(self.model, "model_kind", None),
            "model_class"            : model_class,
            
            # parameters for scaling
            "scale_features"         : self.scale_features,
            "scale_targets"          : self.scale_targets,
            "scaler_type_features"   : self.scaler_type_features,
            "scaler_type_targets"    : self.scaler_type_targets,
            "scaler_kwargs_features" : self.scaler_kwargs_features,
            "scaler_kwargs_targets"  : self.scaler_kwargs_targets,
            
            # parameters for symlog transformation
            "use_symlog_features"    : self.use_symlog_features,
            "use_symlog_targets"     : self.use_symlog_targets,
            "linthresh_features"     : self.linthresh_features,
            "linthresh_targets"      : self.linthresh_targets,

            # scaler summaries (json friendly)
            "scaler_X_summary"       : self._scaler_summary(self.scaler_X),
            "scaler_Y_summary"       : self._scaler_summary(self.scaler_y),
        }


    @staticmethod
    def _jsonable(value: Any) -> Any:
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {k: MLXCCalculator._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [MLXCCalculator._jsonable(v) for v in value]
        return value


    @staticmethod
    def _scaler_summary(scaler: Any) -> Optional[Dict[str, Any]]:
        if scaler is None:
            return None
        summary = {
            "type": type(scaler).__name__,
        }
        # common sklearn scaler attributes
        for key in ("mean_", "scale_", "var_", "center_", "quantile_range_"):
            if hasattr(scaler, key):
                summary[key] = MLXCCalculator._jsonable(getattr(scaler, key))
        return summary


    def _resolve_scalers_path(self, model_dir: Union[str, Path]) -> Path:
        model_dir = Path(model_dir)
        return model_dir / f"{self.model.model_name}_scalers.pkl"


    def _resolve_metadata_path(self, model_dir: Union[str, Path]) -> Path:
        model_dir = Path(model_dir)
        return model_dir / f"{self.model.model_name}_metadata.json"


    def _resolve_model_paths(self, model_dir: Union[str, Path]) -> List[Path]:
        model_dir = Path(model_dir)
        model_file = model_dir / f"{self.model.model_name}_model.{self.model.weights_ext}"
        config_file = model_dir / f"{self.model.model_name}_config.{self.model.config_ext}"
        scalers_file = self._resolve_scalers_path(model_dir)
        metadata_file = self._resolve_metadata_path(model_dir)
        return [model_file, config_file, scalers_file, metadata_file]


    @property
    def model_kind(self) -> str:
        return self.model.model_kind

    