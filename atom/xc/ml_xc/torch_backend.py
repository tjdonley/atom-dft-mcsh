"""
Torch backend for XC models.
"""

from __future__ import annotations

import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type, List, Literal, Union


try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    TORCH_AVAILABLE = False

try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


from .base import XCBaseModel, ModelKind, ConfigExt, DataLoaderType, EvaluationMetrics
from ...data.data_manager import VxcDataLoader, ExcDataLoader




MODEL_NOT_NNMODULE_ERROR = \
    "Loaded model is not a torch.nn.Module, get {} instead."

DATA_LOADER_NOT_VXCDATALOADER_ERROR = \
    "parameter 'data_loader' must be a VxcDataLoader, get {} instead."
DATA_LOADER_NOT_EXCDATALOADER_ERROR = \
    "parameter 'data_loader' must be a ExcDataLoader, get {} instead."

# Training parameters error messages
FEATURE_NOT_IN_DATA_LOADER_ERROR = \
    "feature {} is not in data loader's features list {}. Check the features list of the data loader."
VXC_DATA_LOADER_CONTAINS_NO_SAMPLES_ERROR = \
    "VxcDataLoader contains no samples, please check the data loader to be non-empty."
EXC_DATA_LOADER_CONTAINS_NO_SAMPLES_ERROR = \
    "ExcDataLoader contains no samples, please check the data loader to be non-empty."
OPTIMIZER_NOT_VALID_ERROR = \
    "Invalid optimizer: {}. Use 'Adam' or 'SGD'."
CRITERION_NOT_VALID_ERROR = \
    "Invalid criterion: {}. Use 'L1Loss' or 'MSELoss'."

DATA_LOADER_TYPE_NOT_VALID_ERROR = \
    "Data loader type is not valid: {}. Use VxcDataLoader or ExcDataLoader."
WEIGHTS_NOT_SAME_AS_SAMPLES_ERROR = \
    "The number of weights must be the same as the number of samples, get {} weights and {} samples."

TORCH_NOT_AVAILABLE_FOR_TORCHXCMODEL_ERROR = \
    "TorchXCModel requires torch to be installed."
SKLEARN_NOT_AVAILABLE_FOR_MODEL_EVALUATION_ERROR = \
    "TorchXCModel requires sklearn to be installed to evaluate the model."


# Warnings
PATIENCE_NOT_NONE_WHEN_VAL_LOADER_IS_NONE_WARNING = \
    "WARNING: patience should be None when val_loader is None, get {} instead."




OptimizerType = Literal["Adam", "SGD"]
CriterionType = Literal["L1Loss", "MSELoss"]


class TorchXCModel(XCBaseModel):
    """
    Unified Torch model for both potential and energy.
    """

    model : nn.Module = field(init=False)

    def __init__(
        self,
        *,
        model_kind        : ModelKind,
        features_list     : List[str],
        model_cls         : Optional[Type["nn.Module"]] = None,
        model_init_kwargs : Optional[Dict[str, Any]]    = None,
        model_instance    : Optional["nn.Module"]       = None,
        model_name        : Optional[str]               = None,
        model_dir         : str                         = "./models",
        model_config      : Optional[Dict[str, Any]]    = None,
        config_ext        : ConfigExt                   = "json",
        device            : Optional[str]               = "cpu",
    ) -> None:

        if not TORCH_AVAILABLE:
            raise ImportError(TORCH_NOT_AVAILABLE_FOR_TORCHXCMODEL_ERROR)

        super().__init__(
            model_kind        = model_kind,
            features_list     = features_list,
            weights_ext       = "pth",
            config_ext        = config_ext,
            model_cls         = model_cls,
            model_init_kwargs = model_init_kwargs,
            model_instance    = model_instance,
            model_name        = model_name,
            model_dir         = model_dir,
            device            = device,
        )

        if not isinstance(self.model, nn.Module):
            raise TypeError(MODEL_NOT_NNMODULE_ERROR.format(type(self.model)))

        if device is not None:
            self.model.to(device)


    def train(
        self,
        train_loader     : DataLoaderType,
        val_loader       : Optional[DataLoaderType] = None,
        epochs           : int                      = 100,
        lr               : float                    = 1e-3,
        batch_size       : int                      = 256,
        optimizer        : str                      = "Adam",
        criterion        : str                      = "L1Loss",
        patience         : Optional[int]            = 100,
        optimizer_kwargs : Dict[str, Any]           = {},
        seed             : Optional[int]            = 42,
    ):

        # check the validation data loader
        if val_loader is not None:
            # check the validation data loader
            try:
                self._check_vxc_data_loader(val_loader)
            except ValueError as e:
                print(f"WARNING: {e}")
                print("\tThe validation loader will set to be None for this run.")
                val_loader = None
        
        # set patience to be larger than epochs if validation loader is not provided
        if val_loader is None and patience is not None:
            print(PATIENCE_NOT_NONE_WHEN_VAL_LOADER_IS_NONE_WARNING.format(patience))
            print("\tThe patience will be set to be larger than the number of epochs for this run.")
            patience = epochs + 1


        # check parameters
        self.check_epochs(epochs)
        self.check_lr(lr)
        self.check_patience(patience)

        # check and get the optimizer
        if optimizer == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, **optimizer_kwargs)
        elif optimizer == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, **optimizer_kwargs)
        else:
            raise ValueError(OPTIMIZER_NOT_VALID_ERROR.format(optimizer))
        

        # check and get the criterion
        if criterion == "L1Loss":
            criterion = torch.nn.L1Loss(reduction='none')
        elif criterion == "MSELoss":
            criterion = torch.nn.MSELoss(reduction='none')
        else:
            raise ValueError(CRITERION_NOT_VALID_ERROR.format(criterion))
        

        # train the model
        if self.model_kind == "potential":
            return self._train_potential(train_loader, val_loader, epochs, batch_size, patience, optimizer, criterion, seed)
        elif self.model_kind == "energy":
            return self._train_energy(train_loader, val_loader, epochs, batch_size, patience, optimizer, criterion, seed)
        else:
            raise ValueError(f"Invalid model kind: {self.model_kind}")


    def _train_potential(
        self,
        train_loader : VxcDataLoader,
        val_loader   : Optional[VxcDataLoader],
        epochs       : int,
        batch_size   : int,
        patience     : Optional[int],
        optimizer    : torch.optim.Optimizer,
        criterion    : torch.nn.Module,
        seed         : Optional[int] = None,
    ):
        # check and transform the training data loader
        self._check_vxc_data_loader(train_loader)
        train_loader = self.transform_to_torch_loader(train_loader, batch_size, shuffle=True, seed=seed)


        # train the model
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        # loss list
        train_loss_list = []
        val_loss_list = []

        for epoch in range(epochs):
            # training
            self.model.train()
            train_loss = 0.0

            batch_count = 0
            for batch_data in train_loader:
                train_X, train_y, train_weights = batch_data
                train_weights = train_weights.squeeze()

                optimizer.zero_grad()
                outputs = self.model(train_X)
                per_sample_loss = criterion(outputs, train_y).squeeze()

                assert train_weights.shape == per_sample_loss.shape, \
                    WEIGHTS_NOT_SAME_AS_SAMPLES_ERROR.format(train_weights.shape, per_sample_loss.shape)

                loss = (per_sample_loss * train_weights).mean()
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                batch_count += 1
            train_loss = train_loss / max(batch_count, 1)
            train_loss_list.append(train_loss)
            

            # validation
            self.model.eval()

            with torch.no_grad():

                val_loss = 0.0
                if val_loader is not None:
                    # compute the validation loss if val_loader is not None
                    # get the features and target data
                    X_val = torch.FloatTensor(val_loader.get_features_data(self.features_list))
                    y_val_true = torch.FloatTensor(val_loader.y)
                    val_weights = torch.FloatTensor(val_loader.weights_for_training)
                    
                    # get the predicted values
                    y_val_pred = self.model(X_val).detach()

                    # compute the weighted loss
                    per_sample_loss = criterion(y_val_pred, y_val_true).squeeze()
                    val_loss = (per_sample_loss * val_weights).mean()
                    val_loss_list.append(val_loss.item())
                else:
                    val_loss_list.append(-1.0)

            # early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            if (epoch + 1) % 1 == 0:
                print(f"Epoch {epoch+1:>4d}/{epochs}, Train Loss: {train_loss:.6e}, Val Loss: {val_loss:.6e}")

        # load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return self.model, train_loss_list, val_loss_list


    def _train_energy(
        self,
        train_loader : ExcDataLoader,
        val_loader   : Optional[ExcDataLoader],
        epochs       : int,
        batch_size   : int,
        patience     : Optional[int],
        optimizer    : Optional[torch.optim.Optimizer],
        criterion    : Optional[torch.nn.Module],
        seed         : Optional[int] = None,
    ):
        # check the training data loader
        self._check_exc_data_loader(train_loader)
        if train_loader.configuration_ids_per_sample is None:
            raise ValueError("ExcDataLoader must provide 'configuration_ids_per_sample' for energy training.")
        if train_loader.quadrature_nodes_per_sample is None:
            raise ValueError("ExcDataLoader must provide 'quadrature_nodes_per_sample' for energy training.")
        if train_loader.y_vxc is None:
            raise ValueError("ExcDataLoader must provide 'y_vxc' for energy training.")

        # check the validation data loader
        if val_loader is not None:
            self._check_exc_data_loader(val_loader)

        assert optimizer is not None, "optimizer must be provided for energy training"
        assert criterion is not None, "criterion must be provided for energy training"

        rho_idx  = self.features_list.index("rho")           if "rho"           in self.features_list else None
        grad_idx = self.features_list.index("grad_rho_norm") if "grad_rho_norm" in self.features_list else None
        lap_idx  = self.features_list.index("lap_rho")       if "lap_rho"       in self.features_list else None
        if rho_idx is None or grad_idx is None or lap_idx is None:
            raise ValueError("features_list must include 'rho', 'grad_rho_norm', and 'lap_rho' for energy training.")

        device = next(self.model.parameters()).device

        # train the model
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        train_loss_list = []
        val_loss_list = []

        for epoch in range(epochs):
            # training
            self.model.train()
            train_loss = 0.0
            batch_count = 0

            for batch in train_loader.iter_configuration_batches(batch_size, shuffle=True, seed=seed):
                optimizer.zero_grad()
                batch_loss = 0.0

                for config in batch:
                    X_pre = torch.as_tensor(config["X"], dtype=torch.float32, device=device)
                    y_exc_pre = torch.as_tensor(config["y_exc"], dtype=torch.float32, device=device)
                    y_vxc_pre = torch.as_tensor(config["y_vxc"], dtype=torch.float32, device=device)
                    weights_exc = torch.as_tensor(config["weights_exc"], dtype=torch.float32, device=device).squeeze()
                    weights_vxc = torch.as_tensor(
                        config.get("weights_vxc", config["weights_exc"])
                    , dtype=torch.float32, device=device).squeeze()

                    # Build differentiable preprocessing path: physical -> transformed -> model
                    X_phys = self._inverse_transform_features_torch(X_pre, train_loader)
                    X_phys = X_phys.detach().requires_grad_(True)
                    X_model_in = self._transform_features_torch(X_phys, train_loader)

                    y_exc_pred_pre = self.model(X_model_in)
                    per_sample_loss = criterion(y_exc_pred_pre, y_exc_pre).squeeze()
                    loss_exc = (per_sample_loss * weights_exc).mean()

                    # Inverse target transform to physical space for derivative-based v_xc
                    delta_e_phys = self._inverse_transform_targets_torch(y_exc_pred_pre, train_loader).squeeze()

                    # Compute partial derivatives of delta_e w.r.t. features (physical space)
                    grads = torch.autograd.grad(
                        delta_e_phys.sum(),
                        X_phys,
                        create_graph=True,
                        retain_graph=True
                    )[0]

                    dE_drho = grads[:, rho_idx]
                    dE_dgrad = grads[:, grad_idx]
                    dE_dlap = grads[:, lap_idx]

                    if "derivative_matrix" not in config or "laplacian_matrix" not in config:
                        raise ValueError("Each configuration must include 'derivative_matrix' and 'laplacian_matrix'.")

                    D = torch.as_tensor(config["derivative_matrix"], dtype=torch.float32, device=device)
                    L = torch.as_tensor(config["laplacian_matrix"], dtype=torch.float32, device=device)

                    rho = X_phys[:, rho_idx]
                    grad_rho = torch.matmul(D, rho)
                    grad_rho_norm = torch.abs(grad_rho) + 1e-12

                    div_term = torch.matmul(D, dE_dgrad)
                    lap_term = torch.matmul(L, dE_dlap)

                    delta_vxc_phys = dE_drho - (grad_rho / grad_rho_norm) * div_term + lap_term
                    delta_vxc_pred_pre = self._transform_targets_torch(
                        delta_vxc_phys.unsqueeze(1),
                        train_loader
                    )

                    per_sample_vxc_loss = criterion(delta_vxc_pred_pre, y_vxc_pre).squeeze()
                    loss_vxc = (per_sample_vxc_loss * weights_vxc).mean()

                    batch_loss = batch_loss + loss_exc + loss_vxc

                batch_loss = batch_loss / max(len(batch), 1)
                batch_loss.backward()
                optimizer.step()

                train_loss += batch_loss.item()
                batch_count += 1

            train_loss = train_loss / max(batch_count, 1)
            train_loss_list.append(train_loss)

            # validation
            self.model.eval()
            val_loss = 0.0
            if val_loader is not None:
                val_batch_count = 0
                for batch in val_loader.iter_configuration_batches(batch_size, shuffle=False):
                    batch_val_loss = 0.0
                    for config in batch:
                        X_pre = torch.as_tensor(config["X"], dtype=torch.float32, device=device)
                        y_exc_pre = torch.as_tensor(config["y_exc"], dtype=torch.float32, device=device)
                        y_vxc_pre = torch.as_tensor(config["y_vxc"], dtype=torch.float32, device=device)
                        weights_exc = torch.as_tensor(config["weights_exc"], dtype=torch.float32, device=device).squeeze()
                        weights_vxc = torch.as_tensor(
                            config.get("weights_vxc", config["weights_exc"])
                        , dtype=torch.float32, device=device).squeeze()

                        X_phys = self._inverse_transform_features_torch(X_pre, val_loader)
                        X_phys = X_phys.detach().requires_grad_(True)
                        X_model_in = self._transform_features_torch(X_phys, val_loader)

                        y_exc_pred_pre = self.model(X_model_in)
                        per_sample_loss = criterion(y_exc_pred_pre, y_exc_pre).squeeze()
                        loss_exc = (per_sample_loss * weights_exc).mean()

                        delta_e_phys = self._inverse_transform_targets_torch(y_exc_pred_pre, val_loader).squeeze()
                        grads = torch.autograd.grad(
                            delta_e_phys.sum(),
                            X_phys,
                            create_graph=False,
                            retain_graph=False
                        )[0]

                        dE_drho = grads[:, rho_idx]
                        dE_dgrad = grads[:, grad_idx]
                        dE_dlap = grads[:, lap_idx]

                        D = torch.as_tensor(config["derivative_matrix"], dtype=torch.float32, device=device)
                        L = torch.as_tensor(config["laplacian_matrix"], dtype=torch.float32, device=device)
                        rho = X_phys[:, rho_idx]
                        grad_rho = torch.matmul(D, rho)
                        grad_rho_norm = torch.abs(grad_rho) + 1e-12

                        div_term = torch.matmul(D, dE_dgrad)
                        lap_term = torch.matmul(L, dE_dlap)
                        delta_vxc_phys = dE_drho - (grad_rho / grad_rho_norm) * div_term + lap_term
                        delta_vxc_pred_pre = self._transform_targets_torch(
                            delta_vxc_phys.unsqueeze(1),
                            val_loader
                        )

                        per_sample_vxc_loss = criterion(delta_vxc_pred_pre, y_vxc_pre).squeeze()
                        loss_vxc = (per_sample_vxc_loss * weights_vxc).mean()

                        batch_val_loss = batch_val_loss + loss_exc + loss_vxc
                    batch_val_loss = batch_val_loss / max(len(batch), 1)
                    val_loss += batch_val_loss.item()
                    val_batch_count += 1

                val_loss = val_loss / max(val_batch_count, 1)
                val_loss_list.append(val_loss)
            else:
                val_loss_list.append(-1.0)

            # early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            if (epoch + 1) % 1 == 0:
                print(f"Epoch {epoch+1:>4d}/{epochs}, Train Loss: {train_loss:.6e}, Val Loss: {val_loss:.6e}")

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return self.model, train_loss_list, val_loss_list

    def _torch_symlog(self, x: torch.Tensor, linthresh: float) -> torch.Tensor:
        return torch.sign(x) * torch.log1p(torch.abs(x) / linthresh) * linthresh

    def _torch_symexp(self, y: torch.Tensor, linthresh: float) -> torch.Tensor:
        return torch.sign(y) * torch.expm1(torch.abs(y) / linthresh) * linthresh

    def _torch_apply_scaler(self, x: torch.Tensor, scaler) -> torch.Tensor:
        if scaler is None:
            return x
        if hasattr(scaler, "mean_"):
            center = scaler.mean_
        elif hasattr(scaler, "center_"):
            center = scaler.center_
        else:
            center = 0.0
        scale = scaler.scale_ if hasattr(scaler, "scale_") else 1.0
        center_t = torch.as_tensor(center, dtype=x.dtype, device=x.device)
        scale_t = torch.as_tensor(scale, dtype=x.dtype, device=x.device)
        return (x - center_t) / scale_t

    def _torch_inverse_scaler(self, x: torch.Tensor, scaler) -> torch.Tensor:
        if scaler is None:
            return x
        if hasattr(scaler, "mean_"):
            center = scaler.mean_
        elif hasattr(scaler, "center_"):
            center = scaler.center_
        else:
            center = 0.0
        scale = scaler.scale_ if hasattr(scaler, "scale_") else 1.0
        center_t = torch.as_tensor(center, dtype=x.dtype, device=x.device)
        scale_t = torch.as_tensor(scale, dtype=x.dtype, device=x.device)
        return x * scale_t + center_t

    def _transform_features_torch(self, x: torch.Tensor, data_loader: ExcDataLoader) -> torch.Tensor:
        if data_loader.use_symlog_features:
            x = self._torch_symlog(x, data_loader.linthresh_features)
        if data_loader.scale_features:
            x = self._torch_apply_scaler(x, data_loader.scaler_X)
        return x

    def _inverse_transform_features_torch(self, x: torch.Tensor, data_loader: ExcDataLoader) -> torch.Tensor:
        if data_loader.scale_features:
            x = self._torch_inverse_scaler(x, data_loader.scaler_X)
        if data_loader.use_symlog_features:
            x = self._torch_symexp(x, data_loader.linthresh_features)
        return x

    def _transform_targets_torch(self, y: torch.Tensor, data_loader: ExcDataLoader) -> torch.Tensor:
        if data_loader.use_symlog_targets:
            y = self._torch_symlog(y, data_loader.linthresh_targets)
        if data_loader.scale_targets:
            y = self._torch_apply_scaler(y, data_loader.scaler_y)
        return y

    def _inverse_transform_targets_torch(self, y: torch.Tensor, data_loader: ExcDataLoader) -> torch.Tensor:
        if data_loader.scale_targets:
            y = self._torch_inverse_scaler(y, data_loader.scaler_y)
        if data_loader.use_symlog_targets:
            y = self._torch_symexp(y, data_loader.linthresh_targets)
        return y
    

    def transform_to_torch_loader(
        self, 
        data_loader : DataLoaderType,
        batch_size  : int,
        shuffle     : bool,
        seed        : Optional[int] = None,
    ) -> torch.utils.data.DataLoader:

        # transform the data loader to a torch data loader
        from torch.utils.data import TensorDataset, DataLoader

        self.check_batch_size(batch_size)
        self.check_shuffle(shuffle)

        generator = None
        if seed is not None:
            torch.manual_seed(seed)
            generator = torch.Generator()
            generator.manual_seed(seed)

        if isinstance(data_loader, VxcDataLoader):
            X = torch.FloatTensor(data_loader.get_features_data(self.features_list))
            y = torch.FloatTensor(data_loader.y)
            weights = torch.FloatTensor(data_loader.weights_for_training)
            dataset = TensorDataset(X, y, weights)
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


        elif isinstance(data_loader, ExcDataLoader):
            X = torch.FloatTensor(data_loader.get_features_data(self.features_list))
            y = torch.FloatTensor(data_loader.y)
            weights = torch.FloatTensor(data_loader.weights_for_training)
            dataset = TensorDataset(X, y, weights)
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)
        else:
            raise ValueError(DATA_LOADER_TYPE_NOT_VALID_ERROR.format(type(data_loader)))




    def eval_model(
        self,
        data_loader : DataLoaderType,
    ) -> EvaluationMetrics:
        if isinstance(data_loader, VxcDataLoader):
            return self._eval_vxc_model(data_loader)
        elif isinstance(data_loader, ExcDataLoader):
            return self._eval_exc_model(data_loader)
        else:
            raise ValueError(DATA_LOADER_TYPE_NOT_VALID_ERROR.format(type(data_loader)))


    def forward(self, features: np.ndarray) -> np.ndarray:
        """
        Forward pass of the model.
        """
        features = torch.FloatTensor(features)
        return self.model(features).detach().numpy()



    def _eval_vxc_model(self, data_loader: VxcDataLoader) -> EvaluationMetrics:

        # check if sklearn is available
        assert SKLEARN_AVAILABLE, \
            SKLEARN_NOT_AVAILABLE_FOR_MODEL_EVALUATION_ERROR
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


        # check the data loader
        self._check_vxc_data_loader(data_loader)

        # get the features and target data
        self.model.eval()
        X = torch.FloatTensor(data_loader.get_features_data(self.features_list))
        y_pred = self.model(X).detach().numpy()
        y_true = data_loader.y
        
        # calculate the evaluation metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Calculate the maximum relative error
        max_abs_true = np.max(np.abs(y_true))
        if max_abs_true > 0:
            max_relative_error = np.max(np.abs(y_true - y_pred)) / max_abs_true
        else:
            max_relative_error = np.inf if np.any(y_true != y_pred) else 0.0


        return EvaluationMetrics(
            mae  = mae,
            rmse = rmse,
            r2   = r2,
            max_relative_error = max_relative_error
        )


    def save_model(
        self,
        model_dir: Optional[Union[str, Path]] = None,
        overwrite: bool = False,
    ) -> None:  
        """
        Save the model to a directory.
        """
        # save the model
        model_dir = model_dir if model_dir is not None else self.model_dir
        model_dir = Path(model_dir)

        model_filepath = model_dir / f"{self.model_name}_model.{self.weights_ext}"
        config_filepath = model_dir / f"{self.model_name}_config.{self.config_ext}"

        if not overwrite:
            existing = [p for p in (model_filepath, config_filepath) if p.exists()]
            if existing:
                print("The following files already exist:")
                for path in existing:
                    print(f" - {path}")
                choice = input("Overwrite existing files? [y/N]: ").strip().lower()
                if choice not in ("y", "yes"):
                    print("Canceled save.")
                    return

        model_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), model_filepath)
        with open(config_filepath, "w") as f:
            json.dump(self.model_init_kwargs, f, indent=2)


    def _eval_exc_model(self, data_loader: ExcDataLoader) -> EvaluationMetrics:
        # TODO: Implement energy evaluation
        raise NotImplementedError("Energy evaluation is not implemented")




    def _check_vxc_data_loader(self, data_loader: VxcDataLoader) -> None:
        # type check
        assert isinstance(data_loader, VxcDataLoader), \
            DATA_LOADER_NOT_VXCDATALOADER_ERROR.format(type(data_loader))
        for feature in self.features_list:
            assert feature in data_loader.features_list, \
                FEATURE_NOT_IN_DATA_LOADER_ERROR.format(feature, data_loader.features_list)
        if data_loader.n_samples == 0:
            raise ValueError(VXC_DATA_LOADER_CONTAINS_NO_SAMPLES_ERROR)



    def _check_exc_data_loader(self, data_loader: ExcDataLoader) -> None:
        # type check
        assert isinstance(data_loader, ExcDataLoader), \
            DATA_LOADER_NOT_EXCDATALOADER_ERROR.format(type(data_loader))
        for feature in self.features_list:
            assert feature in data_loader.features_list, \
                FEATURE_NOT_IN_DATA_LOADER_ERROR.format(feature, data_loader.features_list)
        if data_loader.n_samples == 0:
            raise ValueError(EXC_DATA_LOADER_CONTAINS_NO_SAMPLES_ERROR)


