__author__ = "Qihao Cheng"

"""Visualization utilities for atomic DFT calculations."""

import numpy as np
from typing import Dict, Optional, List, Tuple
from .data_manager import AtomicDataset
from .data_loading import DataLoader

try:
    import matplotlib.colors as mcolors
    MATPLOTLIB_COLORS_AVAILABLE = True
except ImportError:
    mcolors = None
    MATPLOTLIB_COLORS_AVAILABLE = False


# Error messages
MATPLOTLIB_NOT_INSTALLED_ERROR = \
    "matplotlib is required for visualization. Install it with: pip install matplotlib."
SCF_FOLDER_NOT_FOUND_ERROR = \
    "SCF folder {} does not exist."

WEIGHTS_NOT_UPDATED_ERROR = \
    "weights_data is not available. Call dataset.update_weights_data(...) first."


TARGET_NAME_NOT_IN_VALID_LIST_ERROR = \
    "target_name must be one of {}, get {} instead."
FEATURE_NAME_NOT_STRING_ERROR = \
    "feature_name must be a string, get {} instead."
FEATURE_NAME_NOT_IN_DATASET_ERROR = \
    "feature_name '{}' is not in dataset.features_list: {}."
TARGET_FUNCTIONAL_REQUIRED_ERROR = \
    "target_functional must be provided when dataset has no scf_xc_functional."
TARGET_FUNCTIONAL_NOT_STRING_ERROR = \
    "target_functional must be a string, get {} instead."
TARGET_FUNCTIONAL_NOT_IN_DATASET_ERROR = \
    "target_functional '{}' is not in dataset: scf={}, forward_pass={}."
REFERENCE_FUNCTIONAL_NOT_STRING_ERROR = \
    "reference_functional must be a string, get {} instead."
REFERENCE_FUNCTIONAL_NOT_IN_DATASET_ERROR = \
    "reference_functional '{}' is not in dataset: scf={}, forward_pass={}."
TARGET_E_X_NOT_AVAILABLE_ERROR = \
    "e_x is not available: dataset.include_energy_density is False."
TARGET_E_C_NOT_AVAILABLE_ERROR = \
    "e_c is not available: dataset.include_energy_density is False."
TARGET_E_XC_NOT_AVAILABLE_ERROR = \
    "e_xc is not available: dataset.include_energy_density is False."
REFERENCE_E_X_NOT_AVAILABLE_ERROR = \
    "Reference e_x is not available: dataset.include_energy_density is False."
REFERENCE_E_C_NOT_AVAILABLE_ERROR = \
    "Reference e_c is not available: dataset.include_energy_density is False."
REFERENCE_E_XC_NOT_AVAILABLE_ERROR = \
    "Reference e_xc is not available: dataset.include_energy_density is False."



WARNING_REFERENCE_FUNCTIONAL_SAME_AS_TARGET = \
    "Warning: reference_functional is the same as target_functional; delta will be zero."
WARNING_NO_VALID_DATA = \
    "Warning: No valid data points for plotting"
WARNING_NO_VALID_DATA_LOG = \
    "Warning: No valid data points for plotting (all values must be positive for log scale)"
WARNING_ATOMIC_NUMBERS_MISMATCH_TRUE = \
    "Warning: atomic_numbers length doesn't match y_true length. Ignoring atomic_numbers."
WARNING_ATOMIC_NUMBERS_MISMATCH_RADIUS = \
    "Warning: atomic_numbers length doesn't match radius length. Ignoring atomic_numbers."



# Default labels/captions for common feature names.
DEFAULT_FEATURE_LABEL_MAP: Dict[str, str] = {
    "rho"              : r"Electron Density $\rho(r)$",
    "grad_rho"         : r"$\nabla\rho(r)$",
    "grad_rho_norm"    : r"$|\nabla\rho(r)|$",
    "grad_rho_reduced" : r"Reduced Density Gradient $s(r)$",
    "lap_rho"          : r"$\nabla^2\rho(r)$",
    "lap_rho_reduced"  : r"Reduced Laplacian $q(r)$",
    "tau"              : r"Kinetic Energy Density $\tau(r)$",
    "hartree"          : r"Hartree Potential $V_h(r)$",
    "lda_xc"           : r"LDA XC Potential $V_{xc}^{LDA}(r)$",
}

DEFAULT_FEATURE_CAPTION_MAP: Dict[str, str] = {
    "rho"              : r"Electron Density $\rho(r)$ vs Radius",
    "grad_rho"         : r"Density Gradient $\nabla\rho(r)$ vs Radius",
    "grad_rho_norm"    : r"Density Gradient Magnitude $|\nabla\rho(r)|$ vs Radius",
    "grad_rho_reduced" : r"Reduced Density Gradient $s(r)$ vs Radius",
    "lap_rho"          : r"Laplacian of Density $\nabla^2\rho(r)$ vs Radius",
    "lap_rho_reduced"  : r"Reduced Laplacian $q(r)$ vs Radius",
    "tau"              : r"Kinetic Energy Density $\tau(r)$ vs Radius",
    "hartree"          : r"Hartree Potential $V_h(r)$ vs Radius",
    "lda_xc"           : r"LDA XC Potential $V_{xc}^{LDA}(r)$ vs Radius",
}

DEFAULT_TARGET_TEXT_MAP: Dict[str, str] = {
    "v_x"  : "Exchange Potential",
    "v_c"  : "Correlation Potential",
    "v_xc" : "Exchange-Correlation Potential",
    "e_x"  : "Exchange Energy Density",
    "e_c"  : "Correlation Energy Density",
    "e_xc" : "Exchange-Correlation Energy Density",
}

DEFAULT_TARGET_LABEL_MAP: Dict[str, str] = {
    "v_x"  : r"Exchange Potential $v_x(r)$",
    "v_c"  : r"Correlation Potential $v_c(r)$",
    "v_xc" : r"Exchange-Correlation Potential $v_{xc}(r)$",
    "e_x"  : r"Exchange Energy Density $e_x(r)$",
    "e_c"  : r"Correlation Energy Density $e_c(r)$",
    "e_xc" : r"Exchange-Correlation Energy Density $e_{xc}(r)$",
}


class DataVisualizer:
    """
    Class for visualizing atomic DFT data.
    """


    @staticmethod
    def visualize_dataset_cutoffs(
        dataset  : AtomicDataset,
        save_path: Optional[str] = None
    ):
        """
        Visualize cutoff radii for all atoms.
        
        Parameters
        ----------
        dataset   : AtomicDataset
            Atomic dataset object
        save_path : Optional[str]
            Path to save the figure. If None, displays the figure.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print(MATPLOTLIB_NOT_INSTALLED_ERROR)
            return None, None

        if not MATPLOTLIB_COLORS_AVAILABLE or mcolors is None:
            print(MATPLOTLIB_NOT_INSTALLED_ERROR)
            return None, None
        
        cutoff_radii                     = dataset.cutoff_radii
        atomic_numbers_per_configuration = dataset.atomic_numbers_per_configuration
        
        # Sort by atomic number for better visualization
        sort_indices          = np.argsort(atomic_numbers_per_configuration)
        sorted_atomic_numbers = atomic_numbers_per_configuration[sort_indices]
        sorted_cutoff_radii   = cutoff_radii[sort_indices]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Cutoff radius vs atomic number
        ax1 = axes[0]
        ax1.plot(sorted_atomic_numbers, sorted_cutoff_radii, 'o-', markersize=4, linewidth=1.5)
        ax1.set_xlabel('Atomic Number', fontsize=12)
        ax1.set_ylabel('Cutoff Radius (a.u.)', fontsize=12)
        ax1.set_title('Cutoff Radius vs Atomic Number', fontsize=14)
        ax1.grid(True, alpha=0.6)
        ax1.set_xlim([min(sorted_atomic_numbers) - 1, max(sorted_atomic_numbers) + 1])
        # Disable offset notation to show actual values
        ax1.ticklabel_format(useOffset=False, style='plain')
        
        # Plot 2: Histogram of cutoff radii
        ax2 = axes[1]
        ax2.hist(cutoff_radii, bins=30, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Cutoff Radius (a.u.)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Distribution of Cutoff Radii', fontsize=14)
        ax2.grid(True, alpha=0.6, axis='y')
        ax2.axvline(np.mean(cutoff_radii), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(cutoff_radii):.4f}')
        ax2.legend()
        # Disable offset notation to show actual values
        ax2.ticklabel_format(useOffset=False, style='plain', axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            plt.show()
        
        return fig, axes


    @staticmethod
    def visualize_single_atom_cutoff(
        dataset       : AtomicDataset,
        atomic_number : int,
        save_path     : Optional[str] = None
    ):
        """
        Visualize cutoff for a single atom, showing the filtering criteria.
        This function is used to visualize the cutoff for a single atom's data at converged configuration.

        Parameters
        ----------
        dataset       : AtomicDataset
            Atomic dataset object
        atomic_number : int
            Atomic number to visualize
        save_path     : Optional[str]
            Path to save the figure. If None, displays the figure.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print(MATPLOTLIB_NOT_INSTALLED_ERROR)
            return None, None
        
        import os
        
        scf_folder_path = dataset.get_scf_folder_path(atomic_number)
        assert os.path.exists(scf_folder_path), \
            SCF_FOLDER_NOT_FOUND_ERROR.format(scf_folder_path)

        # Load data
        quadrature_nodes = DataLoader.load_quadrature_nodes_data(scf_folder_path)
        rho = DataLoader.load_rho_data(scf_folder_path, len(quadrature_nodes))
        v_x = DataLoader.load_v_x_data(scf_folder_path, len(quadrature_nodes))
        v_c = DataLoader.load_v_c_data(scf_folder_path, len(quadrature_nodes))
        radius_cutoff_rho_threshold = dataset.radius_cutoff_rho_threshold
        radius_cutoff_v_x_threshold = dataset.radius_cutoff_v_x_threshold
        radius_cutoff_v_c_threshold = dataset.radius_cutoff_v_c_threshold

        # Compute cutoff index
        cutoff_idx = DataLoader.compute_cutoff_index(
            rho = rho,
            v_x = v_x,
            v_c = v_c,
            rho_threshold = radius_cutoff_rho_threshold,
            v_x_threshold = radius_cutoff_v_x_threshold,
            v_c_threshold = radius_cutoff_v_c_threshold
        )
        cutoff_radius = quadrature_nodes[cutoff_idx - 1] if cutoff_idx > 0 else 0.0
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        
        # Plot rho
        ax1 = axes[0]
        ax1.semilogy(quadrature_nodes, np.abs(rho), 'b-', linewidth=1.5, label='rho')
        ax1.axhline(1e-6, color='r', linestyle='--', linewidth=2, label='Threshold (1e-6)')
        ax1.axvline(cutoff_radius, color='g', linestyle='--', linewidth=2, label=f'Cutoff ({cutoff_radius:.4f})')
        ax1.set_xlabel('Radius (a.u.)', fontsize=12)
        ax1.set_ylabel('Density (log scale)', fontsize=12)
        ax1.set_title(f'Atom {atomic_number}: Density vs Radius', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.6)
        
        # Plot v_x
        ax2 = axes[1]
        ax2.semilogy(quadrature_nodes, np.abs(v_x), 'b-', linewidth=1.5, label='|v_x|')
        ax2.axhline(1e-8, color='r', linestyle='--', linewidth=2, label='Threshold (1e-8)')
        ax2.axvline(cutoff_radius, color='g', linestyle='--', linewidth=2, label=f'Cutoff ({cutoff_radius:.4f})')
        ax2.set_xlabel('Radius (a.u.)', fontsize=12)
        ax2.set_ylabel('|v_x| (log scale)', fontsize=12)
        ax2.set_title(f'Atom {atomic_number}: Exchange Potential vs Radius', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.6)
        
        # Plot v_c
        ax3 = axes[2]
        ax3.semilogy(quadrature_nodes, np.abs(v_c), 'b-', linewidth=1.5, label='|v_c|')
        ax3.axhline(1e-8, color='r', linestyle='--', linewidth=2, label='Threshold (1e-8)')
        ax3.axvline(cutoff_radius, color='g', linestyle='--', linewidth=2, label=f'Cutoff ({cutoff_radius:.4f})')
        ax3.set_xlabel('Radius (a.u.)', fontsize=12)
        ax3.set_ylabel('|v_c| (log scale)', fontsize=12)
        ax3.set_title(f'Atom {atomic_number}: Correlation Potential vs Radius', fontsize=14)
        ax3.legend()
        ax3.grid(True, alpha=0.6)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            plt.show()
        
        return fig, axes


    @staticmethod
    def _auto_yscale(
        values: np.ndarray,
        linthresh: float = 1e-2
    ) -> Tuple[str, Optional[float]]:
        if np.all(values > 0):
            return "log", None
        if np.all(values < 0):
            return "neglog", None
        return "symlog", linthresh


    @classmethod
    def plot_all_features_by_atom(
        cls, 
        dataset       : AtomicDataset, 
        feature_names : Optional[List[str]]      = None, 
        label_map     : Optional[Dict[str, str]] = None,
        caption_map   : Optional[Dict[str, str]] = None,
        save_path     : Optional[str]            = None,
    ):
        try:
            import matplotlib.pyplot as plt
            from matplotlib.ticker import FuncFormatter
        except ImportError:
            print(MATPLOTLIB_NOT_INSTALLED_ERROR)
            return None, None

        radius         = dataset.quadrature_nodes
        atomic_numbers = dataset.atomic_numbers_per_sample

        if isinstance(feature_names, str):
            feature_names = [feature_names]
        if feature_names is None:
            feature_names = list(dataset.features_list)
        label_map = {**DEFAULT_FEATURE_LABEL_MAP, **(label_map or {})}
        caption_map = {**DEFAULT_FEATURE_CAPTION_MAP, **(caption_map or {})}

        unique_atoms = np.sort(np.unique(atomic_numbers))
        cmap = plt.cm.get_cmap('tab20' if len(unique_atoms) <= 20 else 'viridis')
        norm = mcolors.Normalize(vmin=unique_atoms.min(), vmax=unique_atoms.max())

        n_features = len(feature_names)
        ncols = min(4, n_features)
        nrows = int(np.ceil(n_features / ncols))

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(6 * ncols, 4 * nrows),
            constrained_layout=True
        )
        axes = np.atleast_2d(axes)

        for i in range(n_features):
            ax = axes[i // ncols, i % ncols]
            y = dataset.get_features_data(feature_names[i])
            yscale, linthresh = cls._auto_yscale(y)
            if yscale == "neglog":
                # Plot in transformed space but label ticks in original values.
                eps = np.finfo(float).tiny
                y = -np.log(np.clip(-y, eps, None))

            for atom_z in unique_atoms:
                mask = atomic_numbers == atom_z
                sort_idx = np.argsort(radius[mask])
                ax.plot(
                    radius[mask][sort_idx],
                    y[mask][sort_idx],
                    color=cmap(norm(atom_z)),
                    linewidth=1.0,
                    alpha=0.5
                )

            ax.set_xlabel('Radius (a.u.)', fontsize=12)
            feature_name = feature_names[i]
            feature_label = label_map.get(feature_name, feature_name)
            feature_caption = caption_map.get(feature_name, f'{feature_label} vs Radius')
            ax.set_ylabel(feature_label, fontsize=12)
            ax.set_xscale('log')
            if yscale == 'symlog':
                ax.set_yscale('symlog', linthresh=linthresh)
            elif yscale == "log":
                ax.set_yscale(yscale)
            else:
                ax.set_yscale("linear")
            if yscale == "neglog":
                def _neglog_formatter(v: float, _: int) -> str:
                    original = -np.exp(-v)
                    if original >= 0:
                        return ""
                    power = int(np.round(np.log10(-original)))
                    return rf"$-10^{{{power}}}$"

                ax.yaxis.set_major_formatter(FuncFormatter(_neglog_formatter))
            ax.set_title(feature_caption, fontsize=13, fontweight='bold')
            ax.minorticks_on()
            ax.grid(True, which='major', linestyle='-', linewidth=0.9, alpha=0.8)
            ax.grid(True, which='minor', linestyle='--', linewidth=0.6, alpha=0.5)

        for j in range(n_features, nrows * ncols):
            axes[j // ncols, j % ncols].axis('off')

        cbar = fig.colorbar(
            plt.cm.ScalarMappable(cmap=cmap, norm=norm),
            ax=axes,
            pad=0.02,
            shrink=0.9
        )
        cbar.set_label('Atomic Number (Z)', fontsize=11, labelpad=10)
        ticks = unique_atoms if len(unique_atoms) <= 20 else unique_atoms[np.linspace(0, len(unique_atoms) - 1, min(10, len(unique_atoms)), dtype=int)]
        cbar.set_ticks(ticks)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        plt.show()
        plt.close()
        return fig, axes


    @classmethod
    def plot_all_targets_by_atom(
        cls,
        dataset   : AtomicDataset,
        save_path : Optional[str] = None
    ):
        """
        Visualize all targets for different atoms from an AtomicDataset.
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.ticker import FuncFormatter
        except ImportError:
            print(MATPLOTLIB_NOT_INSTALLED_ERROR)
            return None, None
        
        radius         = dataset.quadrature_nodes
        atomic_numbers = dataset.atomic_numbers_per_sample

        unique_atoms = np.sort(np.unique(atomic_numbers))
        cmap = plt.cm.get_cmap('tab20' if len(unique_atoms) <= 20 else 'viridis')
        norm = mcolors.Normalize(vmin=unique_atoms.min(), vmax=unique_atoms.max())

        component_map = {
            "v_x": (0, r"Exchange Potential $V_x(r)$", "Exchange Potential vs Radius"),
            "v_c": (1, r"Correlation Potential $V_c(r)$", "Correlation Potential vs Radius"),
        }
        if dataset.include_energy_density:
            component_map.update({
                "e_x": (2, r"Exchange Energy Density $e_x(r)$", "Exchange Energy Density vs Radius"),
                "e_c": (3, r"Correlation Energy Density $e_c(r)$", "Correlation Energy Density vs Radius"),
            })

        components = list(component_map.keys())
        functionals = [dataset.scf_xc_functional] + list(dataset.forward_pass_xc_functional_list)

        nrows = len(functionals)
        ncols = len(components)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(6 * ncols, 4 * nrows),
            constrained_layout=True
        )
        axes = np.atleast_2d(axes)

        for row_idx, functional in enumerate(functionals):
            xc_data = dataset.get_xc_data(functional)
            for col_idx, component in enumerate(components):
                data_index, y_label, caption = component_map[component]
                y = xc_data[data_index]
                ax = axes[row_idx, col_idx]

                if y is None:
                    ax.axis('off')
                    continue

                yscale, linthresh = cls._auto_yscale(y)
                if yscale == "neglog":
                    eps = np.finfo(float).tiny
                    y = -np.log(np.clip(-y, eps, None))

                for atom_z in unique_atoms:
                    mask = atomic_numbers == atom_z
                    sort_idx = np.argsort(radius[mask])
                    ax.plot(
                        radius[mask][sort_idx],
                        y[mask][sort_idx],
                        color=cmap(norm(atom_z)),
                        linewidth=1.0,
                        alpha=0.5
                    )

                ax.set_xlabel('Radius (a.u.)', fontsize=12)
                ax.set_ylabel(y_label, fontsize=12)
                ax.set_title(f"{functional}: {caption}", fontsize=13, fontweight='bold')
                ax.set_xscale('log')
                if yscale == 'symlog':
                    ax.set_yscale('symlog', linthresh=linthresh)
                elif yscale == "log":
                    ax.set_yscale(yscale)
                else:
                    ax.set_yscale("linear")
                if yscale == "neglog":
                    def _neglog_formatter(v: float, _: int) -> str:
                        original = -np.exp(-v)
                        if original >= 0:
                            return ""
                        power = int(np.round(np.log10(-original)))
                        return rf"$-10^{{{power}}}$"

                    ax.yaxis.set_major_formatter(FuncFormatter(_neglog_formatter))
                ax.grid(True, alpha=0.6, which='both')

        cbar = fig.colorbar(
            plt.cm.ScalarMappable(cmap=cmap, norm=norm),
            ax=axes,
            pad=0.02,
            shrink=0.9
        )
        cbar.set_label('Atomic Number (Z)', fontsize=11, labelpad=10)
        ticks = unique_atoms if len(unique_atoms) <= 20 else unique_atoms[np.linspace(0, len(unique_atoms) - 1, min(10, len(unique_atoms)), dtype=int)]
        cbar.set_ticks(ticks)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        plt.show()
        plt.close()

        return fig, axes


    @classmethod
    def plot_target_by_feature(
        cls,
        dataset              : AtomicDataset,
        target_name          : str,
        feature_name         : str,
        target_functional    : Optional[str] = None,
        reference_functional : Optional[str] = None,
        save_path            : Optional[str] = None,
    ):
        """
        Visualize a target vs a feature for different atoms from an AtomicDataset.
        
        Parameters
        ----------
        dataset : AtomicDataset
            Atomic dataset object containing features and XC data.
        target_name : str
            Target component name. Supported values:
            "v_x", "v_c", "v_xc", "e_x", "e_c", "e_xc".
        feature_name : str
            Feature name to plot on the x-axis (must be in dataset.features_list).
        target_functional : Optional[str]
            XC functional used to construct the target. If None, defaults to
            dataset.target_functional when available. If provided, it must exist
            in the dataset (SCF or forward-pass list).
        reference_functional : Optional[str]
            Reference XC functional for delta targets. If None, plots absolute targets.
            If provided, it must exist in the dataset. If it is the same as
            target_functional, a warning will be printed.
        save_path : Optional[str]
            Path to save the figure. If None, displays the figure.
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.ticker import FuncFormatter
        except ImportError:
            print(MATPLOTLIB_NOT_INSTALLED_ERROR)
            return None, None
        
        target_key, target_functional, reference_functional = cls._validate_target_by_feature_inputs(
            dataset=dataset,
            target_name=target_name,
            feature_name=feature_name,
            target_functional=target_functional,
            reference_functional=reference_functional,
        )
        
        feature_values = np.asarray(dataset.get_features_data(feature_name)).flatten()
        atomic_numbers = np.asarray(dataset.atomic_numbers_per_sample).flatten()
        
        xc_data = dataset.get_xc_data(target_functional)
        if target_key == "v_x":
            target_values = xc_data[0]
        elif target_key == "v_c":
            target_values = xc_data[1]
        elif target_key == "v_xc":
            target_values = xc_data[0] + xc_data[1]
        elif target_key == "e_x":
            if not dataset.include_energy_density or xc_data[2] is None:
                raise ValueError(TARGET_E_X_NOT_AVAILABLE_ERROR)
            target_values = xc_data[2]
        elif target_key == "e_c":
            if not dataset.include_energy_density or xc_data[3] is None:
                raise ValueError(TARGET_E_C_NOT_AVAILABLE_ERROR)
            target_values = xc_data[3]
        else:  # e_xc
            if not dataset.include_energy_density or xc_data[2] is None or xc_data[3] is None:
                raise ValueError(TARGET_E_XC_NOT_AVAILABLE_ERROR)
            target_values = xc_data[2] + xc_data[3]
        
        target_values = np.asarray(target_values).flatten()
        
        if reference_functional is not None:
            ref_xc_data = dataset.get_xc_data(reference_functional)
            if target_key == "v_x":
                ref_values = ref_xc_data[0]
            elif target_key == "v_c":
                ref_values = ref_xc_data[1]
            elif target_key == "v_xc":
                ref_values = ref_xc_data[0] + ref_xc_data[1]
            elif target_key == "e_x":
                if not dataset.include_energy_density or ref_xc_data[2] is None:
                    raise ValueError(REFERENCE_E_X_NOT_AVAILABLE_ERROR)
                ref_values = ref_xc_data[2]
            elif target_key == "e_c":
                if not dataset.include_energy_density or ref_xc_data[3] is None:
                    raise ValueError(REFERENCE_E_C_NOT_AVAILABLE_ERROR)
                ref_values = ref_xc_data[3]
            else:  # e_xc
                if not dataset.include_energy_density or ref_xc_data[2] is None or ref_xc_data[3] is None:
                    raise ValueError(REFERENCE_E_XC_NOT_AVAILABLE_ERROR)
                ref_values = ref_xc_data[2] + ref_xc_data[3]
            target_values = target_values - np.asarray(ref_values).flatten()
        
        if len(feature_values) != len(target_values) or len(feature_values) != len(atomic_numbers):
            print(
                f"Error: Mismatch in array lengths: "
                f"feature={len(feature_values)}, target={len(target_values)}, atomic_numbers={len(atomic_numbers)}"
            )
            return None, None
        
        valid_mask = np.isfinite(feature_values) & np.isfinite(target_values)
        if not np.any(valid_mask):
            print(WARNING_NO_VALID_DATA)
            return None, None
        
        x = feature_values[valid_mask]
        y = target_values[valid_mask]
        z = atomic_numbers[valid_mask]
        
        xscale, xlinthresh = cls._auto_yscale(x)
        yscale, ylinthresh = cls._auto_yscale(y)
        
        if xscale == "neglog":
            eps = np.finfo(float).tiny
            x = -np.log(np.clip(-x, eps, None))
        if yscale == "neglog":
            eps = np.finfo(float).tiny
            y = -np.log(np.clip(-y, eps, None))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        unique_atoms = np.sort(np.unique(z))
        cmap = plt.cm.get_cmap('tab20' if len(unique_atoms) <= 20 else 'viridis')
        norm = mcolors.Normalize(vmin=unique_atoms.min(), vmax=unique_atoms.max())
        
        scatter = ax.scatter(
            x, y,
            c=z,
            cmap=cmap,
            norm=norm,
            s=6,
            alpha=0.5,
            edgecolors='none'
        )
        
        feature_label = DEFAULT_FEATURE_LABEL_MAP.get(feature_name, feature_name)
        target_label = DEFAULT_TARGET_LABEL_MAP.get(target_key, target_key)
        target_text = DEFAULT_TARGET_TEXT_MAP.get(target_key, target_key)
        
        if xscale == 'symlog':
            ax.set_xscale('symlog', linthresh=xlinthresh)
        elif xscale == 'log':
            ax.set_xscale('log')
        else:
            ax.set_xscale('linear')
        if xscale == "neglog":
            def _neglog_x_formatter(v: float, _: int) -> str:
                original = -np.exp(-v)
                if original >= 0:
                    return ""
                power = int(np.round(np.log10(-original)))
                return rf"$-10^{{{power}}}$"
            ax.xaxis.set_major_formatter(FuncFormatter(_neglog_x_formatter))
        
        if yscale == 'symlog':
            ax.set_yscale('symlog', linthresh=ylinthresh)
        elif yscale == 'log':
            ax.set_yscale('log')
        else:
            ax.set_yscale('linear')
        if yscale == "neglog":
            def _neglog_y_formatter(v: float, _: int) -> str:
                original = -np.exp(-v)
                if original >= 0:
                    return ""
                power = int(np.round(np.log10(-original)))
                return rf"$-10^{{{power}}}$"
            ax.yaxis.set_major_formatter(FuncFormatter(_neglog_y_formatter))
        
        xlabel, ylabel, title = cls._format_target_vs_feature_labels(
            feature_label        = feature_label,
            target_text          = target_text,
            target_key           = target_key,
            target_functional    = target_functional,
            reference_functional = reference_functional,
        )
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.minorticks_on()
        ax.grid(True, which='major', linestyle='-', linewidth=0.9, alpha=0.8)
        ax.grid(True, which='minor', linestyle='--', linewidth=0.6, alpha=0.5)
        
        cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label('Atomic Number (Z)', fontsize=11, labelpad=10)
        ticks = unique_atoms if len(unique_atoms) <= 20 else unique_atoms[
            np.linspace(0, len(unique_atoms) - 1, min(10, len(unique_atoms)), dtype=int)
        ]
        cbar.set_ticks(ticks)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        plt.show()
        
        return fig, ax



    @staticmethod
    def plot_symlog_and_symexp(
        linthresh: float = 0.002,
        save_path: Optional[str] = None
    ):
        """
        Demo of symlog and symexp transforms on symmetric data.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print(MATPLOTLIB_NOT_INSTALLED_ERROR)
            return None, None

        from .data_processing import DataProcessor

        x_pos = np.logspace(-6, 2, 400)
        x = np.concatenate([-x_pos[::-1], x_pos])

        y_symlog = DataProcessor.symlog(x, linthresh=linthresh)
        x_recovered = DataProcessor.symexp(y_symlog, linthresh=linthresh)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

        axes[0].plot(x, y_symlog, linewidth=1.2)
        axes[0].scatter(x, y_symlog, color='red', alpha=0.5, s=10)
        axes[0].axvline(-linthresh, color='gray', linestyle='--', linewidth=1.0)
        axes[0].axvline(linthresh, color='gray', linestyle='--', linewidth=1.0)
        axes[0].set_title("Symlog: y = symlog(x)", fontsize=12, fontweight='bold')
        axes[0].set_xlabel("x", fontsize=11)
        axes[0].set_ylabel("symlog(y)", fontsize=11)
        axes[0].grid(True, alpha=0.6, which='both')

        axes[1].plot(y_symlog, x_recovered, linewidth=1.2)
        axes[1].scatter(y_symlog, x_recovered, color='red', alpha=0.5, s=10)
        axes[1].axhline(-linthresh, color='gray', linestyle='--', linewidth=1.0)
        axes[1].axhline(linthresh, color='gray', linestyle='--', linewidth=1.0)
        axes[1].set_title("Symexp: y = symexp(x)", fontsize=12, fontweight='bold')
        axes[1].set_xlabel("x", fontsize=11)
        axes[1].set_ylabel("y", fontsize=11)
        axes[1].grid(True, alpha=0.6, which='both')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        plt.show()

        return fig, axes



    @staticmethod
    def plot_weights_by_atom(
        dataset   : AtomicDataset,
        save_path : Optional[str] = None
    ):
        """
        Visualize weights for different atoms from an AtomicDataset.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print(MATPLOTLIB_NOT_INSTALLED_ERROR)
            return None, None
        if not hasattr(dataset, "potential_weights_data_is_updated") or not dataset.potential_weights_data_is_updated:
            raise ValueError(WEIGHTS_NOT_UPDATED_ERROR)

        weights = np.asarray(dataset.potential_weights_data)
        atomic_numbers = np.asarray(dataset.atomic_numbers_per_sample)
        radius_data = np.asarray(dataset.quadrature_nodes)

        if len(weights) != len(atomic_numbers) or len(weights) != len(radius_data):
            print(
                f"Error: Mismatch in array lengths: "
                f"weights={len(weights)}, atomic_numbers={len(atomic_numbers)}, radius={len(radius_data)}"
            )
            return None, None
        
        unique_atoms = np.sort(np.unique(atomic_numbers))
        cmap = plt.cm.get_cmap('tab20' if len(unique_atoms) <= 20 else 'viridis')
        norm = mcolors.Normalize(vmin=unique_atoms.min(), vmax=unique_atoms.max())
        
        fig, ax = plt.subplots(figsize=(12, 8))
        for atom_z in unique_atoms:
            mask = (atomic_numbers == atom_z)
            sort_idx = np.argsort(radius_data[mask])
            ax.plot(radius_data[mask][sort_idx], weights[mask][sort_idx], 
                color=cmap(norm(atom_z)), linewidth=1.5, alpha=0.7, label=f'Z={atom_z}')
        
        ax.set_xlabel('Radius (a.u.)', fontsize=12)
        ax.set_ylabel('Weight', fontsize=12)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title('Weights vs Radius for Different Atoms', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.6, which='both')
        
        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, pad=0.02)
        cbar.set_label('Atomic Number (Z)', fontsize=11, labelpad=10)
        cbar.set_ticks(unique_atoms if len(unique_atoms) <= 20 else 
                    unique_atoms[np.linspace(0, len(unique_atoms)-1, min(10, len(unique_atoms)), dtype=int)])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        plt.show()



    @staticmethod
    def plot_predictions_vs_true(
        y_true, y_pred, model_name="", target_name="Potential", 
        save_path=None, figsize=(10, 8), alpha=0.5, s=1, scale='auto', linthresh=0.01,
        atomic_numbers=None,
    ):
        """
        Plot predicted values vs true values with y=x reference line.
        
        Parameters
        ----------
        y_true : array-like
            True values (will be on x-axis)
        y_pred : array-like
            Predicted values (will be on y-axis)
        model_name : str
            Name of the model (for title)
        target_name : str
            Name of the target variable (for axis labels)
        save_path : str, optional
            Path to save the figure. If None, displays the figure.
        figsize : tuple
            Figure size (width, height)
        alpha : float
            Transparency of scatter points
        s : float
            Size of scatter points
        scale : str
            Scale type: 'auto', 'log', 'symlog', or 'linear'. 
            If 'auto', automatically chooses based on data (symlog if negative values exist, else log)
        linthresh : float
            Linear threshold for symlog scale (only used if scale='symlog' or 'auto' with negative values)
        atomic_numbers : array-like, optional
            Atomic numbers corresponding to each data point. If provided, points will be colored
            by atomic number with a colorbar showing the mapping.
        label_atoms : bool
            Deprecated parameter (kept for backward compatibility). If atomic_numbers is provided,
            points are automatically colored by atomic number.
        
        Returns
        -------
        fig, ax : matplotlib figure and axes
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import Normalize
            from matplotlib.cm import ScalarMappable
        except ImportError:
            print("matplotlib is required for visualization. Install it with: pip install matplotlib")
            return None, None
        
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        # Process atomic_numbers if provided
        atomic_numbers_valid = None
        if atomic_numbers is not None:
            atomic_numbers = np.asarray(atomic_numbers).flatten()
            if len(atomic_numbers) != len(y_true):
                print(WARNING_ATOMIC_NUMBERS_MISMATCH_TRUE)
                atomic_numbers = None
        
        # Filter out invalid values (NaN, inf)
        valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        if atomic_numbers is not None:
            atomic_numbers_valid = atomic_numbers[valid_mask]
        
        if len(y_true_valid) == 0:
            print(WARNING_NO_VALID_DATA)
            return None, None
        
        # Determine scale type
        if scale == 'auto':
            # Check if data contains negative or zero values
            has_negative = np.any(y_true_valid <= 0) or np.any(y_pred_valid <= 0)
            if has_negative:
                scale = 'symlog'
            else:
                scale = 'log'
        
        # For log scale, filter out non-positive values
        if scale == 'log':
            valid_mask_log = (y_true_valid > 0) & (y_pred_valid > 0)
            y_true_valid = y_true_valid[valid_mask_log]
            y_pred_valid = y_pred_valid[valid_mask_log]
            if atomic_numbers_valid is not None:
                atomic_numbers_valid = atomic_numbers_valid[valid_mask_log]
            if len(y_true_valid) == 0:
                print(WARNING_NO_VALID_DATA_LOG)
                return None, None
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Scatter plot with color mapping if atomic_numbers provided
        if atomic_numbers_valid is not None:
            # Use colormap to color points by atomic number
            scatter = ax.scatter(y_true_valid, y_pred_valid, c=atomic_numbers_valid, 
                            alpha=alpha, s=s, cmap='viridis', label='Predictions')
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Atomic Number', fontsize=11)
        else:
            # Single color if no atomic_numbers
            ax.scatter(y_true_valid, y_pred_valid, alpha=alpha, s=s, label='Predictions')
        
        # y=x reference line
        min_val = min(np.min(y_true_valid), np.min(y_pred_valid))
        max_val = max(np.max(y_true_valid), np.max(y_pred_valid))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
                label='y=x (Perfect Prediction)', alpha=0.8)
        
        # Set scale for both axes
        if scale == 'symlog':
            ax.set_xscale('symlog', linthresh=linthresh)
            ax.set_yscale('symlog', linthresh=linthresh)
        elif scale == 'log':
            ax.set_xscale('log')
            ax.set_yscale('log')
        # else: linear scale (default, no need to set)
        
        # Labels and title
        ax.set_xlabel(f'True {target_name}', fontsize=12)
        ax.set_ylabel(f'Predicted {target_name}', fontsize=12)
        title = f'Predicted vs True {target_name}'
        if model_name:
            title += f' ({model_name})'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Grid
        ax.grid(True, alpha=0.6, which='both')
        
        # Legend
        ax.legend(fontsize=10)
        
        # Calculate and display R² on the plot
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true_valid, y_pred_valid)
        ax.text(0.05, 0.95, f'R² = {r2:.8f}', transform=ax.transAxes, 
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            plt.show()
        
        return fig, ax



    @staticmethod
    def plot_difference_vs_radius(radius, y_true, y_pred, atomic_numbers=None, 
                                model_name="", target_name="Potential",
                                save_path=None, figsize=(10, 8), alpha=0.5, s=1, 
                                linthresh=0.01
    ):
        """
        Plot prediction error (difference) vs radius with symlog scale.
        
        Parameters
        ----------
        radius : array-like
            Radius values (will be on x-axis)
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values
        atomic_numbers : array-like, optional
            Atomic numbers corresponding to each data point. If provided, points will be colored
            by atomic number with a colorbar showing the mapping.
        model_name : str
            Name of the model (for title)
        target_name : str
            Name of the target variable (for axis labels)
        save_path : str, optional
            Path to save the figure. If None, displays the figure.
        figsize : tuple
            Figure size (width, height)
        alpha : float
            Transparency of scatter points
        s : float
            Size of scatter points
        linthresh : float
            Linear threshold for symlog scale
        
        Returns
        -------
        fig, ax : matplotlib figure and axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is required for visualization. Install it with: pip install matplotlib")
            return None, None
        
        radius = np.asarray(radius).flatten()
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        # Calculate difference
        difference = y_true - y_pred
        
        # Process atomic_numbers if provided
        atomic_numbers_valid = None
        if atomic_numbers is not None:
            atomic_numbers = np.asarray(atomic_numbers).flatten()
            if len(atomic_numbers) != len(radius):
                print(WARNING_ATOMIC_NUMBERS_MISMATCH_RADIUS)
                atomic_numbers = None
        
        # Filter out invalid values (NaN, inf)
        valid_mask = np.isfinite(radius) & np.isfinite(difference)
        radius_valid = radius[valid_mask]
        difference_valid = difference[valid_mask]
        if atomic_numbers is not None:
            atomic_numbers_valid = atomic_numbers[valid_mask]
        
        if len(radius_valid) == 0:
            print(WARNING_NO_VALID_DATA)
            return None, None
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Scatter plot with color mapping if atomic_numbers provided
        if atomic_numbers_valid is not None:
            # Use colormap to color points by atomic number
            scatter = ax.scatter(radius_valid, difference_valid, c=atomic_numbers_valid, 
                            alpha=alpha, s=s, cmap='viridis', label='Prediction Error')
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Atomic Number', fontsize=11)
        else:
            # Single color if no atomic_numbers
            ax.scatter(radius_valid, difference_valid, alpha=alpha, s=s, label='Prediction Error')
        
        # Horizontal line at y=0 (perfect prediction)
        ax.axhline(0, color='r', linestyle='--', linewidth=2, 
                label='Perfect Prediction (y=0)', alpha=0.8)
        
        # Set scales
        ax.set_xscale('log')
        ax.set_yscale('symlog', linthresh=linthresh)
        
        # Add symlog threshold lines (marking the linear region boundaries)
        ax.axhline(linthresh, color='gray', linestyle=':', linewidth=1.5, 
                label=f'Symlog Threshold (+{linthresh})', alpha=0.6)
        ax.axhline(-linthresh, color='gray', linestyle=':', linewidth=1.5, 
                label=f'Symlog Threshold (-{linthresh})', alpha=0.6)
        
        # Labels and title
        ax.set_xlabel('Radius (a.u.)', fontsize=12)
        ax.set_ylabel(f'Prediction Error ({target_name})', fontsize=12)
        title = f'Prediction Error vs Radius'
        if model_name:
            title += f' ({model_name})'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Grid
        ax.grid(True, alpha=0.6, which='both')
        
        # Legend
        ax.legend(fontsize=10)
        
        # Calculate and display statistics on the plot
        mae = np.mean(np.abs(difference_valid))
        rmse = np.sqrt(np.mean(difference_valid**2))
        ax.text(0.05, 0.95, f'MAE = {mae:.6f}\nRMSE = {rmse:.6f}', 
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            plt.show()
        
        return fig, ax


    @staticmethod
    def plot_relative_error_vs_radius(radius, y_true, y_pred, atomic_numbers=None, 
                                    model_name="", target_name="Potential",
                                    save_path=None, figsize=(10, 8), alpha=0.5, s=1
    ):
        """
        Plot relative prediction error vs radius with linear scale.
        
        Parameters
        ----------
        radius : array-like
            Radius values (will be on x-axis)
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values
        atomic_numbers : array-like, optional
            Atomic numbers corresponding to each data point. If provided, points will be colored
            by atomic number with a colorbar showing the mapping.
        model_name : str
            Name of the model (for title)
        target_name : str
            Name of the target variable (for axis labels)
        save_path : str, optional
            Path to save the figure. If None, displays the figure.
        figsize : tuple
            Figure size (width, height)
        alpha : float
            Transparency of scatter points
        s : float
            Size of scatter points
        
        Returns
        -------
        fig, ax : matplotlib figure and axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is required for visualization. Install it with: pip install matplotlib")
            return None, None
        
        radius = np.asarray(radius).flatten()
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        # Calculate relative error: |y_true - y_pred| / |y_true|
        # Avoid division by zero by using a small epsilon
        epsilon = 1e-10
        relative_error = np.abs(y_true - y_pred) / (np.abs(y_true) + epsilon)
        
        # Process atomic_numbers if provided
        atomic_numbers_valid = None
        if atomic_numbers is not None:
            atomic_numbers = np.asarray(atomic_numbers).flatten()
            if len(atomic_numbers) != len(radius):
                print(WARNING_ATOMIC_NUMBERS_MISMATCH_RADIUS)
                atomic_numbers = None
        
        # Filter out invalid values (NaN, inf) and zero true values
        valid_mask = np.isfinite(radius) & np.isfinite(relative_error) & (np.abs(y_true) > epsilon)
        radius_valid = radius[valid_mask]
        relative_error_valid = relative_error[valid_mask]
        if atomic_numbers is not None:
            atomic_numbers_valid = atomic_numbers[valid_mask]
        
        if len(radius_valid) == 0:
            print(WARNING_NO_VALID_DATA)
            return None, None
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Scatter plot with color mapping if atomic_numbers provided
        if atomic_numbers_valid is not None:
            # Use colormap to color points by atomic number
            scatter = ax.scatter(radius_valid, relative_error_valid, c=atomic_numbers_valid, 
                            alpha=alpha, s=s, cmap='viridis', label='Relative Error')
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Atomic Number', fontsize=11)
        else:
            # Single color if no atomic_numbers
            ax.scatter(radius_valid, relative_error_valid, alpha=alpha, s=s, label='Relative Error')
        
        # Horizontal line at y=0 (perfect prediction)
        ax.axhline(0, color='r', linestyle='--', linewidth=2, 
                label='Perfect Prediction (y=0)', alpha=0.8)
        
        # Set scales
        ax.set_xscale('log')
        ax.set_yscale('linear')  # Normal linear scale for relative error
        
        # Set y-axis limits to focus on 0-2 range (clip outliers for better visualization)
        ax.set_ylim(0, 2.0)
        
        # Labels and title
        ax.set_xlabel('Radius (a.u.)', fontsize=12)
        ax.set_ylabel(f'Relative Error ({target_name})', fontsize=12)
        title = f'Relative Prediction Error vs Radius'
        if model_name:
            title += f' ({model_name})'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Grid
        ax.grid(True, alpha=0.6, which='both')
        
        # Legend
        ax.legend(fontsize=10)
        
        # Calculate and display statistics on the plot
        mean_rel_error = np.mean(relative_error_valid)
        max_rel_error = np.max(relative_error_valid)
        median_rel_error = np.median(relative_error_valid)
        ax.text(0.05, 0.95, f'Mean Rel. Error = {mean_rel_error:.6f}\nMax Rel. Error = {max_rel_error:.6f}\nMedian Rel. Error = {median_rel_error:.6f}', 
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            plt.show()
        
        return fig, ax


    @staticmethod
    def plot_atom_delta_xc(
        atomic_number: int,
        radius: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot Delta XC predictions for a single atom.
        Left plot: Delta XC and Predicted Delta XC vs radius (log scale)
        Right plot: Difference vs radius (symlog scale)
        Tick labels show original values, not transformed values.
        
        Parameters
        ----------
        atomic_number : int
            Atomic number
        radius : np.ndarray
            Radius values
        y_true : np.ndarray
            True Delta XC values
        y_pred : np.ndarray
            Predicted Delta XC values
        model_name : str
            Model name for title
        save_path : str, optional
            Path to save the figure
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.ticker as ticker
        except ImportError:
            print("matplotlib is required for visualization. Install it with: pip install matplotlib")
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left plot: Delta XC and Predicted Delta XC vs radius (log scale)
        ax1.plot(radius, y_true, 'b-', linewidth=2, label='True Delta XC', alpha=0.8)
        ax1.plot(radius, y_pred, 'r--', linewidth=2, label='Predicted Delta XC', alpha=0.9)
        ax1.set_xlabel('Radius (a.u.)', fontsize=12)
        ax1.set_ylabel('Delta XC (Ha)', fontsize=12)
        ax1.set_xscale('log')
        ax1.set_yscale('symlog', linthresh=0.01)
        ax1.set_title(f'Atom {atomic_number}: Delta XC vs Radius', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.6, which='both')
        
        # Customize log scale tick labels to show original values
        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:.2e}'))
        
        # Right plot: Difference vs radius (symlog scale)
        difference = y_pred - y_true
        ax2.plot(radius, difference, 'g-', linewidth=2, label='Difference (Pred - True)', alpha=0.8)
        ax2.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Zero')
        ax2.set_xlabel('Radius (a.u.)', fontsize=12)
        ax2.set_ylabel('Difference (Ha)', fontsize=12)
        ax2.set_xscale('log')
        ax2.set_yscale('symlog', linthresh=0.01)
        ax2.set_title(f'Atom {atomic_number}: Prediction Error', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.6, which='both')
        
        # Add symlog threshold lines
        ax2.axhline(0.01, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
        ax2.axhline(-0.01, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
        
        # Customize symlog scale tick labels to show original values
        ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:.2e}'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
        
        return fig, (ax1, ax2)


    @staticmethod
    def _validate_target_by_feature_inputs(
        dataset              : AtomicDataset,
        target_name          : str,
        feature_name         : str,
        target_functional    : Optional[str],
        reference_functional : Optional[str],
    ) -> Tuple[str, str, Optional[str]]:
        valid_targets = {"v_x", "v_c", "v_xc", "e_x", "e_c", "e_xc"}
        target_key = target_name.strip().lower()
        if target_key not in valid_targets:
            raise ValueError(TARGET_NAME_NOT_IN_VALID_LIST_ERROR.format(sorted(valid_targets), target_name))
        
        if not isinstance(feature_name, str):
            raise ValueError(FEATURE_NAME_NOT_STRING_ERROR.format(type(feature_name)))
        if feature_name not in dataset.features_list:
            raise ValueError(FEATURE_NAME_NOT_IN_DATASET_ERROR.format(feature_name, dataset.features_list))
        
        if target_functional is None:
            if hasattr(dataset, "scf_xc_functional") and dataset.scf_xc_functional is not None:
                target_functional = dataset.scf_xc_functional
            else:
                raise ValueError(TARGET_FUNCTIONAL_REQUIRED_ERROR)
        
        if not isinstance(target_functional, str):
            raise ValueError(TARGET_FUNCTIONAL_NOT_STRING_ERROR.format(type(target_functional)))
        
        if hasattr(dataset, "exists_functional"):
            if not dataset.exists_functional(target_functional):
                raise ValueError(
                    TARGET_FUNCTIONAL_NOT_IN_DATASET_ERROR.format(
                        target_functional,
                        dataset.scf_xc_functional,
                        dataset.forward_pass_xc_functional_list
                    )
                )
        else:
            if target_functional != dataset.scf_xc_functional and target_functional not in dataset.forward_pass_xc_functional_list:
                raise ValueError(
                    TARGET_FUNCTIONAL_NOT_IN_DATASET_ERROR.format(
                        target_functional,
                        dataset.scf_xc_functional,
                        dataset.forward_pass_xc_functional_list
                    )
                )
        
        if reference_functional is not None:
            if not isinstance(reference_functional, str):
                raise ValueError(REFERENCE_FUNCTIONAL_NOT_STRING_ERROR.format(type(reference_functional)))
            if hasattr(dataset, "exists_functional"):
                if not dataset.exists_functional(reference_functional):
                    raise ValueError(
                        REFERENCE_FUNCTIONAL_NOT_IN_DATASET_ERROR.format(
                            reference_functional,
                            dataset.scf_xc_functional,
                            dataset.forward_pass_xc_functional_list
                        )
                    )
            else:
                if reference_functional != dataset.scf_xc_functional and reference_functional not in dataset.forward_pass_xc_functional_list:
                    raise ValueError(
                        REFERENCE_FUNCTIONAL_NOT_IN_DATASET_ERROR.format(
                            reference_functional,
                            dataset.scf_xc_functional,
                            dataset.forward_pass_xc_functional_list
                        )
                    )
            if reference_functional == target_functional:
                print(WARNING_REFERENCE_FUNCTIONAL_SAME_AS_TARGET)
        
        return target_key, target_functional, reference_functional



    @staticmethod
    def _format_target_vs_feature_labels(
        feature_label        : str,
        target_text          : str,
        target_key           : str,
        target_functional    : str,
        reference_functional : Optional[str],
    ) -> Tuple[str, str, str]:
        def _escape_functional(name: str) -> str:
            return name.replace("_", r"\_")

        symbol_map = {
            "v_x": r"v_x",
            "v_c": r"v_c",
            "v_xc": r"v_{xc}",
            "e_x": r"e_x",
            "e_c": r"e_c",
            "e_xc": r"e_{xc}",
        }
        symbol = symbol_map.get(target_key, target_key)
        target_tex = _escape_functional(target_functional)

        if reference_functional is None:
            formula = rf"${symbol}^{{\mathrm{{{target_tex}}}}}(r)$"
            title = f"{target_text} {formula} v.s. {feature_label}"
            ylabel = f"{target_text} {formula}"
        else:
            ref_tex = _escape_functional(reference_functional)
            title = f"{target_text} Difference $\\Delta {symbol}(r)$ v.s. {feature_label}"
            formula = (
                rf"{symbol}^{{\mathrm{{{target_tex}}}}}(r) - "
                rf"{symbol}^{{\mathrm{{{ref_tex}}}}}(r)"
            )
            ylabel = f"{target_text} Difference\n$\\Delta {symbol}(r) = {formula}$"

        xlabel = feature_label
        return xlabel, ylabel, title