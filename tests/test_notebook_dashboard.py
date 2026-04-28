"""Regression checks for the interactive ATOM DFT notebook dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ipywidgets = pytest.importorskip("ipywidgets")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from atom.descriptors import MultipoleResult
from atom.ui import AtomDFTDashboard


def _fake_descriptor_result() -> MultipoleResult:
    return MultipoleResult(
        grid_indices=np.array([[0, 0, 0]], dtype=int),
        grid_positions=np.array([[0.0, 0.0, 0.0]]),
        descriptors=np.array([[[1.0]]]),
        rcuts=[0.5],
        l_max=0,
        spacing=(0.5, 0.5, 0.5),
        angular_basis="mcsh",
        radial_basis="heaviside",
        radial_order=0,
        center=(1.0, 1.0, 1.0),
    )


def test_queued_job_snapshots_complete_descriptor_config():
    ui = AtomDFTDashboard()
    ui.include_descriptors.value = True
    ui.descriptor_box.value = 8.0
    ui.descriptor_rcuts.value = (0.5, 1.0)

    job = ui._collect_single_job()
    assert set(job["descriptor_config"]) == {
        "angular_basis",
        "radial_basis",
        "radial_order",
        "rcuts",
        "l_max",
        "box_size",
        "spacing",
        "periodic",
    }

    ui.descriptor_rcuts.value = (2.0, 3.0)
    calculator = ui._solver_kwargs_from_job(job)["descriptor_calculators"][0]
    assert calculator.rcuts == (0.5, 1.0)


def test_batch_updates_last_descriptor_metadata(monkeypatch):
    ui = AtomDFTDashboard()
    ui.include_descriptors.value = True
    ui.descriptor_rcuts.value = (0.5,)
    descriptor_job = ui._collect_single_job()

    ui.include_descriptors.value = False
    plain_job = ui._collect_single_job()
    fake_descriptor = _fake_descriptor_result()

    def fake_run_job(job):
        result = {
            "descriptor_results": {"multipole": fake_descriptor} if job["descriptors"] else {},
            "converged": True,
            "iterations": 1,
            "rho_residual": 0.0,
            "energy": -1.0,
            "wall_time_seconds": 0.0,
        }
        ui._attach_result_metadata(result, job)
        return object(), result, ""

    monkeypatch.setattr(ui, "_run_job", fake_run_job)
    monkeypatch.setattr(ui, "_plot_result", lambda result: None)

    ui.state["queue"] = [descriptor_job, plain_job]
    ui._run_queue()

    assert ui.state["last_result"]["dashboard_metadata"]["descriptors_requested"] is False
    assert ui.state["last_descriptor"] is fake_descriptor
    assert ui.state["last_descriptor_metadata"]["symbol"] == "H"
    assert ui.state["last_descriptor_metadata"]["descriptor_config"]["rcuts"] == (0.5,)


def test_export_uses_descriptor_metadata_not_current_widgets(tmp_path: Path):
    ui = AtomDFTDashboard(project_root=tmp_path)
    ui.state["last_descriptor"] = _fake_descriptor_result()
    ui.state["last_descriptor_metadata"] = {
        "atomic_number": 1,
        "symbol": "H",
        "xc_functional": "GGA_PBE",
        "mode": "AE",
        "descriptor_result_name": "multipole",
        "descriptor_config": {"rcuts": (0.5,)},
    }
    ui.atom_select.value = 6

    ui._export_last_descriptor()

    path = tmp_path / "outputs" / "01_H_GGA_PBE_AE_multipole_descriptor.npz"
    assert path.exists()
    with np.load(path) as data:
        metadata = json.loads(str(data["dashboard_metadata_json"]))
    assert metadata["atomic_number"] == 1
    assert metadata["symbol"] == "H"


def test_psp_unavailable_state_and_batch_restriction(tmp_path: Path):
    psp_dir = tmp_path / "psps"
    psp_dir.mkdir()
    (psp_dir / "01.psp8").write_text("", encoding="utf-8")

    ui = AtomDFTDashboard(project_root=tmp_path)
    ui.atom_select.value = 2
    ui.all_electron.value = False

    assert ui.run_button.disabled is True
    assert "PSP unavailable" in ui.psp_indicator.value
    assert [value for _, value in ui.batch_atoms.options] == [1]
    with pytest.raises(ValueError, match="psps/02.psp8"):
        ui._collect_single_job()


def test_descriptor_rcut_box_warning_blocks_inline_descriptor_run():
    ui = AtomDFTDashboard()
    ui.include_descriptors.value = True
    ui.descriptor_box.value = 8.0
    ui.descriptor_rcuts.value = (5.0,)

    assert "Descriptor warning" in ui.descriptor_warning.value
    assert ui.run_button.disabled is True
    with pytest.raises(ValueError, match="Largest rcut"):
        ui._collect_single_job()


def test_residual_plot_data_uses_outer_loop_history():
    ui = AtomDFTDashboard()
    info = SimpleNamespace(
        inner_iterations=[],
        outer_iterations=[
            SimpleNamespace(outer_iteration=1, outer_rho_residual=1e-2),
            SimpleNamespace(outer_iteration=2, outer_rho_residual=3e-4),
        ],
    )

    plot_data = ui._residual_plot_data({"intermediate_info": info})

    assert plot_data["title"] == "Outer SCF residual"
    assert plot_data["xlabel"] == "Outer iteration"
    assert plot_data["x"] == [1, 2]
    assert plot_data["y"] == [1e-2, 3e-4]
