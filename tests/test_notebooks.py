from pathlib import Path

import nbformat
import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    ROOT / "models" / "cloud_training.ipynb",
    ROOT / "models" / "uta_rldd_cloud.ipynb",
)


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda path: path.name)
def test_kaggle_notebooks_are_valid_and_have_stable_cell_ids(notebook_path):
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)

    cell_ids = [cell.id for cell in notebook.cells]
    assert all(cell_ids)
    assert len(cell_ids) == len(set(cell_ids))
