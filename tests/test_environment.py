import os
from importlib.resources import files
from shutil import which

import pytest


@pytest.mark.environment
@pytest.mark.parametrize("command", ["melodic", "fix"])
def test_required_executable_is_on_path(command):
    executable = which(command)

    assert executable is not None, f"Required executable '{command}' is not on PATH"
    assert os.access(executable, os.X_OK), f"'{executable}' is not executable"


@pytest.mark.parametrize("species", ["mouseDSURQE", "mouseAIBS", "ratSIGMA"])
@pytest.mark.parametrize("filename", ["ica.nii.gz", "highres_pveseg.nii.gz"])
def test_species_templates_are_packaged(species, filename):
    resource = files("fixrabies").joinpath("templates", species, filename)

    assert resource.is_file(), f"Missing packaged template: {species}/{filename}"
    with resource.open("rb") as template:
        assert template.read(2) == b"\x1f\x8b", f"Not a gzip file: {species}/{filename}"
