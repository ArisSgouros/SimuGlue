# tests/conftest.py
import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--update-gold",
        action="store_true",
        default=False,
        help="Overwrite gold/reference files with new outputs instead of comparing."
    )

@pytest.fixture
def update_gold(request):
    """Fixture to check if --update-gold was given."""
    return request.config.getoption("--update-gold")

@pytest.fixture
def tmp_path_cwd(tmp_path, monkeypatch):
  monkeypatch.chdir(tmp_path)
  return tmp_path
