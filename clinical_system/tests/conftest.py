"""Test set-up: a throw-away data folder and the tiny dummy model, so tests run in seconds."""
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_tmp = tempfile.mkdtemp(prefix="skinlens_test_")
os.environ.setdefault("APP_DATA_DIR", _tmp)
os.environ.setdefault("MODEL_KIND", "dummy")
os.environ.setdefault("LIME_SAMPLES", "16")
os.environ.setdefault("LIME_SEGMENTS", "20")
os.environ.setdefault("PAPER_FIGURES_DIR", str(Path(_tmp) / "paper_figures"))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

SAMPLE_IMAGE = ROOT.parent / "references" / "test_images" / "sample.jpg"


def csrf_of(html: str) -> str:
    m = re.search(r'name="csrf" value="([^"]+)"', html)
    assert m, "no CSRF token on page"
    return m.group(1)


@pytest.fixture(scope="session")
def client():
    from webapp.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def make_client():
    from webapp.main import app

    def _make():
        return TestClient(app)
    return _make
