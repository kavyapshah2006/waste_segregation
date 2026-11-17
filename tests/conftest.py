import shutil
from pathlib import Path

import pytest

from kb.build_kb import KB_DIR, build
from pipeline.kb import KnowledgeBase


@pytest.fixture(scope="session")
def kb_path(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("kb") / "kb.sqlite"
    build(out=out)
    return out


@pytest.fixture(scope="session")
def kb(kb_path) -> KnowledgeBase:
    return KnowledgeBase(kb_path)


@pytest.fixture
def kb_source_copy(tmp_path) -> Path:
    """A writable copy of the KB sources, for tests that add cities or break items."""
    dest = tmp_path / "kb"
    shutil.copytree(KB_DIR, dest, ignore=shutil.ignore_patterns("build", "__pycache__"))
    return dest
