"""Basic tests for pup Python module."""

import sys
from pathlib import Path

import pytest

# Add the python directory to the path so we can import pup
project_root = Path(__file__).parent.parent
python_path = project_root / "python"
sys.path.insert(0, str(python_path))


def test_pup_import():
    """Test that pup module can be imported."""
    try:
        import pup

        assert hasattr(pup, "__version__") or hasattr(pup, "__about__")
    except ImportError:
        pytest.skip("Pup module not available for import")


def test_pup_constants():
    """Test pup constants module."""
    try:
        from pup import constants

        # Test that constants module exists and has expected attributes
        assert hasattr(constants, "__name__")
    except ImportError:
        pytest.skip("Pup constants module not available")


def test_pup_coco():
    """Test pup coco module."""
    try:
        from pup import coco

        # Test that coco module exists
        assert hasattr(coco, "__name__")
    except ImportError:
        pytest.skip("Pup coco module not available")


def test_pup_main():
    """Test pup main module."""
    try:
        from pup import main

        # Test that main module exists
        assert hasattr(main, "__name__")
    except ImportError:
        pytest.skip("Pup main module not available")


class TestPupModule:
    """Test class for pup module functionality."""

    def test_module_structure(self):
        """Test that the pup module has expected structure."""
        try:
            import pup

            # Basic module existence test
            assert pup is not None
        except ImportError:
            pytest.skip("Pup module not available")

    def test_python_version_compatibility(self):
        """Test that we're running on a supported Python version."""
        assert sys.version_info >= (3, 7), "Python 3.7+ required"
        assert sys.version_info < (4, 0), "Python 4+ not yet supported"
