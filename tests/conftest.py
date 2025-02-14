import pytest
import os
import shutil

@pytest.fixture(autouse=True)
def cleanup_after_test(request, tmp_path):
    """Cleanup temporary files after each test."""
    def cleanup():
        shutil.rmtree(tmp_path)
    request.addfinalizer(cleanup)