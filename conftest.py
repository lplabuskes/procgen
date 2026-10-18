"""Test-wide setup: Qt renders offscreen, and the shared fixture modules are in scope for every package's tests."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest_plugins = ["region.test.fixtures", "localmap.test.fixtures", "settlegen.test.fixtures"]
