#!/usr/bin/env python
"""
setup.py for PyART (Python Analytical Relativity Toolkit)

Static metadata lives in pyproject.toml; this file only exists to print a
post-install reminder, which pyproject.toml cannot express declaratively.
"""

from setuptools import setup
from setuptools.command.install import install
from setuptools.command.develop import develop

PATCH_MESSAGE = """
============================================================
  PyART post-install step required
------------------------------------------------------------
  pycbc <= 2.10.0 is incompatible with numpy 2.x.
  Run the following command once to patch pycbc in-place:

      python scripts/patch_pycbc_numpy2.py

  The patch is idempotent and safe to re-run after pycbc
  upgrades.
============================================================
"""


def print_patch_notice(command_cls):
    class Patched(command_cls):
        def run(self):
            super().run()
            print(PATCH_MESSAGE)

    return Patched


setup(
    cmdclass={
        "install": print_patch_notice(install),
        "develop": print_patch_notice(develop),
    },
)
