# Create a setup.py file to install the optimizers module using invoke.

from setuptools import setup
from extensions import *
from toml import load

version = load("pyproject.toml")["project"]["version"]

setup(
    version=version,
    setup_requires=["numpy==2.1.3"],
    ext_modules=[OptBuildExtension("gob", version)],
    cmdclass={"build_ext": OptBuild},
    zip_safe=False,
)
