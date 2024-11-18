# Create a setup.py file to install the optimizers module using invoke.

from setuptools import setup
from extensions import *

setup(
    version="0.1.0",
    setup_requires=["numpy==2.1.3"],
    ext_modules=[OptBuildExtension("gob", "0.1.0")],
    cmdclass={"build_ext": OptBuild},
    zip_safe=False,
)
