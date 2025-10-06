from setuptools import setup
from extensions import *
from toml import load

name = load("pyproject.toml")["project"]["name"]
version = load("pyproject.toml")["project"]["version"]

setup(
    name=name,
    version=version,
    setup_requires=["numpy==2.1.3"],
    ext_modules=[OptBuildExtension("gob", version)],
    cmdclass={"build_ext": OptBuild},
    zip_safe=False,
)
