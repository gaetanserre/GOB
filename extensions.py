import os
import platform
import shutil
from pathlib import Path
from subprocess import check_call
import numpy as np
import sys
import urllib.request

sys.dont_write_bytecode = True

from setuptools import Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.build_ext import build_ext
from importlib.machinery import EXTENSION_SUFFIXES


def get_shared_lib_ext():
    if sys.platform.startswith("linux"):
        return ".so"
    elif sys.platform.startswith("darwin"):
        return ".dylib"
    else:
        return ".dll"


def create_directory(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(exist_ok=True, parents=True)
    return path


def mkdir(dir):
    if platform.system() == "Windows":
        return f"mkdir {dir}"
    else:
        return f"mkdir -p {dir}"


class OptBuildExtension(Extension):
    def __init__(self, name: str, version: str):
        super().__init__(name, sources=[])
        # Source dir should be at the root directory
        self.source_dir = Path(__file__).parent.absolute()
        self.version = version


class OptBuild(build_ext):
    def run(self):
        try:
            check_call(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake must be installed")

        if platform.system() not in ("Windows", "Linux", "Darwin"):
            raise RuntimeError(f"Unsupported os: {platform.system()}")

        for ext in self.extensions:
            if isinstance(ext, OptBuildExtension):
                self.build_extension(ext)

    @property
    def config(self):
        return "Debug" if self.debug else "Release"

    def build_cma(self):
        if platform.system() == "Windows":
            return 'cmake -DLIBCMAES_BUILD_EXAMPLES=OFF -G "Visual Studio 17 2022" -A x64 .. && cmake --build . --config Release'
        else:
            return "cmake -DLIBCMAES_BUILD_EXAMPLES=OFF .. && make -j"

    def build_gob(self, lib_name, pkg_name):
        if platform.system() == "Windows":
            return f'cmake -DPython_EXECUTABLE={sys.executable} -DNUMPY_INCLUDE_DIRS={np.get_include()} -DEXT_NAME={lib_name} -DCYTHON_CPP_FILE={pkg_name}.cc .. -G "Visual Studio 17 2022" -A x64 && cmake --build . --config Release'
        else:
            return f"cmake -DPython_EXECUTABLE={sys.executable} -DNUMPY_INCLUDE_DIRS={np.get_include()} -DEXT_NAME={lib_name} -DCYTHON_CPP_FILE={pkg_name}.cc .. && make -j"

    def shared_lib_path(self, lib_name):
        if platform.system() == "Windows":
            return f"Release/{lib_name}{get_shared_lib_ext()}"
        else:
            return f"lib{lib_name}{get_shared_lib_ext()}"

    def build_extension(self, ext: Extension):
        cython_src_dir = Path("gob/optimizers/cpp_optimizers")

        # Copy libcmaes files
        os.system(
            f"cd {cython_src_dir} "
            "&& rm -rf libcmaes "
            "&& git clone https://github.com/CMA-ES/libcmaes "
            "&& cd libcmaes "
            f"&& {mkdir('build')} "
            "&& cd build "
            f"&& {self.build_cma()} "
            "&& cd ../.. "
            "&& cp -r libcmaes/include/libcmaes include "
            "&& cp -r libcmaes/build/include/libcmaes/* include/libcmaes "
            f"&& cd src && {mkdir('libcmaes')} && cd .. "
            "&& cp libcmaes/src/**.cc src/libcmaes "
            "&& rm -rf libcmaes"
        )

        # Copy GLPK files
        urllib.request.urlretrieve(
            "https://mirrors.ocf.berkeley.edu/gnu/glpk/glpk-5.0.tar.gz",
            Path(cython_src_dir, "glpk-5.0.tar.gz"),
        )
        os.system(
            f"cd {cython_src_dir} "
            "&& tar -xvf glpk-5.0.tar.gz "
            "&& rm glpk-5.0.tar.gz"
        )

        ext_dir = Path(self.get_ext_fullpath(ext.name)).parent.absolute()
        create_directory(ext_dir)

        pkg_name = "cpp_optimizers"
        ext_suffix = EXTENSION_SUFFIXES[0]
        lib_name = ".".join((pkg_name + ext_suffix).split(".")[:-1])
        pkg_ext = ".pyd" if platform.system() == "Windows" else ".so"

        # Compile the Cython file
        os.system(
            f"cython --cplus -3 {cython_src_dir}/{pkg_name}.pyx -o {cython_src_dir}/{pkg_name}.cc"
        )

        # Compile the C++ files
        os.system(
            f"cd {cython_src_dir} "
            "&& ls -la .. "
            f"&& rm -rf ../*{pkg_ext} "
            f"&& {mkdir('build')} "
            "&& cd build "
            f"&& {self.build_gob(lib_name, pkg_name)} "
            f"&& mv {self.shared_lib_path(lib_name)} ../../{lib_name}{pkg_ext} "
            f"&& cd {ext.source_dir.as_posix()}"
        )

        # Clean up
        os.system(f"rm -rf {cython_src_dir / 'build'}")

        # Copy files to the build directory
        os.system(f"cp -r gob {ext_dir}")
