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

    def build_extension(self, ext: Extension):
        cython_src_dir = Path("gob/optimizers/cpp_optimizers")

        # Copy libcmaes files
        os.system(
            f"cd {cython_src_dir} "
            "&& rm -rf libcmaes "
            "&& git clone https://github.com/CMA-ES/libcmaes.git "
            "&& cd libcmaes "
            "&& mkdir -p build "
            "&& cd build "
            "&& cmake -DLIBCMAES_BUILD_EXAMPLES=OFF .. "
            "&& make -j "
            "&& cd ../.. "
            "&& cp -r libcmaes/include/libcmaes include "
            "&& cp -r libcmaes/build/include/libcmaes/* include/libcmaes "
            "&& mkdir -p src/libcmaes "
            "&& cp libcmaes/src/**.cc src/libcmaes "
            "&& rm -rf libcmaes"
        )

        # Copy GLPK files
        urllib.request.urlretrieve(
            "http://ftp.gnu.org/gnu/glpk/glpk-5.0.tar.gz",
            Path(cython_src_dir, "glpk-5.0.tar.gz"),
        )

        os.system(
            f"cd {cython_src_dir} "
            "&& tar -xvf glpk-5.0.tar.gz "
            "&& mkdir -p src/glpk "
            "&& mkdir -p include/glpk "
            "&& cp -r glpk-5.0/src/**/*.c src/glpk "
            "&& cp -r glpk-5.0/src/**/*.h include/glpk "
            "&& cp -r glpk-5.0/src/*.h include/glpk "
            "&& rm -rf glpk-5.0 glpk-5.0.tar.gz src/glpk/main.c"
        )

        ext_dir = Path(self.get_ext_fullpath(ext.name)).parent.absolute()
        create_directory(ext_dir)

        pkg_name = "cpp_optimizers"
        ext_suffix = os.popen("python3-config --extension-suffix").read().strip()
        lib_name = ".".join((pkg_name + ext_suffix).split(".")[:-1])

        # Compile the Cython file
        os.system(
            f"cython --cplus -3 {cython_src_dir}/{pkg_name}.pyx -o {cython_src_dir}/{pkg_name}.cc"
        )

        # Compile the C++ files
        os.system(
            f"cd {cython_src_dir} "
            "&& mkdir -p build "
            "&& cd build "
            f"&& cmake -DNUMPY_INCLUDE_DIRS={np.get_include()} -DEXT_NAME={lib_name} -DCYTHON_CPP_FILE={pkg_name}.cc .. "
            "&& make -j "
            f"&& mv lib{lib_name}{get_shared_lib_ext()} ../../{lib_name}.so "
            f"&& cd {ext.source_dir.as_posix()}"
        )

        # Clean up
        os.system(f"rm -rf {cython_src_dir / 'build'}")

        # Copy files to the build directory
        os.system(f"cp -r gob {ext_dir}")
