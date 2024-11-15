import invoke
import os
import numpy as np

# Get the include path for the python3 interpreter
python_include = os.popen("python -m pybind11 --includes").read().strip()

# check if macos
lib_name = "liboptimizers.so"
linker_flag = os.popen("python3-config --ldflags").read().strip()
if os.uname().sysname == "Darwin":
    lib_name = "liboptimizers.dylib"
    python_version = ".".join(
        os.popen("python --version").read().strip().split(" ")[-1].split(".")[:-1]
    )
    linker_flag += f" -shared -lpython{python_version}"


def print_banner(msg):
    print("==================================================")
    print("= {} ".format(msg))


@invoke.task()
def build_cmaes_lib(c):
    print_banner("Building CMA-ES Library")
    os.system(
        "rm -rf libcmaes "
        "&& git clone https://github.com/CMA-ES/libcmaes.git "
        "&& cd libcmaes "
        "&& mkdir -p build "
        "&& cd build "
        "&& cmake -DCMAKE_INSTALL_PREFIX=../.. .. "
        "&& make -j install"
    )
    print("* Complete")


@invoke.task(build_cmaes_lib)
def build_optimizers_lib(c):
    print_banner("Building C++ Library")
    os.system(
        f"mkdir -p build "
        "&& cd build "
        f"&& cmake -DNUMPY_INCLUDE_DIRS={np.get_include()} .. "
        f"&& make -j && cp {lib_name} .."
    )
    print("* Complete")


def compile_python_module(cpp_name, extension_name):
    invoke.run(
        "g++ -O3 -Wall -shared -std=c++20 "
        f"-fPIC {python_include} "
        f"{cpp_name} "
        f"-o {extension_name}`python3-config --extension-suffix` "
        f"{linker_flag} "
        "-L. -Llib/ -loptimizers -lcmaes -Wl,-rpath,."
    )


@invoke.task(build_optimizers_lib)
def build_optimizers(c):
    print_banner("Building optimizers Module")
    invoke.run("cython --cplus -3 optimizers.pyx -o optimizers.cc")
    compile_python_module("optimizers.cc", "optimizers")
    print("* Complete")


""" @invoke.task(build_optimizers)
def test_optimizers(c):
    print_banner("Testing pyGKLS Module")
    invoke.run("python tests.py", pty=True) """
