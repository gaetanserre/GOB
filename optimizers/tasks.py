import invoke
import os
import eigency

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
def build_optimizers_lib(c):
    print_banner("Building C++ Library")
    os.system(f"mkdir -p build && cd build && cmake .. && make -j && cp {lib_name} ..")
    print("* Complete")


def compile_python_module(cpp_name, extension_name):

    eigency_include = eigency.get_includes()
    eigency_include.reverse()
    eigency_include = " -I".join(eigency_include)
    invoke.run(
        "g++ -O3 -Wall -shared -std=c++20 "
        f"-I{eigency_include} "
        f"-fPIC {python_include} "
        f"{cpp_name} "
        f"-o {extension_name}`python3-config --extension-suffix` "
        f"{linker_flag} "
        "-L. -loptimizers -Wl,-rpath,."
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
