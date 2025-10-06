# Quickstart

## Using wheel

The easiest way to install GOB is via pip from PyPI:
```bash
pip install gob
```
Alternatively, you can download the latest wheel from the [releases page](https://github.com/gaetanserre/GOB/releases) on GitHub and install it using pip:

```bash
pip install gob-<version>-<architecture>.whl
```

## Build from source

Make sure you have CMake (≥ 3.28), a c++ compiler, and the eigen3 library installed. Then clone the repository and run:
```bash
pip install . -v
```
It should build the C++ extensions and install the package. You can also build the documentation with:
```bash
./local_compile.sh
cd docs
pip install -r requirements.txt
make html
```

## Contributing

When contributing to this repository, please first discuss the change you wish to make via issue or any other method with the owners of this repository before making a change.

Create a pull request from your forked repository to the `main` branch of the original repository. The pull request will be reviewed and discussed.

### Dependencies

To contribute, you will need to locally compile the package. Make sure you have CMake (≥ 3.28), a c++ compiler, and the eigen3 library installed. Then, you can either install the package on your system:
```bash
pip install . -v
```
or compile it in-place for development:
```bash
./local_compile.sh
```
