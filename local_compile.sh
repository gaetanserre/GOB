# /bin/bash

pkg_name=cpp_optimizers
ext_suffix=$(python -c "from importlib.machinery import EXTENSION_SUFFIXES; print(EXTENSION_SUFFIXES[0])")

lib_name=${pkg_name}${ext_suffix}
shared_library_header=lib

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  shared_library_ext=.so
elif [[ "$OSTYPE" == "darwin"* ]]; then
  shared_library_ext=.dylib
else
  shared_library_header=Release/
  shared_library_ext=.dll
fi

build_cmaes() {
  if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
    cmake -DLIBCMAES_BUILD_EXAMPLES=OFF ..
    make -j
  else
    cmake -DLIBCMAES_BUILD_EXAMPLES=OFF -G "Visual Studio 17 2022" -A x64 ..
    cmake --build . --config Release
  fi
}

build_gob() {
  if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
    cmake -DNUMPY_INCLUDE_DIRS=$numpy_include -DEXT_NAME=$lib_name -DCYTHON_CPP_FILE=$pkg_name.cc ..
    make -j
  else
    cmake -DNUMPY_INCLUDE_DIRS=$numpy_include -DEXT_NAME=$lib_name -DCYTHON_CPP_FILE=$pkg_name.cc -G "Visual Studio 17 2022" -A x64 ..
    cmake --build . --config Release
  fi
}

cd gob/optimizers/cpp_optimizers

cython --cplus -3 $pkg_name.pyx -o $pkg_name.cc


if [ ! -f include/libcmaes/cmaes_export.h ]; then
  git clone https://github.com/gaetanserre/libcmaes
  cd libcmaes
  mkdir build
  cd build
  build_cmaes
  cd ../..
  cp -r libcmaes/include/libcmaes include
  cp -r libcmaes/build/include/libcmaes/* include/libcmaes
  mkdir -p src/libcmaes
  cp -r libcmaes/src/*.cc src/libcmaes
  rm -rf libcmaes
fi

if [ ! -f include/glpk/glpk.h ]; then
curl http://ftp.gnu.org/gnu/glpk/glpk-5.0.tar.gz -o glpk-5.0.tar.gz
tar -xzf glpk-5.0.tar.gz
mkdir -p src/glpk
mkdir -p include/glpk
cp -r glpk-5.0/src/**/*.c src/glpk
cp -r glpk-5.0/src/**/*.h include/glpk
cp -r glpk-5.0/src/*.h include/glpk
rm -rf glpk-5.0 glpk-5.0.tar.gz src/glpk/main.c
fi

numpy_include=$(python -c "import numpy; print(numpy.get_include())")

mkdir -p build
cd build
build_gob
mv $shared_library_header$lib_name$shared_library_ext ../../$lib_name
