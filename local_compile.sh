# /bin/bash

pkg_name=cpp_optimizers
ext_suffix=$(python3-config --extension-suffix)

lib_name=${pkg_name}${ext_suffix}

cd gob/optimizers/cpp_optimizers

cython --cplus -3 $pkg_name.pyx -o $pkg_name.cc

if ! [ -f "lib/libcmaes.so" ]; then
  rm -rf libcmaes
  git clone https://github.com/CMA-ES/libcmaes.git
  cd libcmaes
  mkdir build
  cd build
  cmake -DCMAKE_INSTALL_PREFIX=../.. ..
  make -j install
  cd ../..
  rm -rf libcmaes
fi

numpy_include=$(python3 -c "import numpy; print(numpy.get_include())")

mkdir -p build
cd build
cmake -DNUMPY_INCLUDE_DIRS=$numpy_include -DEXT_NAME=$lib_name -DCYTHON_CPP_FILE=$pkg_name.cc ..
make -j
mv lib$lib_name.so ../../$lib_name
