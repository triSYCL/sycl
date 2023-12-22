# ArchGenMLIR

The triSYCL project also contains part of the ArchGenMLIR
tool. ArchGenMLIR is a tool to automatically generate approximations
for mathematical fixed-point functions to optimize hardware usage for
low precision computations. This was presented at the following conference:

>  Luc FORGET, Gauthier HARNISCH, Ronan KERYELL and Florent DE
>  DINECHIN. « A single-source C++ 20 HLS ﬂow for function evaluation
>  on FPGA and beyond. » *In HEART2022: International Symposium on
>  Highly-Efficient Accelerators and Reconﬁgurable Technologies*, pages
>  51–58. Association for Computing Machinery, Tsukuba, Japan,
>  June 2022. doi:10.1145/3535044. 3535051. https://hal.archives-ouvertes.fr/hal-03684757

It is in 2 parts, the compiler plugin part in this repository and the
library part in the Marto repository.  It also depends on FloPoCo and
Sollya.

Here is how to set it up:
```bash
# install Sollya
sudo apt install libsollya

# install FloPoCo
git clone https://gitlab.com/flopoco/flopoco.git
cd flopoco
mkdir -p build-release
cd build-release
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_BUILD_TYPE="Release"
make -j`nproc`
cd ..
cmake --install build-release --prefix install

FLOPOCO_PATH=`pwd`/install

cd ..

git clone git@github.com:triSYCL/sycl.git
cd sycl
python3 ./buildbot/configure.py \
-o build-release \
--shared-libs \
--cmake-gen Ninja \
-t Release \
--cmake-opt="-DCMAKE_C_COMPILER=/usr/bin/clang" \
--cmake-opt="-DCMAKE_CXX_COMPILER=/usr/bin/clang++" \
--cmake-opt=-DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
--cmake-opt=-DCMAKE_PREFIX_PATH=$FLOPOCO_PATH \
--llvm-external-projects=mlir,compiler-rt

# compiler clang++ and ArchGenMLIR
ninja -C build-release archgen
# run ArchGenMLIR test
ninja -C build-release check-archgen

COMPILER_PATH=`pwd`/build-release

cd ..
# Install the Marto runtime
git clone git@github.com:lforg37/marto.git
cd marto
git checkout leaf_disambiguation
mkdir build-release
cd build-release
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
-DCMAKE_CXX_COMPILER=$COMPILER_PATH/bin/clang++ \
-DCMAKE_BUILD_TYPE="Release" -DBUILD_TESTING=ON \
-DARCHGEN_MLIR_PLUGIN_PATH=$COMPILER_PATH/lib/ArchGenMLIRPlugin.so
cd ..
make -C build-release/ test_expr_mlir
./build-release/archgenlib/examples/test_expr_mlir
```

`test_expr` will test every input of the function and validate the
outputs approximation is within expected range.
