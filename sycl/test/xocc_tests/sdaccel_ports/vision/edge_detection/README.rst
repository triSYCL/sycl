Edge detection with SYCL
========================

Edge detection
--------------

These files implements an edge detection program based on sobel filter.
The program takes in input a picture and transform it to have the edges
in white and the rest in black. To do so it applies the sobel filter on each
pixel of the input picture to compute the value of the correponding pixel of the
output picture.

I/O type
--------

You can find here two versions of the programs. The first one,
``edge_detection.cpp``, takes and produces ``.bmp`` files, the second one,
``edge_detection_with_webcam.cpp`` take the input of a webcam and display
the output on the screen in live. In both programs input and output are
handled using the opencl library.

Compilation
-----------

To compile this you need to setup your environment like described in the
file `GettingStartedAlveo <https://github.com/triSYCL/sycl/blob/sycl/unified/master/sycl/doc/GettingStartedAlveo.md>`_.
Once this is done you can compile using the following command :

``$SYCL_BIN_DIR/clang++ -std=c++2a -fsycl
-fsycl-targets=fpga64-xilinx-unknown-sycldevice
edge_detection.cpp -o a.out -lOpenCL -I/opt/xilinx/xrt/include
`pkg-config --libs opencv```
