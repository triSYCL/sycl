/*
  Attempt at translating SDAccel Examples edge_detection example to SYCL

  Intel compile example command (use intel_selector from device_selectors.hpp):
  $ISYCL_BIN_DIR/clang++ -std=c++17 -fsycl edge_detection.cpp -o \
    edge_detection -lsycl -lOpenCL `pkg-config --libs opencv`

  NOTE: POCL won't actually work with this example because it uses std::sqrt
    POCL doesn't have the appropriate SPIR builtin manglings in it's library
    so it doesn't execute correctly (can't find the correct symbols). This is
    a problem on their end rather than ours and I'm not sure its worth making
    a fix to their problem (POCL issue #698).
  POCL compile example, unfortunately there is no 1 instruction compilation for
  POCL at the moment it needs to be 2 stepped as it uses spir-df:
  1) $ISYCL_BIN_DIR/clang++ --sycl -fsycl-use-bitcode -Xclang \
    -fsycl-int-header=edge_detection-int-header.h -c edge_detection.cpp -o \
    kernel.bin
  2) $ISYCL_BIN_DIR/clang++ -std=c++14 -include edge_detection-int-header.h \
      edge_detection.cpp -o edge_detection -lsycl -lOpenCL \
      \ `pkg-config --libs opencv`


  XOCC compile command:
  $ISYCL_BIN_DIR/clang++ -D__SYCL_SPIR_DEVICE__ -DXILINX -std=c++17 -fsycl \
    -fsycl-xocc-device edge_detection.cpp -o edge_detection \
    -lsycl -lOpenCL `pkg-config --libs opencv`

*/

#include <CL/sycl.hpp>
#include <iostream>
#include <iterator>
#include <string>
#include <fstream>
#include <algorithm>
#include <cstdlib>


// OpenCV Includes
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "../../../utilities/device_selectors.hpp"

using namespace cl::sycl;

int main(int argc, char* argv[]) {
  if(argc != 2) {
      std::cout << "Usage: " << argv[0] << "<input> \n";
      return 1;
  }

  // read in image convert to grayscale then convert to unsigned 8 bit values
  cv::Mat inputColor = cv::imread(argv[1]);
  cv::Mat inputRaw, input;
  cv::cvtColor(inputColor, inputRaw, CV_BGR2GRAY);
  inputRaw.convertTo(input, CV_8UC1);

  // using fixed constexpr values stays more true to the original implementation
  // however you can in theory just use input.rows/cols to support a wider range
  // of images sizes. In either case having these outside relies on the SYCL
  // implementation being able to capture these values in the kernel which at
  // the moment it doesn't seem to be able to do.
  #define WIDTH  1024
  #define HEIGHT 1895
  // constexpr auto height = 1895; // input.rows;
  // constexpr auto width = 1024; // input.cols;
  // constexpr auto area = height * width;

#ifdef XILINX
  selector_defines::XOCLDeviceSelector selector;
#else
  selector_defines::IntelDeviceSelector selector;
#endif
  queue q { selector , property::queue::enable_profiling() };

  // may need to modify this to be different if input.isContinuous
  buffer<uchar> ib(input.begin<uchar>(), input.end<uchar>());

  buffer<uchar> ob(range<1>{HEIGHT * WIDTH/*area*/});

  std::cout << "Calculating Max Energy... \n";

  short iMax = 0;
  // Work around for bug in the main SYCL implementation relating to unusable
  // std lib functions when compiling for device (device doesn't care about the
  // host components but still has to compile them)
#ifndef __SYCL_DEVICE_ONLY__
  iMax = *std::max_element(input.begin<uchar>(), input.end<uchar>(),
                            [](auto i, auto j){ return i < j; });
#endif

  std::cout << "inputBits = " << ceil(log2(iMax)) << " coefMax = 2 \n";
  std::cout << "Max Energy = " << ceil(log2((long long)iMax * 2 * 3 * 3)) + 1
            << " Bits \n";
  std::cout << "Image Dimensions: " << input.size() << "\n";

  // mapping the enqueueTask call to a single_task, interested in seeing if a
  // parallel_for without a fixed 1-1-1 mapping is workable on an FPGA though..
  // as its a much cleaner way to express this algorithm. I'm pretty sure sw and
  // hw emulation would work with a parallel_for as is, but how would a real
  // FPGA deal with it?
  std::cout << "Launching Kernel... \n";

  auto event = q.submit([&](handler &cgh) {
    auto pixel_rb = ib.get_access<access::mode::read>(cgh);
    auto pixel_wb = ob.get_access<access::mode::write>(cgh);

    printf("pixel_rb size in submit: %zu \n", pixel_rb.get_size());
    printf("pixel_rb count in submit: %zu \n", pixel_rb.get_count());

    cgh.single_task<class krnl_sobel>(
     [=]() {
#ifdef XILINX
      auto gX = xilinx::partition_array<char, 9,
                xilinx::partition::complete<0>>({-1, 0, 1, -2, 0, 2, -1, 0, 1});

      auto gY = xilinx::partition_array<char, 9,
                xilinx::partition::complete<0>>({1, 2, 1, 0, 0, 0, -1, -2, -1});
#else
      char const gX[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
      char const gY[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
#endif
      int magX, magY, gI, pIndex, sum;


      // Simplified version of krnl_sobelfilter.cl that gives the same output
      // results however the krnl_sobelfilter.cl is more hardware optimized than
      // this so gets better results 25062438 ns vs 17447508 ns
      // the krnl_sobelfilter uses:
      // * xcl_array_partition for sobel kernels
      // * xcl_array_partition for intermediate registers
      // * uses xcl_pipeline_loop to pipeline the for loop
      // * restricts the kernel using reqd_work_group_size
      // * generally pays more attention to bit width when transferring data

      // NOTE: To pipeline the top loops similar to SDAccel's example this has
      // to  be reworked a little as currently the memory consumption of this
      // loop is too large for pipelining and causes an error when compiling
      // for HW. Currently pipelining the inner loop instead.
      for (size_t x = 1; x < WIDTH - 1/*width - 1*/; ++x) {
        for (size_t y = 1; y < HEIGHT - 1/*height - 1*/; ++y) {
            magX = 0; magY = 0;

#ifdef XILINX
            xilinx::pipeline([&] {
#endif
              for(size_t k = 0; k < 3; ++k) {
                for(size_t l = 0; l < 3; ++l) {
                  gI = k * 3 + l;
                  pIndex =  (x + k - 1) + (y + l - 1) * WIDTH;
                  magX += gX[gI] * pixel_rb[pIndex];
                  magY += gY[gI] * pixel_rb[pIndex];
                }
              }
#ifdef XILINX
            });
#endif

            // capping at 0xFF means no blurring of edges when it gets
            // converted back to a char from an int
            sum = std::abs(magX) + std::abs(magY);
            pixel_wb[x + y * WIDTH] = (sum > 0xFF) ? 0xFF : (char)sum;
        }
      }
    });
  });

  // a buffer access or a wait MUST be used before querying the event when
  // using Intel SYCL runtime at the moment as get_profiling_info is not a
  // blocking event in the SYCL specification at the moment(change in progress).
  auto pixel_rb = ob.get_access<access::mode::read>();

  std::cout << "Getting Result... \n";
  auto nstimeend = event.get_profiling_info<info::event_profiling::command_end>();
  auto nstimestart = event.get_profiling_info<info::event_profiling::command_start>();
  auto duration = nstimeend-nstimestart;
  std::cout << "Kernel Duration: " << duration << " ns \n";

  std::cout << "Calculating Output energy.... \n";

  cv::Mat output(HEIGHT/*height*/, WIDTH/*width*/, CV_8UC1, pixel_rb.get_pointer());

  short oMax = 0;
#ifndef __SYCL_DEVICE_ONLY__
  oMax = *std::max_element(output.begin<uchar>(), output.end<uchar>(),
                            [=](auto i, auto j){ return i < j; });
#endif

  std::cout << "outputBits = " << ceil(log2(oMax)) << "\n";

  cv::imwrite("input.bmp", input);
  cv::imwrite("output.bmp", output);

  std::cout << "Completed Successfully \n";

  return 0;
}
