// REQUIRES: vitis && opencv4

// RUN: %clangxx %EXTRA_COMPILE_FLAGS-std=c++20 -fsycl -fsycl-targets=%sycl_triple -o %t.out %s %opencv4_flags
// RUN: %run_if_hw %ACC_RUN_PLACEHOLDER %t.out %S/data/input/eiffel.bmp
// RUN: %run_if_hw %ACC_RUN_PLACEHOLDER %t.out %S/data/input/lola.bmp
// RUN: %run_if_hw %ACC_RUN_PLACEHOLDER %t.out %S/data/input/vase.bmp
// RUN: %run_if_sw_emu %ACC_RUN_PLACEHOLDER %t.out %S/data/input/eiffel.bmp
// RUN: %run_if_sw_emu %ACC_RUN_PLACEHOLDER %t.out %S/data/input/lola.bmp
// RUN: %run_if_sw_emu %ACC_RUN_PLACEHOLDER %t.out %S/data/input/vase.bmp

// ./build-release/bin/clang++ -g -std=c++20 -fsycl -fsycl-targets=fpga64_hls_hw sycl/test/vitis/edge_detection/edge_detection.cpp -o edge_detection `pkg-config --libs --cflags opencv4`

/*
  First attempt at translating SDAccel Examples edge_detection example to SYCL
  https://github.com/Xilinx/SDAccel_Examples/tree/master/vision/edge_detection
  For now there is no implementation of the original code using internal line
  buffers.

  Compile command for AMD FPGA:
  $SYCL_BIN_DIR/clang++ -std=c++20 -fsycl \
    -fsycl-targets=fpga64_hls_hw edge_detection.cpp \
    -o edge_detection `pkg-config --libs --cflags opencv4`

  Compile command for default selector and target (typically OpenCL CPU
  implementation):
  $SYCL_BIN_DIR/clang++ -std=c++20 -fsycl edge_detection.cpp \
    -o edge_detection `pkg-config --libs --cflags opencv4`
*/

#include <sycl/sycl.hpp>
#include <sycl/ext/xilinx/fpga.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>

// OpenCV Includes
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char* argv[]) {
  if (argc != 2) {
      std::cout << "Usage: " << argv[0] << " <input>\n";
      return 1;
  }

  // read in image convert to grayscale then convert to unsigned 8 bit values
  cv::Mat inputColor = cv::imread(argv[1]);
  cv::Mat inputRaw, input;
  cv::cvtColor(inputColor, inputRaw, cv::COLOR_BGR2GRAY);
  inputRaw.convertTo(input, CV_8UC1);

  // using fixed constexpr values stays more true to the original implementation
  // however you can in theory just use input.rows/cols to support a wider range
  // of image sizes.
  const size_t height = input.rows;
  const size_t width = input.cols;
  const size_t area = height * width;

  sycl::queue q {sycl::property::queue::enable_profiling()};

  // may need to modify this to be different if input.isContinuous
  sycl::buffer ib {input.begin<uchar>(), input.end<uchar>()};
  sycl::buffer<uchar> ob { area };

  std::cout << "Calculating Max Energy... \n";

  auto iMax = *std::max_element(input.begin<uchar>(), input.end<uchar>(),
                                [](auto i, auto j) { return i < j; });

  std::cout << "inputBits = " << std::ceil(std::log2(iMax)) << " coefMax = 2 \n";
  std::cout << "Max Energy = " << std::ceil(std::log2((long long)iMax * 2 * 3 * 3)) + 1
            << " Bits \n";
  std::cout << "Image Dimensions: " << input.size() << std::endl;
  std::cout << "Used Size: " << height << "x" << width << " = " << area << std::endl;

  // mapping the enqueueTask call to a single_task, interested in seeing if a
  // parallel_for without a fixed 1-1-1 mapping is workable on an FPGA though..
  // as its a much cleaner way to express this algorithm. I'm pretty sure SW and
  // HW emulation would work with a parallel_for as is, but how would a real
  // FPGA deal with it?
  std::cout << "Launching Kernel... \n";

  auto event = q.submit([&](sycl::handler &cgh) {
    sycl::accessor pixel_rb{ib, cgh, sycl::read_only};
    sycl::accessor pixel_wb{ob, cgh, sycl::write_only};

    std::cout << "pixel_rb size in submit: " << pixel_rb.get_size() << std::endl;
    std::cout << "pixel_rb count in submit: " << pixel_rb.size() << std::endl;

    cgh.single_task<class krnl_sobel>(
    // The reqd_work_group_size is already actually applied internally for single
    // tasks but this showcases it's usage none the less, as it can be applied
    // to parallel_fors with local sizes
     [=] {
      // Partition completely the following arrays along their first dimension
      auto gX = sycl::ext::xilinx::partition_array<
          char, 9, sycl::ext::xilinx::partition::complete<1>>(
          {-1, 0, 1, -2, 0, 2, -1, 0, 1});

      auto gY = sycl::ext::xilinx::partition_array<
          char, 9, sycl::ext::xilinx::partition::complete<1>>(
          {1, 2, 1, 0, 0, 0, -1, -2, -1});

      int magX, magY, gI, pIndex;

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
      // to be reworked a little as currently the memory consumption of this
      // loop is too large for pipelining and causes an error when compiling
      // for HW. Currently pipelining the inner loop instead.
      for (size_t x = 1; x < width - 1; ++x) {
        for (size_t y = 1; y < height - 1; ++y) {
          magX = 0; magY = 0;

          sycl::ext::xilinx::pipeline([&] {
            for(size_t k = 0; k < 3; ++k) {
              for(size_t l = 0; l < 3; ++l) {
                gI = k * 3 + l;
                pIndex =  (x + k - 1) + (y + l - 1) * width;
                magX += gX[gI] * pixel_rb[pIndex];
                magY += gY[gI] * pixel_rb[pIndex];
              }
            }
          });

          // capping at 0xFF means no blurring of edges when it gets
          // converted back to a char from an int
          pixel_wb[x + y * width] = sycl::min((int)(sycl::abs(magX)
                                                  + sycl::abs(magY)), 0xFF);
        }
      }
    });
  });

  // a buffer access or a wait MUST be used before querying the event when
  // using Intel SYCL runtime at the moment as get_profiling_info is not a
  // blocking event in the SYCL specification at the moment(change in progress).
  sycl::host_accessor pixel_rb{ob, sycl::read_only};

  std::cout << "Getting Result... \n";
  auto nstimeend =
      event.get_profiling_info<sycl::info::event_profiling::command_end>();
  auto nstimestart =
      event.get_profiling_info<sycl::info::event_profiling::command_start>();
  auto duration = nstimeend - nstimestart;
  std::cout << "Kernel Duration: " << duration << " ns \n";

  std::cout << "Calculating Output energy.... \n";

  cv::Mat output(height, width, CV_8UC1, (void*)pixel_rb.get_pointer());

  auto oMax = *std::max_element(output.begin<uchar>(), output.end<uchar>(),
                                [=](auto i, auto j) { return i < j; });

  std::cout << "outputBits = " << std::ceil(std::log2(oMax)) << std::endl;

  cv::imwrite("input.bmp", input);
  cv::imwrite("output.bmp", output);

  std::cout << "Completed Successfully \n";

  return 0;
}
