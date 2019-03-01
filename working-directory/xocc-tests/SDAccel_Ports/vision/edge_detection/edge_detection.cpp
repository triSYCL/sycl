/*
  Attempt at translating SDAccel Examples edge_detection example to SYCL

  Intel compile example command (use intel_selector from device_selectors.hpp):
  $ISYCL_BIN_DIR/clang++ -std=c++14 -fsycl edge_detection.cpp -o \
    edge_detection -lsycl -lOpenCL `pkg-config --libs opencv`

  POCL compile example, unfortunately there is no 1 instruction compilation for
  POCL at the moment it needs to be 2 stepped as it uses spir-df:
  1) $ISYCL_BIN_DIR/clang++ --sycl -fsycl-use-bitcode -Xclang \
    -fsycl-int-header=edge_detection-int-header.h -c edge_detection.cpp -o \
    kernel.bin
  2) $ISYCL_BIN_DIR/clang++ -std=c++14 -include edge_detection-int-header.h \
      edge_detection.cpp -o edge_detection -lsycl -lOpenCL \
      \ `pkg-config --libs opencv`

*/

// TODO: Double check there is an issue with event profiling between pocl and iocl
//       perhaps check what XRT does. If there is, then dig into where this is a
//       problem.

#include <CL/sycl.hpp>

#include <iostream>
#include <iterator>
#include <string>
#include <fstream>
#include <algorithm>
#include <cstdlib>


// OpenCV Includes - Could probably swap these out if they're only required for
// outputting an image
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "../../../device_selectors.hpp"

using namespace cl::sycl;

// This or a variant of it should be added to our variation of the SYCL runtime
// to prevent redefining it in all examples
// class XOCLDeviceSelector : public device_selector {
//  public:
//    int operator()(const device &Device) const override {
//      const std::string DeviceVendor = Device.get_info<info::device::vendor>();
//      return (DeviceVendor.find("Xilinx") != std::string::npos) ? 1 : -1;
//    }
//  };

 // Using this until the SYCL implementation works out a way to deal with the
 // std library when compiling for device (issue #15)
 short getMax(cv::Mat mat) {
 	short max = 0;

 	size_t rows = mat.rows;
 	size_t cols = mat.cols;

 	for(size_t r = 0; r < rows; r++) {
 		for(size_t c = 0; c < cols; c++) {
 			uchar tmp = mat.at<uchar>(r,c);
 			if(tmp > max) {
 				max = tmp;
 			}
 		}
 	}

 	return max;
 }

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

  // could make this and the calculations based on it all fixed constexpr values
  // if we want to stay true to the usage of fixed #defines from the example
  auto height = input.rows; // old val 1895
  auto width = input.cols;  // old val 1024
  auto area = height * width;

  selector_defines::XOCLDeviceSelector xocl;
  selector_defines::POCLDeviceSelector pocl;
  selector_defines::IntelDeviceSelector iocl;

  // queue q { xocl };
  queue q { iocl , property::queue::enable_profiling() }; // should default to IOCL

  // may need to modify this to be different if input.isContinuous
  buffer<uchar> ib(input.begin<uchar>(), input.end<uchar>());
  buffer<uchar> ob(range<1>(height*width));

  std::cout << "Calculating Max Energy... \n";
  // auto iMax = *std::max_element(input.begin<uchar>(), input.end<uchar>(),
  //                                   [](auto i, auto j){ return i < j; });
  short iMax = getMax(input);
  std::cout << "inputBits = " << ceil(log2(iMax)) << " coefMax = 2 \n";
  std::cout << "Max Energy = " << ceil(log2((long long)iMax * 2 * 3 * 3)) + 1
            << " Bits \n";

  printf("%d \n", width);


  // mapping the enqueueTask call to a single_task, interested in seeing if a
  // parallel_for without a fixed 1-1-1 mapping is workable on an FPGA though..
  // as its a much cleaner way to express this algorithm. I'm pretty sure sw and
  // hw emulation would work with a parallel_for as is, but how would a real
  // FPGA deal with it?
  std::cout << "Launching Kernel..." << std::endl;
  auto event = q.submit([&](handler &cgh) {
    auto pixel_rb = ib.get_access<access::mode::read>(cgh);
    auto pixel_wb = ob.get_access<access::mode::write>(cgh);
    // printf("submitting kernel \n");
    //
    int w = 512;

    // A typical FPGA-style pipelined kernel loosely based on
    // krnl_sobelfilter.cl from the SDAccel example set
    // TODO: -reqd_work_group_size should be applied to the kernel perhaps
    cgh.single_task<class krnl_sobel>([=]() {
      printf("in single_task \n");
      // // TODO: Port xcl_array_partition opt from triSYCL for these in the
      // // future
      // char const gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
      // char const gy[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};
      // int magX, magY;
      //
      printf("%d \n", w);
      // for (int x = 0; x < width; ++x) {
      //   for (int y = 0; y < height; ++y) {
      //     magX = 0; magY = 0;
      //     // printf("%d \n", x + y * width);
      //     // printf("%u \n", pixel_rb[x + y * width]);
      //     for (int i = 0; i < 3; ++i) {
      //       for (int j = 0; j < 3; ++j) {
      //         int index = (x + i - 1) + (y + j - 1) * width;
      //         // magX += pixel_rb[index] * gx[i][j];
      //         // magY += pixel_rb[index] * gy[i][j];
      //       }
      //     }
      //
      //     // pixel_wb[x + y * width] = (uchar)cl::sycl::sqrt((float)magX);
      //     // pixel_wb[x + y * width] = 10; //(int)cl::sycl::sqrt(10.0f);
      //     // cl::sycl::sqrt(10.0f)
      //   }
      // }

    });
  });

  // a buffer access or a wait MUST be used before querying the event when
  // using intel cl as it's these block, get_profiling_info is not a blocking
  // event in the sycl specication at the moment. Is querying an event in OpenCL?
  q.wait();

  std::cout << "Getting Result... \n";
  // may not work just yet, as it doesn't seem to work on the host.
  // if not can show off CL interop for the time being using event.get()
  auto nstimeend = event.get_profiling_info<info::event_profiling::command_end>();
  auto nstimestart = event.get_profiling_info<info::event_profiling::command_start>();
  auto duration = nstimeend-nstimestart;
  std::cout << "Kernel Duration: " << duration << " ns \n";

  auto pixel_rb = ob.get_access<access::mode::read>();

  std::cout << "Calculating Output energy.... \n";

  cv::Mat output(height, width, CV_8UC1, pixel_rb.get_pointer());

  // auto oMax = *std::max_element(output.begin<uchar>(), output.end<uchar>(),
  //                               [=](auto i, auto j){ return i < j; });
  auto oMax = getMax(output);

  std::cout << "outputBits = " << ceil(log2(oMax)) << "\n";

  cv::imwrite("input.bmp", input);
  cv::imwrite("output.bmp", output);

  std::cout << "Completed Successfully \n";

  return 0;
}
