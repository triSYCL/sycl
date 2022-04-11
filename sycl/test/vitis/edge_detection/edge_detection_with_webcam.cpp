// RUN: true

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

#include "../../../utilities/device_selectors.hpp"
#include <CL/sycl.hpp>

// OpenCV Includes
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace cl::sycl;

class krnl_sobel;

int main(int argc, char *argv[]) {

  constexpr auto height = 480; // input.rows;
  constexpr auto width = 640;  // input.cols;
  constexpr auto area = height * width;

  cv::VideoCapture cap;
  // If opencv don't find a webcam the program exit here
  if (!cap.open(0)) {
    std::cerr << "Unable to connect to the webcam" << std::endl;
    return 1;
  }

  // set the size of picture taken by the webcam
  cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);

  queue q{property::queue::enable_profiling{}};

  // This infinite loop takes a picture from the camera
  // applies the edge detection filter with the FPGA kernel
  // and displays the input and the output in separated windows
  for (;;) {
    cv::Mat inputColor;
    // take the picture from the camera
    cap >> inputColor;

    cv::imshow("inputColor", inputColor);
    cv::Mat inputRaw, input;
    // convert the colored picture into grey
    cv::cvtColor(inputColor, inputRaw, CV_BGR2GRAY);
    inputRaw.convertTo(input, CV_8UC1);

    buffer<uchar> ib{input.begin<uchar>(), input.end<uchar>()};
    buffer<uchar> ob{range<1>{area}};

    // The loop break when pressing the Esc key
    if (cv::waitKey(10) == 27)
      break;

    auto event = q.submit([&](handler &cgh) {
      auto pixel_rb = ib.get_access<access::mode::read>(cgh);
      auto pixel_wb = ob.get_access<access::mode::write>(cgh);

      cgh.single_task<xilinx::reqd_work_group_size<1, 1, 1, krnl_sobel>>([=] {
        xilinx::partition_array<char, 9, xilinx::partition::complete<1>> gX{
            {-1, 0, 1, -2, 0, 2, -1, 0, 1}};
        xilinx::partition_array<char, 9, xilinx::partition::complete<1>> gY{
            {1, 2, 1, 0, 0, 0, -1, -2, -1}};

        for (size_t x = 1; x < width - 1; ++x) {
          for (size_t y = 1; y < height - 1; ++y) {
            int magX = 0, magY = 0;

            xilinx::pipeline([&] {
              for (size_t k = 0; k < 3; ++k) {
                for (size_t l = 0; l < 3; ++l) {
                  int pIndex = (x + k - 1) + (y + l - 1) * width;
                  auto gI = k * 3 + l;
                  magX += gX[gI] * pixel_rb[pIndex];
                  magY += gY[gI] * pixel_rb[pIndex];
                }
              }
            });

            pixel_wb[x + y * width] = cl::sycl::min(
                static_cast<int>(cl::sycl::abs(magX) + cl::sycl::abs(magY)),
                0xFF);
          }
        }
      });
    });

    auto pixel_rb = ob.get_access<access::mode::read>();

    cv::Mat output(height, width, CV_8UC1, pixel_rb.get_pointer());

    cv::imshow("output", output);
    cv::imshow("input", input);
  }

  return 0;
}
