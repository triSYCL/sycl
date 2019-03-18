/*
  $ISYCL_BIN_DIR/clang++ -std=c++14 -fsycl median_filter.cpp -o \
    median_filter -lsycl -lOpenCL `pkg-config --libs opencv`
*/

#include <CL/sycl.hpp>
#include <iostream>

#include <opencv2/opencv.hpp>

#include "../../../utilities/device_selectors.hpp"

using namespace cl::sycl;

// very wip implementation of the median_filter SDAccel example
int main(int argc, char* argv[]) {
  if (argc < 2 || argc > 3 )
  {
    printf("Usage: %s <input bitmap> <golden bitmap(optional)>\n", argv[0]);
    return -1;
  }

  cv::Mat input = cv::imread(argv[1]);
  int height = input.rows;
  int width = input.cols;

  if (!input.data) return 1;

  std::cout << "Setting up SYCL hardware and software...\n";

  buffer<int> ib(input.begin<int>(), input.end<int>());
  buffer<int> ob(height * width);

  // Temporary work around for lambda variables not being captured and
  // transferred to the kernel implicitly
  int wh[2] = {width, height};
  buffer<int> whb(wh, range<1>{2});

  selector_defines::XOCLDeviceSelector xocl;
  selector_defines::IntelDeviceSelector iocl;

  queue q { iocl , property::queue::enable_profiling() }; // should default to IOCL

  std::cout << "Found Device="
            << q.get_device().get_info<info::device::name>() << "\n";

  std::cout << "Launching Kernel... \n";
  // this is perhaps better expressed through a parallel_for_workgroup due to
  // the barrier requirement, but that's not quite implemented yet in the SYCL
  // runtime
  auto event = q.submit([&](handler &cgh) {
    auto wh_rb = whb.get_access<access::mode::read>(cgh);
    auto pixel_rb = ib.get_access<access::mode::read>(cgh);
    auto pixel_wb = ob.get_access<access::mode::write>(cgh);

    cgh.single_task<class median_filter>([=]() {
      printf("single_task 1 launched");
      printf("%d \n", wh_rb[0]);
      printf("%d \n", wh_rb[1]);
    });
  });


  // .get_access is a blocking event
  auto rb = ob.get_access<access::mode::read>();

  bool match = true;
  if (argc == 3) {
    cv::Mat goldenImage = cv::imread(argv[3]);
    if (!goldenImage.data) {
      std::cout << "ERROR:Unable to Read Golden Bitmap File "<< argv[3] << "\n";
      return 1;
    }

    if (goldenImage.rows != input.rows ||
        goldenImage.cols != input.cols) {
          match = false;
    } else {
      for (int i = 0; i < height*width; ++i) {
        if (goldenImage.data[i] != rb[i]) {
          match = false;
          printf("Pixel %d Mismatch Output %x and Expected %x \n", i, rb[i],
                   goldenImage.data[i]);
          break;
        }
      }
    }
  }

  printf("Writing RAW Image \n");
  cv::imwrite("output.bmp", cv::Mat(height, width, CV_8UC3, rb.get_pointer()));

  std::cout << (match ? "TEST PASSED" : "TEST FAILED") << std::endl;
  return match ? 0 : 1;
}
