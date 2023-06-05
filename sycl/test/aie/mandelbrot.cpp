/// This test requires a display
// REQUIRES: aie && false

// RUN: %aie_clang %s -o %t.bin
// RUN: %if_run_on_device %run_on_device %t.bin > %t.check 2>&1
// RUN: %if_run_on_device FileCheck %s --input-file=%t.check

#include "aie.hpp"
#include "ext/graphics.hpp"
#include <complex>

int main(int argc, char **argv) {
  aie::ext::application<uint8_t> a;
  aie::device<1, 1> dev;
  aie::queue q(dev);
  /// The current maximum size of a memory module is 8192 bytes
  /// sqrt(8192) ~ 90.5... so 90 is the largest integral value we can put here.
  /// at least until the 8192 bytes goes away.
  static auto constexpr image_size = 90;

  a.start(argc, argv, dev.sizeX, dev.sizeY, 1, 1, 100)
      .get_image_grid()
      .get_palette()
      .set(aie::ext::palette::rainbow, 150, 2, 127);
  q.submit(
      [](auto &ht) {
        ht.single_task([](auto &dt) {
          std::uint8_t plane[image_size][image_size];
          // Computation rectangle in the complex plane
          static auto constexpr x0 = -2.1, y0 = -1.2, x1 = 0.6, y1 = 1.2;
          static auto constexpr D = 100; // Divergence norm
          // Size of an image tile
          static auto constexpr xs = (x1 - x0) / dt.size_x() / image_size;
          static auto constexpr ys = (y1 - y0) / dt.size_y() / image_size;

          do {
            for (int i = 0; i < image_size; ++i)
              for (int k, j = 0; j < image_size; ++j) {
                std::complex c{x0 + xs * (dt.size_x() * image_size + i),
                               y0 + ys * (dt.size_y() * image_size + j)};
                std::complex z{0.0};
                for (k = 0; k <= 255; k++) {
                  z = z * z + c;
                  if (norm(z) > D)
                    break;
                }
                plane[j][i] = k;
              }
          } while (!dt.service().update_image(&plane[0][0], 0, 255));
        });
      },
      aie::add_service(a.get_service()));
}
