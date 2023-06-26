/// This test requires a display
// REQUIRES: aie && false

// RUN: %aie_clang %s -o %t.bin
// RUN: %if_run_on_device %run_on_device %t.bin > %t.check 2>&1
// RUN: %if_run_on_device FileCheck %s --input-file=%t.check

#include "aie.hpp"
#include "ext/graphics.hpp"
#include <complex>

using data_type = double;

auto constexpr K = 1.0 / 300.0;
auto constexpr g = 9.81;
auto constexpr alpha = K * g;
auto constexpr damping = 0.999;

auto constexpr image_size = 4;
auto constexpr zoom = 5;
/// Add a drop almost between tile (1,1) and (2,2)
auto constexpr x_drop = image_size * 1 - image_size / 2 - 1;
auto constexpr y_drop = image_size * 1 - image_size / 2 - 1;
auto constexpr drop_value = 100.0;
auto constexpr drop_radius = 5.0;

/** Time-step interval between each display.
    Use 1 to display all the frames, 2 for half the frame and so on. */
auto constexpr display_time_step = 1;

auto epsilon = 0.01;

/// Compute the square power of a value
constexpr auto square = [](auto v) constexpr { return v * v; };

/// Compute the contribution of a drop to the water height
constexpr auto add_a_drop = [](auto x, auto y) constexpr {
  // The square radius to the drop center
  auto r = square(x - x_drop) + square(y - y_drop);
  // A cone of height drop_value centered on the drop center
  return r < square(drop_radius)
             ? drop_value * (square(drop_radius) - r) / square(drop_radius)
             : 0;
};

/// Add a circular shoal in the water with half the depth
constexpr auto shoal_factor = [](auto x, auto y) constexpr {
  /// The shoal center coordinates
  auto constexpr x_shoal = image_size * 8 - 3;
  auto constexpr y_shoal = image_size * 4;
  auto constexpr shoal_radius = 200.0;

  // The square radius to the shoal center
  auto r = square(x - x_shoal) + square(y - y_shoal);
  // A disk centered on the shoal center
  return r < square(shoal_radius) ? 0.5 : 1;
};

/// Add a square harbor in the water
constexpr auto is_harbor = [](auto x, auto y) constexpr -> bool {
  /// The square harbor center coordinates
  auto constexpr x_harbor = image_size * 3 - image_size / 3;
  auto constexpr y_harbor = image_size * 2 - image_size / 3;
  auto constexpr length_harbor = image_size;

  // A square centered on the harbor center
  auto harbor =
      x_harbor - length_harbor / 2 <= x && x <= x_harbor + length_harbor / 2 &&
      y_harbor - length_harbor / 2 <= y && y <= y_harbor + length_harbor / 2;
  // Add also a breakwater below
  auto constexpr width_breakwater = image_size / 3;
  auto breakwater = x_harbor <= x && x <= x_harbor + width_breakwater &&
                    y < y_harbor - image_size
                    // Add some 4-pixel holes every image_size/2
                    && (y / 4) % (image_size / 8);
  return false;
  // return harbor || breakwater;
};

void initialize_space(auto& dt) {
  auto& m = dt.mem();
  for (int j = 0; j < image_size; ++j)
    for (int i = 0; i < image_size; ++i) {
      m.u[j][i] = m.v[j][i] = 0;
      // m.side[j][i] = K * (!is_harbor(i + (image_size - 1) * dt.x(),
      //                                j + (image_size - 1) * dt.y()));
      m.side[j][i] = K;
      m.depth[j][i] = 1.0;
      // m.depth[j][i] = 2600.0 * shoal_factor(i + (image_size - 1) * dt.x(),
      //                                       j + (image_size - 1) * dt.y());
      // Add a drop using the global coordinate taking into account the halo
      m.w[j][i] = add_a_drop(i + (image_size - 1) * dt.x(),
                             j + (image_size - 1) * dt.y());
      // m.w[j][i] = add_a_drop(i, j);
    }
}

bool display(auto& dt) {
  auto& m = dt.mem();
  // __builtin_memcpy(frame, &m.w[0][0], image_size * image_size *
  // sizeof(data_type)); a.update_tile_data_image(t::x, t::y, &frame[0][0],
  // -1.0, 1.0);
  return dt.service().update_image(&m.w[0][0], -1.0, 1.0);
}

void compute(auto& dt) {
  auto& m = dt.mem();

  for (int j = 0; j < image_size; ++j)
    for (int i = 0; i < image_size - 1; ++i) {
      // dw/dx
      auto north = m.w[j][i + 1] - m.w[j][i];
      // Integrate horizontal speed
      m.u[j][i] += north * alpha;
    }

  for (int j = 0; j < image_size - 1; ++j)
    for (int i = 0; i < image_size; ++i) {
      // dw/dy
      auto vp = m.w[j + 1][i] - m.w[j][i];
      // Integrate vertical speed
      m.v[j][i] += vp * alpha;
    }

  dt.full_barrier();

  // Transfer first column of u to next memory module to the West
  if constexpr (dt.y() & 1) {
    if constexpr (dt.has_neighbor(aie::hw::dir::east)) {
      auto& east = dt.mem_east();
      for (int j = 0; j < image_size; ++j)
        m.u[j][image_size - 1] = east.u[j][0];
    }
  }
  if constexpr (!(dt.y() & 1)) {
    if constexpr (dt.has_neighbor(aie::hw::dir::west)) {
      auto& west = dt.mem_west();
      for (int j = 0; j < image_size; ++j)
        west.u[j][image_size - 1] = m.u[j][0];
    }
  }

  if constexpr (dt.has_neighbor(aie::hw::dir::south)) {
    auto& below = dt.mem_south();
    for (int i = 0; i < image_size; ++i)
      below.v[image_size - 1][i] = m.v[0][i];
  }

  dt.full_barrier();

  for (int j = 1; j < image_size; ++j)
    for (int i = 1; i < image_size; ++i) {
      // div speed
      auto wp = (m.u[j][i] - m.u[j][i - 1]) + (m.v[j][i] - m.v[j - 1][i]);
      wp *= m.side[j][i] * (m.depth[j][i] + m.w[j][i]);
      // Integrate depth
      m.w[j][i] += wp;
      // Add some dissipation for the damping
      m.w[j][i] *= damping;
    }

  dt.full_barrier();

  if constexpr (dt.has_neighbor(aie::hw::dir::north)) {
    auto& above = dt.mem_north();
    for (int i = 0; i < image_size; ++i)
      above.w[0][i] = m.w[image_size - 1][i];
  }

  dt.full_barrier();

  // Transfer last line of w to next memory module on the East
  if constexpr (dt.y() & 1) {
    if constexpr (dt.has_neighbor(aie::hw::dir::east)) {
      auto& east = dt.mem_east();
      for (int j = 0; j < image_size; ++j)
        east.w[j][0] = m.w[j][image_size - 1];
    }
  }
  if constexpr (!(dt.y() & 1)) {
    if constexpr (dt.has_neighbor(aie::hw::dir::west)) {
      auto& west = dt.mem_west();
      for (int j = 0; j < image_size; ++j)
        m.w[j][0] = west.w[j][image_size - 1];
    }
  }

  dt.full_barrier();
}

int main(int argc, char** argv) {
  aie::ext::application<data_type> a;
  aie::device<1, 1> dev;
  aie::queue q(dev);

  struct tile_data {
    data_type u[image_size][image_size];     //< Horizontal speed
    data_type v[image_size][image_size];     //< Vertical speed
    data_type w[image_size][image_size];     //< Local delta depth
    data_type side[image_size][image_size];  //< Hard wall limit
    data_type depth[image_size][image_size]; //< Average depth
  };

  a.start(argc, argv, dev.sizeX, dev.sizeY, image_size, image_size, 30)
      .get_image_grid()
      .get_palette()
      .set(aie::ext::palette::rainbow, 150, 2, 127);
  q.submit_uniform<tile_data>(
      [](auto& ht) {
        ht.single_task([](auto& dt) {
          auto m = dt.mem();
          double arr[16] = {92.0, 96.0, 92.0, 80.0, 96.0, 100.0, 96.0, 84.0, 92.0, 96.0, 92.0, 80.0, 80.0, 84.0, 80.0, 68.0};
          __builtin_memcpy(m.w, arr, sizeof(arr));
          // initialize_space(dt);
          display(dt);
          do {
            // compute(dt);
            // for (int j = 0; j < image_size; ++j)
            //   for (int i = 0; i < image_size; ++i) {
            //     m.w[j][i] = 96;
            //     // curr += 0.05;
            //   }
          } while (!display(dt));
        });
      },
      aie::add_service(a.get_service()));
}
