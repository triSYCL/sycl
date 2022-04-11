// RUN: true

/*
    Nbody example supplied by j-stephan in issue:
     https://github.com/triSYCL/sycl/issues/28

    It currently has a bunch of different bugs when compiling for HW
    Emulation. And is missing at least one math symbol (rsqrt) in the kernel
    runtime for sw emu (also likely hw_emu as its a shared issue).

    An ideal milestone for now would be getting this lovely example to compile
    in HW EMU! It is being added so it doesn't get lost in the issue history as
    it's an important example.

    This is tied to at least two issues opened by j-stephan and agozillon:
      https://github.com/triSYCL/sycl/issues/32
      https://github.com/triSYCL/sycl/issues/11
*/

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <future>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <CL/sycl.hpp>

constexpr auto eps = 0.001f;
constexpr auto eps2 = eps * eps;
constexpr auto damping = 0.5f;
constexpr auto delta_time = 0.2f;
constexpr auto iterations = 10;

// using these as there don't seem to be swizzles right now
struct float3 {
  float x;
  float y;
  float z;
};

struct float4 {
  float x;
  float y;
  float z;
  float w;
};

/******************************************************************************
 * Device-side N-Body
 *****************************************************************************/
auto force_calculation(
    float4 body_pos,

    cl::sycl::accessor<float4, 1, cl::sycl::access::mode::read,
                       cl::sycl::access::target::global_buffer>
        positions,

    std::size_t n) -> float3 {
  auto acc = float3{0.f, 0.f, 0.f};

  for (auto i = 0ul; i < n; ++i) {
    // r_ij [3 FLOPS]
    const auto j = positions[i];
    auto r = float3{};
    r.x = j.x - body_pos.x;
    r.y = j.y - body_pos.y;
    r.z = j.z - body_pos.z;

    // dist_sqr = dot(r_ij, r_ij) + EPS^2 [6 FLOPS]
    const auto dist_sqr = cl::sycl::fma(
        r.x, r.x, cl::sycl::fma(r.y, r.y, cl::sycl::fma(r.z, r.z, eps2)));

    // inv_dist_cube = 1/dist_sqr^(3/2) [4 FLOPS]
    auto dist_sixth = dist_sqr * dist_sqr * dist_sqr;
    auto inv_dist_cube = cl::sycl::rsqrt(dist_sixth);

    // s = m_j * inv_dist_cube [1 FLOP]
    const auto s = float{j.w} * inv_dist_cube;
    const auto s3 = float3{s, s, s};

    // a_i = a_i + s * r_ij [6 FLOPS]
    acc.x = cl::sycl::fma(r.x, s3.x, acc.x);
    acc.y = cl::sycl::fma(r.y, s3.y, acc.y);
    acc.z = cl::sycl::fma(r.z, s3.z, acc.z);
  }

  return acc;
}

struct body_integrator {
  cl::sycl::accessor<float4, 1, cl::sycl::access::mode::read,
                     cl::sycl::access::target::global_buffer>
      old_pos;

  cl::sycl::accessor<float4, 1, cl::sycl::access::mode::discard_write,
                     cl::sycl::access::target::global_buffer>
      new_pos;

  cl::sycl::accessor<float4, 1, cl::sycl::access::mode::read_write,
                     cl::sycl::access::target::global_buffer>
      vel;

  std::size_t n; // bodies

  auto operator()() -> void {
    for (auto i = 0ul; i < n; ++i) {
      auto position = old_pos[i];
      auto accel = force_calculation(position, old_pos, n);

      /*
       * acceleration = force / mass
       * new velocity = old velocity + acceleration * delta_time
       * note that the body's mass is canceled out here and in
       *  force_calculation. Thus force == acceleration
       */
      auto velocity = vel[i];

      velocity.x += accel.x * delta_time;
      velocity.y += accel.y * delta_time;
      velocity.z += accel.z * delta_time;

      velocity.x *= damping;
      velocity.y *= damping;
      velocity.z *= damping;

      position.x += velocity.x * delta_time;
      position.y += velocity.y * delta_time;
      position.z += velocity.z * delta_time;

      new_pos[i] = position;
      vel[i] = velocity;
    }
  }
};

auto main() -> int {
  try {
    // --------------------------------------------------------------------
    // init SYCL
    // --------------------------------------------------------------------

    // create queue on device
    auto exception_handler = [](cl::sycl::exception_list exceptions) {
      for (std::exception_ptr e : exceptions) {
        try {
          std::rethrow_exception(e);
        } catch (const cl::sycl::exception &err) {
          std::cerr << "Caught asynchronous SYCL exception: " << err.what()
                    << std::endl;
        }
      }
    };

    auto queue = cl::sycl::queue{exception_handler,
                                 cl::sycl::property::queue::enable_profiling{}};

    // --------------------------------------------------------------------
    // Init host memory
    // --------------------------------------------------------------------
    constexpr auto n = 262144ul;

    auto old_positions = std::vector<float4>{};
    auto new_positions = std::vector<float4>{};
    auto velocities = std::vector<float4>{};

    old_positions.resize(n);
    new_positions.resize(n);
    velocities.resize(n);

    auto gen = std::mt19937{std::random_device{}()};
    auto dis = std::uniform_real_distribution<float>{-42.f, 42.f};
    auto init_vec = [&]() { return float4{dis(gen), dis(gen), dis(gen), 0.f}; };

    std::generate(begin(old_positions), end(old_positions), init_vec);
    std::fill(begin(new_positions), end(new_positions), float4{});
    std::generate(begin(velocities), end(velocities), init_vec);

    // --------------------------------------------------------------------
    // Init device memory
    // --------------------------------------------------------------------
    auto d_old_positions =
        cl::sycl::buffer<float4, 1>{cl::sycl::range<1>{old_positions.size()}};
    d_old_positions.set_write_back(false);

    auto d_new_positions =
        cl::sycl::buffer<float4, 1>{cl::sycl::range<1>{new_positions.size()}};
    d_new_positions.set_write_back(false);

    auto d_velocities =
        cl::sycl::buffer<float4, 1>{cl::sycl::range<1>{velocities.size()}};
    d_velocities.set_write_back(false);

    queue.submit([&](cl::sycl::handler &cgh) {
      auto acc =
          d_old_positions.get_access<cl::sycl::access::mode::discard_write,
                                     cl::sycl::access::target::global_buffer>(
              cgh);

      cgh.copy(old_positions.data(), acc);
    });

    queue.submit([&](cl::sycl::handler &cgh) {
      auto acc =
          d_new_positions.get_access<cl::sycl::access::mode::discard_write,
                                     cl::sycl::access::target::global_buffer>(
              cgh);

      cgh.copy(new_positions.data(), acc);
    });

    queue.submit([&](cl::sycl::handler &cgh) {
      auto acc =
          d_velocities.get_access<cl::sycl::access::mode::discard_write,
                                  cl::sycl::access::target::global_buffer>(cgh);

      cgh.copy(velocities.data(), acc);
    });

    // --------------------------------------------------------------------
    // execute kernel
    // --------------------------------------------------------------------
    auto first_event = cl::sycl::event{};
    auto last_event = cl::sycl::event{};

    for (auto i = 0; i < iterations; ++i) {
      last_event = queue.submit([&, n](cl::sycl::handler &cgh) {
        auto old_acc =
            d_old_positions.get_access<cl::sycl::access::mode::read,
                                       cl::sycl::access::target::global_buffer>(
                cgh);

        auto new_acc =
            d_new_positions.get_access<cl::sycl::access::mode::discard_write,
                                       cl::sycl::access::target::global_buffer>(
                cgh);

        auto vel_acc =
            d_velocities.get_access<cl::sycl::access::mode::read_write,
                                    cl::sycl::access::target::global_buffer>(
                cgh);

        auto integrator = body_integrator{old_acc, new_acc, vel_acc, n};
        cgh.single_task(integrator);
      });

      if (i == 0)
        first_event = last_event;

      std::swap(d_old_positions, d_new_positions);
    }
    queue.wait_and_throw();

    // --------------------------------------------------------------------
    // results
    // --------------------------------------------------------------------
    auto start = first_event.get_profiling_info<
        cl::sycl::info::event_profiling::command_start>();
    auto stop =
        last_event
            .get_profiling_info<cl::sycl::info::event_profiling::command_end>();

    auto time_ns = stop - start;
    auto time_s = time_ns / 1e9;
    auto time_ms = time_ns / 1e6;

    constexpr auto flops_per_interaction = 20.;
    auto interactions = static_cast<double>(n * n);
    auto interactions_per_second = interactions * iterations / time_s;
    auto flops = interactions_per_second * flops_per_interaction;
    auto gflops = flops / 1e9;

    std::cout << n << ";" << time_ms << ";" << gflops << std::endl;
  } catch (const cl::sycl::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
