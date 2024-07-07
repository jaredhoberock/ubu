// circle -std=c++20 -O3 -I../../../ -sm_80 --verbose single_phase_coop_reduce.cpp -L/usr/local/cuda/lib64 -lcudart -lfmt -o single_phase_coop_reduce.out
#include "measure_bandwidth_of_invocation.hpp"
#include "validate.hpp"
#include <algorithm>
#include <cassert>
#include <concepts>
#include <fmt/format.h>
#include <numeric>
#include <random>
#include <span>
#include <ubu/ubu.hpp>


constexpr ubu::layout_like_of_rank<3> auto coop_reduction_layout(std::size_t n, int device = 0)
{
  using namespace ubu;

  int num_elements_per_thread = 16;
  auto block_size = 256_c;
  auto max_num_blocks = 4 * multiprocessor_count(device);
  auto num_elements_per_block = num_elements_per_thread * block_size;

  auto num_blocks = ceil_div(n, num_elements_per_block);

  // if we exceed the maximum number of blocks, recompute num_elements_per_block and num_elements_per_thread
  if(num_blocks > max_num_blocks)
  {
    num_blocks = max_num_blocks;

    num_elements_per_block = ceil_div(n, num_blocks);
    num_elements_per_thread = ceil_div(num_elements_per_block, block_size); 
    num_elements_per_block = num_elements_per_thread * block_size;
  }

  std::tuple shape(num_elements_per_thread, block_size, num_blocks);

  // offset = block_size*i + 1*threadIdx.x + num_elements_per_block*blockIdx.x;
  std::tuple stride(block_size, 1_c, num_elements_per_block);

  return strided_layout(shape, stride);
}


template<ubu::sized_vector_like V>
  requires std::is_trivially_copyable_v<V>
constexpr ubu::matrix_like auto as_reduction_matrix(V vec, ubu::cuda::coop_executor ex)
{
  // create a 3d layout
  ubu::layout_like_of_rank<3> auto layout = coop_reduction_layout(ubu::size(vec), ex.device());

  // create a 3d view of the data
  ubu::tensor_like_of_rank<3> auto view3d = ubu::compose(vec, layout);

  // nestle the 3d view into a 2d matrix of slices
  return ubu::nestle(view3d);
}


template<ubu::sized_vector_like I, std::random_access_iterator R, class F>
ubu::cuda::event single_phase_coop_reduce(ubu::cuda::coop_executor gpu, ubu::cuda::device_allocator<std::byte> alloc, const ubu::cuda::event& before, I input, R result, F op)
{
  using namespace ubu;
  using namespace std;
  using T = tensor_element_t<I>;

  // arrange the input into a 2D matrix of 1D tiles
  matrix_like auto tiles = as_reduction_matrix(input, gpu);

  auto grid_shape = shape(tiles);
  auto [num_threads, num_blocks] = grid_shape;

  // each block needs num_warps Ts in its workspace
  int block_workspace_shape = sizeof(T) * (num_threads / cuda::warp_size);

  // the grid needs num_blocks Ts in its workspace
  int grid_workspace_shape = sizeof(T) * num_blocks;

  pair workspace_shape(block_workspace_shape, grid_workspace_shape);

  // circle build 208 -sm_80: 25 registers
  // 712.125 GB/s ~ 95% peak bandwidth on RTX A5000 for large sizes
  return bulk_execute_with_workspace_after(gpu, alloc,
                                           before,
                                           grid_shape, workspace_shape,
                                           [=](ubu::int2 coord, auto workspace)
  {
    // sequentially reduce this thread's tile
    optional thread_sum = reduce(tiles[coord], op);

    // create a representation for the grid of threads
    basic_cooperator grid(coord, grid_shape, workspace);

    // cooperatively reduce the entire grid's results
    if(optional grid_sum = coop_reduce(grid, thread_sum, op))
    {
      // a single thread stores the sum
      *result = *grid_sum;
    }
  });

  return ubu::initial_happening(gpu);
}


template<class T>
using device_vector = std::vector<T, ubu::cuda::managed_allocator<T>>;


void test_single_phase_coop_reduce(std::size_t n)
{
  using namespace std;
  using namespace ubu;

  auto op = plus{};
  device_vector<int> input(n, 1);
  device_vector<int> result(1, 0);

  cuda::coop_executor ex;
  cuda::device_allocator<std::byte> alloc;
  cuda::event before = initial_happening(ex);

  // compare the result of single_phase_coop_reduce to a reference

  validate(
    // reference
    [&]()
    {
      auto h_input = to_host(input);
      return std::accumulate(h_input.begin(), h_input.end(), 0, op);
    },

    // test function
    [&]()
    {
      single_phase_coop_reduce(ex, alloc, before, std::span(input), result.data(), op).wait();
      return result[0];
    },

    // test name
    std::format("test_single_phase_coop_reduce({})", n)
  );
}


void test_correctness(std::size_t max_size, bool verbose = false)
{
  for(auto sz: test_sizes(max_size))
  {
    if(verbose)
    {
      std::cout << "test_single_phase_coop_reduce(" << sz << ")..." << std::flush;
    }

    test_single_phase_coop_reduce(sz);

    if(verbose)
    {
      std::cout << "OK" << std::endl;
    }
  }
}


double test_performance(std::size_t size, int num_trials)
{
  using namespace ubu;

  device_vector<int> input(size, 1);
  device_vector<int> result(1);
  cuda::coop_executor ex;
  cuda::device_allocator<std::byte> alloc;

  ubu::cuda::event before = ubu::initial_happening(ex);

  // warmup
  single_phase_coop_reduce(ex, alloc, before, std::span(input.data(), input.size()), result.data(), std::plus{});

  return measure_bandwidth_of_invocation_in_gigabytes_per_second(num_trials, sizeof(int) * input.size(), [&]
  {
    single_phase_coop_reduce(ex, alloc, before, std::span(input.data(), input.size()), result.data(), std::plus{});
  });
}

// these expected performance intervals are in units of percent of theoretical peak bandwidth
performance_expectations_t single_phase_coop_reduce_expectations = {
  {"NVIDIA GeForce RTX 3070", {0.00, 0.00}},
  {"NVIDIA RTX A5000", {0.90, 0.96}}
};


int main(int argc, char** argv)
{
  std::size_t performance_size = choose_large_problem_size_using_heuristic<int>(1);
  std::size_t num_performance_trials = 1000;
  std::size_t correctness_size = performance_size;

  if(argc == 2)
  {
    std::string_view arg(argv[1]);
    if(arg != "quick")
    {
      std::cerr << "Unrecognized argument \"" << arg << "\"" << std::endl;
      return -1;
    }

    correctness_size = 1 << 16;
    performance_size /= 10;
    num_performance_trials = 30;
  }

  std::cout << "Testing correctness... " << std::flush;
  test_correctness(correctness_size, correctness_size > 23456789);
  std::cout << "Done." << std::endl;
  
  std::cout << "Testing performance... " << std::flush;
  double bandwidth = test_performance(performance_size, num_performance_trials);
  std::cout << "Done." << std::endl;

  report_performance(bandwidth_to_performance(bandwidth), single_phase_coop_reduce_expectations);

  std::cout << "OK" << std::endl;

  return 0; 
}

