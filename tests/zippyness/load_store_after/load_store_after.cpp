// circle --verbose -O3 -std=c++20 -I../../.. -sm_80 load_store_after.cpp -lcudart -lfmt -o load_store_after
#include "measure_bandwidth_of_invocation.hpp"
#include "validate.hpp"
#include <algorithm>
#include <fmt/format.h>
#include <numeric>
#include <optional>
#include <vector>
#include <ubu/ubu.hpp>

// postcondition: is_injective(result)
constexpr ubu::layout_of_rank<3> auto layout_for_load_store(std::size_t n)
{
  using namespace ubu;

  auto num_elements_per_thread = 11_c;
  auto num_threads_per_block = 128_c;
  auto tile_size = num_elements_per_thread * num_threads_per_block;
  auto num_blocks = ceil_div(n, tile_size);

  std::tuple shape(num_elements_per_thread, num_threads_per_block, num_blocks);

  return strided_layout(shape);
}

template<std::size_t N>
ubu::fancy_span<std::byte*> static_shared_memory_view()
{
  return ubu::fancy_span(smem_ptr<N>(), N);
}

template<std::size_t N>
struct static_block_workspace
{
  ubu::fancy_span<std::byte*> buffer;
  ubu::cuda::block_workspace::barrier_type barrier;

  constexpr explicit static_block_workspace()
    : buffer(static_shared_memory_view<N>())
  {}
};

template<ubu::sized_vector_like I, ubu::sized_vector_like R>
ubu::cuda::event load_store_after(ubu::cuda::device_executor gpu, ubu::cuda::device_allocator<std::byte> alloc, const ubu::cuda::event& before, I input, R result)
{
  using namespace ubu;
  using T = tensor_element_t<I>;

  layout_of_rank<3> auto layout = layout_for_load_store(std::size(input));
  tensor_of_rank<3> auto input_tiles  = compose(input, layout);
  tensor_of_rank<3> auto result_tiles = compose(result, layout);

  auto shape = layout.shape();
  constexpr auto max_num_elements_per_thread = get<0>(shape);
  constexpr auto block_size = get<1>(shape);
  std::size_t num_blocks = get<2>(shape);

  constexpr auto tile_size = max_num_elements_per_thread * block_size;

  // kernel configuration
  std::pair kernel_shape(block_size, num_blocks);
  std::pair workspace_shape(constant<sizeof(T)>() * tile_size, 0_c);

  // circle build 203 -sm_80: 40 registers
  // 685.436 GB/s ~91% peak bandwidth on RTX A5000
  return bulk_execute_with_workspace_after(gpu,
                                           alloc,
                                           before,
                                           kernel_shape,
                                           workspace_shape,
                                           [=](ubu::int2 idx, auto workspace)
  {
    basic_cooperator grid(idx, kernel_shape, workspace);

    auto [block, block_idx] = subgroup_and_id(grid);

    // tiles of the input and result are two-dimensional
    matrix auto input_tile  = slice(input_tiles, std::tuple(_, _, block_idx));
    matrix auto result_tile = slice(result_tiles, std::tuple(_, _, block_idx));

    // tiles of the input and result are two-dimensional
    inplace_vector thread_values = coop_load_columns(block, input_tile);

    // store this thread's values
    coop_store_columns(block, thread_values, result_tile);
  });
}

template<class T>
using device_vector = std::vector<T, ubu::cuda::managed_allocator<T>>;

void test_load_store_after(std::size_t n)
{
  using namespace std;
  using namespace ubu;

  device_vector<int> input(n, 1);
  std::iota(input.begin(), input.end(), 0);
  device_vector<int> result(n);

  ubu::cuda::device_executor ex;
  ubu::cuda::device_allocator<std::byte> alloc;
  ubu::cuda::event before = initial_happening(ex);

  // compute the result on the GPU
  load_store_after(ex, alloc, before, std::span(input), std::span(result)).wait();

  // check the result
  validate_result(to_host(input), to_host(input), to_host(result), fmt::format("test_load_store_after({})", n));
}

void test_correctness(std::size_t max_size, bool verbose = false)
{
  for(auto sz: test_sizes(max_size))
  {
    if(verbose)
    {
      std::cout << "test_load_store_after(" << sz << ")...";
    }

    test_load_store_after(sz);

    if(verbose)
    {
      std::cout << "OK" << std::endl;
    }
  }
}

double test_performance(std::size_t size, std::size_t num_trials)
{
  device_vector<int> input(size);
  device_vector<int> result(size);

  std::span input_view(input);
  std::span result_view(result);

  ubu::cuda::device_executor ex;
  ubu::cuda::device_allocator<std::byte> alloc;

  ubu::cuda::event before = ubu::initial_happening(ex);

  // warmup
  load_store_after(ex, alloc, before, input_view, result_view);

  std::size_t num_bytes = input_view.size_bytes() + result_view.size_bytes();

  return measure_bandwidth_of_invocation_in_gigabytes_per_second(num_trials, num_bytes, [&]
  {
    load_store_after(ex, alloc, before, input_view, result_view);
  });
}

// these expected performance intervals are in units of percent of theoretical peak bandwidth
performance_expectations_t load_store_after_expectations = {
  {"NVIDIA GeForce RTX 3070", {0.91, 0.93}},
  {"NVIDIA RTX A5000", {0.90, 0.92}}
};


int main(int argc, char** argv)
{
  std::size_t performance_size = choose_large_problem_size_using_heuristic<int>(2);
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

  report_performance(bandwidth_to_performance(bandwidth), load_store_after_expectations);

  std::cout << "OK" << std::endl;

  return 0; 
}

