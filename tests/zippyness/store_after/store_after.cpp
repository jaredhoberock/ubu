// circle --verbose -O3 -std=c++20 -I../../../ -sm_60 store_after.cpp -lcudart -lfmt -o store_after
#include "measure_bandwidth_of_invocation.hpp"
#include "validate.hpp"
#include <algorithm>
#include <numeric>
#include <optional>
#include <vector>
#include <ubu/ubu.hpp>

// postcondition: is_injective(result)
constexpr ubu::layout_like_of_rank<3> auto layout_for_store(std::size_t n)
{
  using namespace ubu;

  auto num_elements_per_thread = 11_c;
  auto num_threads_per_block = 128_c;
  auto tile_size = num_elements_per_thread * num_threads_per_block;
  auto num_blocks = ceil_div(n, tile_size);

  std::tuple shape(num_elements_per_thread, num_threads_per_block, num_blocks);

  return strided_layout(shape);
}

template<ubu::sized_vector_like R>
ubu::cuda::event store_after(ubu::cuda::device_executor gpu, ubu::cuda::device_allocator<std::byte> alloc, const ubu::cuda::event& before, R result)
{
  using namespace ubu;
  using T = tensor_element_t<R>;

  layout_like_of_rank<3> auto layout = layout_for_store(std::size(result));
  tensor_like_of_rank<3> auto result_tiles = compose(result, layout);

  auto shape = layout.shape();

  constexpr auto max_num_elements_per_thread = get<0>(shape);
  constexpr auto block_size = get<1>(shape);
  std::size_t num_blocks = get<2>(shape);

  constexpr auto tile_size = max_num_elements_per_thread * block_size;

  // kernel configuration
  std::pair kernel_shape(block_size, num_blocks);
  std::pair workspace_shape(constant<sizeof(T)>() * tile_size, 0_c);

  // 20 registers / 408.911 GB/s ~ 93% peak bandwidth on RTX 3070
  return bulk_execute_with_workspace_after(gpu,
                                           alloc,
                                           before,
                                           kernel_shape,
                                           workspace_shape,
                                           [=](ubu::int2 idx, auto workspace)
  {
    basic_cooperator grid(idx, kernel_shape, workspace);

    auto [block, block_idx] = subgroup_and_id(grid);

    // tiles of result are two-dimensional
    matrix_like auto result_tile = slice(result_tiles, std::tuple(_, _, block_idx));

    // generate output for this thread as an inplace_vector
    auto my_result_slice = slice(result_tile, std::pair(_,id(block)));
    auto max_num_elements_per_thread = get<0>(shape);
    size_t thread_begin = my_result_slice.span().data() - result.data();

    inplace_vector<T,max_num_elements_per_thread> thread_values(my_result_slice.span().size());

    #pragma unroll
    for(int i = 0; i < max_num_elements_per_thread; ++i)
    {
      if(i < thread_values.size())
      {
        thread_values[i] = thread_begin + i;
      }
    }

    coop_store(block, thread_values, result_tile.span());
  });
}

template<class T>
using device_vector = std::vector<T, ubu::cuda::managed_allocator<T>>;

void test_store_after(std::size_t n)
{
  using namespace std;
  using namespace ubu;

  device_vector<int> result(n);

  cuda::device_executor ex;
  cuda::device_allocator<std::byte> alloc;
  cuda::event before = initial_happening(ex);

  // compute the result on the GPU
  store_after(ex, alloc, before, std::span(result)).wait();

  device_vector<int> expected(n, 1);
  std::iota(expected.begin(), expected.end(), 0);

  // check the result
  validate_result(to_host(expected), to_host(expected), to_host(result), fmt::format("test_store_after({})", n));
}

void test_correctness(std::size_t max_size, bool verbose = false)
{
  for(auto sz: test_sizes(max_size))
  {
    if(verbose)
    {
      std::cout << "test_store_after(" << sz << ")...";
    }

    test_store_after(sz);

    if(verbose)
    {
      std::cout << "OK" << std::endl;
    }
  }
}

double test_performance(std::size_t size, std::size_t num_trials)
{
  device_vector<int> result(size);

  std::span result_view(result);

  ubu::cuda::device_executor ex;
  ubu::cuda::device_allocator<std::byte> alloc;

  ubu::cuda::event before = ubu::initial_happening(ex);

  // warmup
  store_after(ex, alloc, before, result_view);

  std::size_t num_bytes = result_view.size_bytes();

  return measure_bandwidth_of_invocation_in_gigabytes_per_second(num_trials, num_bytes, [&]
  {
    store_after(ex, alloc, before, result_view);
  });
}

// these expected performance intervals are in units of percent of theoretical peak bandwidth
performance_expectations_t store_after_expectations = {
  {"NVIDIA GeForce RTX 3070", {0.91, 0.93}},
  {"NVIDIA RTX A5000", {0.94, 0.96}}
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

  report_performance(bandwidth_to_performance(bandwidth), store_after_expectations);

  std::cout << "OK" << std::endl;

  return 0; 
}

