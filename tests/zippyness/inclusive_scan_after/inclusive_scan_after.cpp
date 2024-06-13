// circle --verbose -O3 -std=c++20 -I../../.. -sm_80 inclusive_scan_after.cpp -lcudart -lfmt -o inclusive_scan_after
#include "lookback_array.hpp"
#include "maybe_add.hpp"
#include "measure_bandwidth_of_invocation.hpp"
#include "validate.hpp"
#include <atomic>
#include <fmt/core.h>
#include <numeric>
#include <optional>
#include <vector>
#include <ubu/ubu.hpp>


// XXX these should be turned into CPOs that would call all(...) on a non-view

// XXX this needs to require that vec is writeable or something
template<ubu::vector_like V, class Function>
constexpr void inplace_transform(V&& vec, Function f)
{
  using namespace ubu;

  #pragma unroll
  for(int i = 0; i != shape(vec); ++i)
  {
    if(element_exists(vec, i))
    {
      vec[i] = f(vec[i]);
    }
  }
}

// XXX this needs to require that vec is writeable or something
template<ubu::sized_vector_like V, class BinaryFunction>
constexpr void inplace_inclusive_scan(V&& vec, BinaryFunction f)
{
  using namespace ubu;

  #pragma unroll
  for(int i = 1; i != shape(vec); ++i)
  {
    if(element_exists(vec, i))
    {
      vec[i] = f(vec[i-1], vec[i]);
    }
  }
}

// postcondition: is_injective(result)
constexpr ubu::layout_of_rank<3> auto layout_for_scan(std::size_t n)
{
  using namespace ubu;

  // these parameters have been tuned for sm_80 on RTX 3070
  auto num_elements_per_thread = 17_c;
  auto num_threads_per_block = 128_c;

  auto tile_size = num_elements_per_thread * num_threads_per_block;
  auto num_blocks = ceil_div(n, tile_size);

  std::tuple shape(num_elements_per_thread, num_threads_per_block, num_blocks);

  return strided_layout(shape);
}

template<ubu::hierarchical_cooperator C, ubu::integral_like S, ubu::integral_like B, std::invocable<ubu::child_cooperator_t<C>> F>
constexpr std::invoke_result_t<F, ubu::child_cooperator_t<C>> invoke_in_subgroup_and_broadcast(C self, S which_subgroup, B broadcaster, F f)
{
  using namespace ubu;

  auto [subgroup, subgroup_id] = subgroup_and_id(self);

  std::invoke_result_t<F, ubu::child_cooperator_t<C>> result{};
  if(subgroup_id == which_subgroup)
  {
    result = f(subgroup);
  }

  return broadcast(self, broadcaster, result);
}

template<ubu::sized_vector_like I, ubu::sized_vector_like R, class BinaryFunction>
ubu::cuda::event inclusive_scan_after(ubu::cuda::device_executor gpu, ubu::cuda::device_allocator<std::byte> alloc, const ubu::cuda::event& before, I input, R result, ubu::tensor_element_t<I> init, BinaryFunction op)
{
  using namespace ubu;
  using T = tensor_element_t<I>;

  // arrange the input and result into 3D tensors
  auto layout = layout_for_scan(std::size(input));
  tensor_like_of_rank<3> auto input_tiles  = compose(input, layout);
  tensor_like_of_rank<3> auto result_tiles = compose(result, layout);

  // configure kernel launch
  auto [max_num_elements_per_thread, block_size, num_blocks] = shape(layout);
  auto tile_size = max_num_elements_per_thread * block_size;

  std::pair shape(block_size, num_blocks);
  std::pair workspace_shape(sizeof(T) * tile_size, dynamic_size_in_bytes_of_lookback_array<T>(num_blocks));

  // sm_80: 61 registers / 399.384 GB/s ~ 91% peak bandwidth on RTX 3070
  // circle build 201, llvm 18
  return bulk_execute_with_workspace_after(gpu, alloc,
                                           before,
                                           shape, workspace_shape,
                                           [=](ubu::int2 coord, auto workspace)
  {
    basic_cooperator grid(coord, shape, workspace);

    // this array implements the Merrill-Garland 2016 "decoupled lookback" algorithm
    lookback_array tile_sums(grid, num_blocks, init);

    auto [block, block_idx] = subgroup_and_id(grid);

    // tiles of the input and result are two-dimensional
    matrix_like auto input_mtx  = slice(input_tiles, std::tuple(_, _, block_idx));
    matrix_like auto result_mtx = slice(result_tiles, std::tuple(_, _, block_idx));

    // cooperatively load and sequentially scan each thread's column of the input
    inplace_vector thread_values = coop_load_columns(block, input_mtx);
    inplace_inclusive_scan(thread_values, op);

    // scan across the block to compute each thread's carry-in and the sum of the tile
    auto [thread_carry_in, tile_sum] = coop_exclusive_scan_and_fold(block, thread_values.maybe_back(), op);

    // warp 0 stores the tile sum and loads the tile's carry-in; broadcast back from lane 0
    std::optional tile_carry_in = invoke_in_subgroup_and_broadcast(block, 0, 0, [&](auto warp_0)
    {
      return tile_sums.coop_store_sum_and_load_carry_in(warp_0, block_idx, tile_sum, op);
    });

    // accumulate the tile's carry-in into each thread's carry-in
    thread_carry_in = maybe_add(tile_carry_in, thread_carry_in, op);

    // accumulate the thread's carry-in
    inplace_transform(mask(thread_values, thread_carry_in.has_value()), [&](auto value)
    {
      return op(*thread_carry_in, value);
    });

    // store each thread's column of the result
    coop_store_columns(block, thread_values, result_mtx);
  });
}

template<class T>
using device_vector = std::vector<T, ubu::cuda::managed_allocator<T>>;

template<ubu::sized_vector_like V, class BinaryFunction, class I>
  requires std::is_trivially_copyable_v<V>
std::vector<ubu::tensor_element_t<V>> sequential_inclusive_scan(V input, BinaryFunction op, I init)
{
  using namespace std;
  using namespace ubu;

  using T = tensor_element_t<V>;

  auto h_input = to_host(input);

  vector<T> result;
  result.reserve(h_input.size());

  // this loop is inclusive_scan
  optional<T> carry_in = init;
  for(auto x : h_input)
  {
    if(carry_in)
    {
      result.push_back(op(*carry_in, x));
    }
    else
    {
      result.push_back(x);
    }

    carry_in = result.back();
  }

  return result;
}

void test_inclusive_scan_after(std::size_t n)
{
  using namespace std;
  using namespace ubu;

  int init = 13;
  device_vector<int> input(n, 1);
  std::iota(input.begin(), input.end(), 0);
  device_vector<int> result(n);

  cuda::device_executor ex;
  cuda::device_allocator<std::byte> alloc;
  cuda::event before = initial_happening(ex);

  // compute the result on the GPU
  inclusive_scan_after(ex, alloc, before, std::span(input), std::span(result), init, plus()).wait();

  // compute the expected result on the CPU
  vector<int> expected = sequential_inclusive_scan(std::span(input), plus(), init);

  // check the result
  validate_result(expected, to_host(input), to_host(result), fmt::format("test_inclusive_scan_after({})", n));
}

void test_correctness(std::size_t max_size, bool verbose = false)
{
  for(auto sz: test_sizes(max_size))
  {
    if(verbose)
    {
      std::cout << "test_inclusive_scan_after(" << sz << ")..." << std::flush;
    }

    test_inclusive_scan_after(sz);

    if(verbose)
    {
      std::cout << "OK" << std::endl;
    }
  }
}

double test_performance(std::size_t size, std::size_t num_trials)
{
  int init = 13;
  device_vector<int> input(size);
  device_vector<int> result(size);

  std::span input_view(input);
  std::span result_view(result);

  ubu::cuda::device_executor ex;
  ubu::cuda::device_allocator<std::byte> alloc;

  ubu::cuda::event before = ubu::initial_happening(ex);

  // warmup
  inclusive_scan_after(ex, alloc, before, input_view, result_view, init, std::plus{});

  std::size_t num_bytes = input_view.size_bytes() + result_view.size_bytes();

  return measure_bandwidth_of_invocation_in_gigabytes_per_second(num_trials, num_bytes, [&]
  {
    inclusive_scan_after(ex, alloc, before, input_view, result_view, init, std::plus{});
  });
}

// these expected performance intervals are in units of percent of theoretical peak bandwidth
// XXX the reason this kernel's performance is so low is because circle build 201 is generating a stack frame for inclusive_scan_after's lambda
//     without the stack frame, the performance should be ~91% peak bandwidth on RTX 3070
performance_expectations_t single_pass_reduce_expectations = {
//  {"NVIDIA GeForce RTX 3070", {0.90, 0.92}},
//  {"NVIDIA RTX A5000", {0.90, 0.92}}
  {"NVIDIA GeForce RTX 3070", {0.19, 0.20}},
  {"NVIDIA RTX A5000", {0.18, 0.19}}
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

  report_performance(bandwidth_to_performance(bandwidth), single_pass_reduce_expectations);

  std::cout << "OK" << std::endl;

  return 0; 
}

