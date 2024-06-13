// circle --verbose -O3 -std=c++20 -I. -sm_60 load_scan_store_tiles.cpp -lcudart -lfmt -o load_scan_store_tiles
#include "validate.hpp"
#include "measure_bandwidth_of_invocation.hpp"
#include <algorithm>
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
  // XXX this loop isn't quite right because element_exists(vec, 0) could be false
  //     we would want to maintain a cumulative sum and check that each element exists before applying f
  //     or at least remember the index of the previous element
  // XXX alternatively, we should define sized_vector_like to mean that all elements [0, size()) exist

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
constexpr ubu::layout_of_rank<4> auto layout_for_scan(std::size_t n)
{
  using namespace ubu;

  auto num_elements_per_thread = 11_c;
  auto num_threads_per_block = 128_c;
  auto num_tiles_per_block = 2_c;
  auto tile_size = num_elements_per_thread * num_threads_per_block;
  auto num_blocks = ceil_div(n, num_tiles_per_block * tile_size);

  std::tuple shape(num_elements_per_thread, num_threads_per_block, num_tiles_per_block, num_blocks);

  return strided_layout(shape);
}

template<ubu::sized_vector_like I, ubu::sized_vector_like R, class BinaryFunction>
ubu::cuda::event load_scan_store_tiles_after(ubu::cuda::device_executor gpu, ubu::cuda::device_allocator<std::byte> alloc, const ubu::cuda::event& before, I input, R result, BinaryFunction op)
{
  using namespace ubu;
  using T = tensor_element_t<I>;

  auto layout = layout_for_scan(std::size(input));
  tensor_like_of_rank<4> auto input_tiles  = compose(input, layout);
  tensor_like_of_rank<4> auto result_tiles = compose(result, layout);

  auto [max_num_elements_per_thread, block_size, num_tiles_per_block, num_blocks] = shape(layout);
  auto tile_size = max_num_elements_per_thread * block_size;

  // kernel configuration
  std::pair shape(block_size, num_blocks);
  std::pair workspace_shape(sizeof(T) * tile_size, 0);

  // 61 registers / 402.968 GB/s ~ 92% peak bandwidth
  return bulk_execute_with_workspace_after(gpu,
                                           alloc,
                                           before,
                                           shape,
                                           workspace_shape,
                                           [=](ubu::int2 idx, auto workspace)
  {
    basic_cooperator grid(idx, shape, workspace);

    auto [block, block_idx] = subgroup_and_id(grid);

    std::optional<T> tile_carry_in;

    for(int tile_idx = 0; tile_idx < num_tiles_per_block; ++tile_idx)
    {
      // tiles of the input and result are two-dimensional
      matrix_like auto input_tile  = slice(input_tiles, std::tuple(_, _, tile_idx, block_idx));
      matrix_like auto result_tile = slice(result_tiles, std::tuple(_, _, tile_idx, block_idx));

      // load each thread's column of the input
      inplace_vector thread_values = coop_load_columns(block, input_tile);
                                                             
      // each thread does a scan
      inplace_inclusive_scan(thread_values, op);

      // scan across the block to compute each thread's carry-in
      std::optional thread_carry_in = coop_exclusive_scan(block, tile_carry_in, thread_values.maybe_back(), op);

      // accumulate the thread's carry-in
      inplace_transform(mask(thread_values, thread_carry_in.has_value()), [&](auto x)
      {
        return op(*thread_carry_in, x);
      });

      // broadcast the next iteration's carry-in
      tile_carry_in = broadcast(block, last_id(block), thread_values.maybe_back());

      // store each thread's column of the result
      coop_store_columns(block, thread_values, result_tile);
    }
  });
}

template<class T>
using device_vector = std::vector<T, ubu::cuda::managed_allocator<T>>;

template<ubu::sized_vector_like I, class BinaryFunction>
  requires std::is_trivially_copyable_v<I>
std::vector<ubu::tensor_element_t<I>> sequential_inclusive_scan_tiles(I input, BinaryFunction op)
{
  using namespace std;
  using namespace ubu;

  using T = tensor_element_t<I>;

  auto h_input = to_host(input);
  auto tiles = compose(std::span(h_input), layout_for_scan(h_input.size()));

  vector<T> result;
  result.reserve(h_input.size());

  auto [max_num_elements_per_thread, block_size, num_tiles_per_block, num_blocks] = shape(tiles);
  auto tile_size = max_num_elements_per_thread * block_size;

  // loop over blocks
  for(int block_idx = 0; block_idx != num_blocks; ++block_idx)
  {
    std::optional<T> carry_in;

    // loop over tiles
    for(int tile_idx = 0; tile_idx != num_tiles_per_block; ++tile_idx)
    {
      auto tile = slice(tiles, tuple(_, _, tile_idx, block_idx));

      vector<T> result_tile;

      // this loop is inclusive_scan
      for(auto x : tile)
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

      result.insert(result.end(), result_tile.begin(), result_tile.end());
    }
  }

  return result;
}

void test_load_scan_store_after(std::size_t n)
{
  using namespace std;
  using namespace ubu;

  device_vector<int> input(n, 1);
  std::iota(input.begin(), input.end(), 0);
  device_vector<int> result(n);

  cuda::device_executor ex;
  cuda::device_allocator<std::byte> alloc;
  cuda::event before = initial_happening(ex);

  // compute the result on the GPU
  load_scan_store_tiles_after(ex, alloc, before, std::span(input), std::span(result), plus()).wait();

  // compute the expected result on the CPU
  vector<int> expected = sequential_inclusive_scan_tiles(std::span(input), plus());

  // check the result
  validate_result(expected, to_host(input), to_host(result), fmt::format("test_load_scan_store_after({})", n));
}

void test_correctness(std::size_t max_size, bool verbose = false)
{
  for(auto sz: test_sizes(max_size))
  {
    if(verbose)
    {
      std::cout << "test_load_scan_store_tiles_after(" << sz << ")...";
    }

    test_load_scan_store_after(sz);

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
  load_scan_store_tiles_after(ex, alloc, before, input_view, result_view, std::plus{});

  std::size_t num_bytes = input_view.size_bytes() + result_view.size_bytes();

  return measure_bandwidth_of_invocation_in_gigabytes_per_second(num_trials, num_bytes, [&]
  {
    load_scan_store_tiles_after(ex, alloc, before, input_view, result_view, std::plus{});
  });
}

// XXX the reason this kernel's performance is so low is because circle build 201 is not inlining load_scan_store_after's lambda
//     with inlining, the performance should be ~92% peak bandwidth
performance_expectations_t load_scan_store_tiles_expectations = {
  {"NVIDIA GeForce RTX 3070", {0.50, 0.52}},
  {"NVIDIA RTX A5000", {0.50, 0.52}}
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

  report_performance(bandwidth_to_performance(bandwidth), load_scan_store_tiles_expectations);

  std::cout << "OK" << std::endl;

  return 0; 
}

