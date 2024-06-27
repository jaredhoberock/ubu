// circle -std=c++20 -O3 -I../../.. -sm_80 --verbose single_pass_reduce.cpp -L/usr/local/cuda/lib64 -lcudart -lfmt -o single_pass_reduce.out
#include "atomic_accumulator.hpp"
#include "measure_bandwidth_of_invocation.hpp"
#include "validate.hpp"
#include <algorithm>
#include <cassert>
#include <concepts>
#include <numeric>
#include <random>
#include <span>
#include <ubu/ubu.hpp>

template<ubu::vector_like V, std::invocable<ubu::tensor_element_t<V>,ubu::tensor_element_t<V>> F>
constexpr std::optional<ubu::tensor_element_t<V>> reduce(V vector, F binary_op)
{
  std::optional<ubu::tensor_element_t<V>> result;

  bool found_first_element = false;

  for(int i = 0; i < ubu::shape(vector); ++i)
  {
    if(ubu::element_exists(vector, i))
    {
      const auto& element = ubu::element(vector,i);

      if(found_first_element)
      {
        *result = binary_op(*result, element);
      }
      else
      {
        result = element;
        found_first_element = true;
      }
    }
  }

  return result;
}

// postcondition: is_injective(result)
constexpr ubu::layout_of_rank<3> auto even_share_layout(std::size_t num_elements, int device = 0)
{
  using namespace std;
  using namespace ubu;
  
  auto num_elements_per_thread = 16_c;
  auto block_size = 256_c;

  // XXX taken from CUB: reduce_config.sm_occupancy * sm_count * CUB_SUBSCRIPTION_FACTOR(0)
  int occupancy = 6; // XXX this value needs to be computed by the calculator, somehow
  int subscription = 5; // XXX this value should depend on the device
  int max_num_blocks = occupancy * multiprocessor_count(device) * subscription; 

  auto num_elements_per_tile = num_elements_per_thread * block_size;
  std::size_t num_tiles = ceil_div(num_elements, num_elements_per_tile);
  int num_blocks = std::min<std::size_t>(num_tiles, max_num_blocks);

  // we have a number of tiles to consume, and we want to divide them between "small" blocks and "big" blocks
  // to construct our layout, we need to find the number of each type of block, and the number of tiles each
  // type of block will consume

  auto num_tiles_per_small_block = div_or_zero(num_tiles, num_blocks);
  auto num_elements_per_small_block = num_tiles_per_small_block * num_elements_per_tile;

  // big blocks each receive an additional tile of elements
  auto num_elements_per_big_block = num_elements_per_small_block + num_elements_per_tile;

  auto num_big_blocks = num_tiles - num_tiles_per_small_block * num_blocks;
  auto num_tiles_per_big_block = num_elements_per_big_block / num_elements_per_tile; 

  auto num_small_blocks = num_blocks - num_big_blocks;

  // XXX consider making these layouts 4D so we can isolate the constant-sized work in the leading dimension
  //     then, the kernel would need each block to iterate over its tiles
  //     alternatively, we could nestle the first two dimensions of the layout, giving each thread a matrix to iterate over
  //     then the kernel body wouldn't need to change at all

  // create the layout for big blocks
  tuple big_block_shape(num_elements_per_thread * num_tiles_per_big_block, block_size, num_big_blocks);
  tuple big_block_stride(block_size, 1_c, num_elements_per_big_block);
  strided_layout big_block_layout(big_block_shape, big_block_stride);

  // create the layout for small blocks
  tuple small_block_shape(num_elements_per_thread * num_tiles_per_small_block, block_size, num_small_blocks);
  tuple small_block_stride(block_size, 1_c, num_elements_per_small_block);
  strided_layout small_block_layout(small_block_shape, small_block_stride);

  // combine the two layouts by stacking along the block dimension (i.e., 2):
  // big blocks come first, followed by small blocks
  return ubu::stack<2>(big_block_layout, offset(small_block_layout, big_block_layout.size()));
}


template<ubu::sized_vector_like V>
  requires std::is_trivially_copyable_v<V>
constexpr ubu::matrix_like auto as_reduction_matrix(V vec, ubu::cuda::device_executor gpu)
{
  // create a 3d layout
  auto layout = even_share_layout(ubu::size(vec), gpu.device());

  // create a 3d view of the data
  ubu::tensor_like_of_rank<3> auto view3d = ubu::compose(vec, layout);

  // nestle the 3d view into a 2d matrix of slices
  return ubu::nestle(view3d);
}

template<ubu::sized_vector_like I, indirectly_rw R, ubu::elemental_invocable<I,I> F>
ubu::cuda::event inplace_reduce_after(ubu::cuda::device_executor gpu, ubu::cuda::device_allocator<std::byte> alloc, const ubu::cuda::event& before, I input, R result, F op)
{
  using namespace ubu;

  // arrange the input into a 2D matrix of 1D tiles
  matrix_like auto tiles = as_reduction_matrix(input, gpu);

  auto shape = ubu::shape(tiles);
  auto [block_size, num_blocks] = shape;

  // each block needs num_warps Ts in its workspace
  using T = tensor_element_t<I>;
  auto block_workspace_size = sizeof(T)*(block_size / cuda::warp_size);
  auto grid_workspace_size = dynamic_size_in_bytes_of_atomic_accumulator(result, op);

  std::pair workspace_shape(block_workspace_size, grid_workspace_size);

  // circle build 208 -sm_80: 26 registers
  // 421.702 GB/s ~ 96% peak bandwidth on RTX 3070
  // 712.904 GB/s ~ 95% peak bandwidth on RTX A5000
  return bulk_execute_with_workspace_after(gpu, alloc,
                                           before,
                                           shape, workspace_shape,
                                           [=](ubu::int2 coord, auto workspace)
  {
    basic_cooperator grid(coord, shape, workspace);

    // this thing will handle atomic updates to the result
    atomic_accumulator accum(grid, result, op);

    // sequentially reduce this thread's tile
    std::optional thread_sum = reduce(tiles[coord], op);

    // cooperatively reduce this block's results
    auto block = subgroup(grid);
    std::optional block_sum = coop_reduce(block, thread_sum, op);

    // the leader of the block contributes to the sum
    if(block_sum)
    {
      accum.accumulate(*block_sum);
    }
  });
}

template<ubu::sized_vector_like I, class T, indirectly_rw R, ubu::elemental_invocable<I,I> F>
  requires std::indirectly_writable<R,T>
ubu::cuda::event single_pass_reduce(ubu::cuda::device_executor gpu, ubu::cuda::device_allocator<std::byte> alloc, const ubu::cuda::event& before, I input, T init, R result, F op)
{
  using namespace ubu;

  // first, initialize the result
  auto finished_init = execute_after(gpu, before, [=]
  {
    *result = init;
  });

  // accumulate to the result
  return inplace_reduce_after(gpu, alloc, finished_init, input, result, op);
}

template<class T>
using device_vector = std::vector<T, ubu::cuda::managed_allocator<T>>;


void test_single_pass_reduce(std::size_t n)
{
  using namespace std;
  using namespace ubu;

  int init = 13;
  auto op = plus{};
  device_vector<int> input(n, 1);
  device_vector<int> result(1, 0);

  cuda::device_executor ex;
  cuda::device_allocator<std::byte> alloc;
  cuda::event before = initial_happening(ex);

  // compare the result of single_pass_reduce to a reference

  validate(
    // reference
    [&]()
    {
      auto h_input = to_host(input);
      return std::accumulate(h_input.begin(), h_input.end(), init, op);
    },

    // test function
    [&]()
    {
      single_pass_reduce(ex, alloc, before, std::span(input), init, result.data(), op).wait();
      return result[0];
    },

    // test name
    std::format("test_single_pass_reduce({})", n)
  );
}


void test_correctness(std::size_t max_size, bool verbose = false)
{
  for(auto sz: test_sizes(max_size))
  {
    if(verbose)
    {
      std::cout << "test_single_pass_reduce(" << sz << ")..." << std::flush;
    }

    test_single_pass_reduce(sz);

    if(verbose)
    {
      std::cout << "OK" << std::endl;
    }
  }
}


double test_performance(std::size_t size, int num_trials)
{
  using namespace ubu;
  using namespace std;

  device_vector<int> input(size, 1);
  int init = 13;
  auto op = plus{};
  device_vector<int> result(1);

  cuda::device_executor ex;
  cuda::device_allocator<std::byte> alloc;

  ubu::cuda::event before = ubu::initial_happening(ex);

  // warmup
  single_pass_reduce(ex, alloc, before, std::span(input), init, result.data(), op);

  return measure_bandwidth_of_invocation_in_gigabytes_per_second(num_trials, sizeof(int) * input.size(), [&]
  {
    single_pass_reduce(ex, alloc, before, std::span(input), init, result.data(), op);
  });
}


// these expected performance intervals are in units of percent of theoretical peak bandwidth
performance_expectations_t single_pass_reduce_expectations = {
  {"NVIDIA GeForce RTX 3070", {0.92, 0.97}},
  {"NVIDIA RTX A5000", {0.92, 0.96}}
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

  report_performance(bandwidth_to_performance(bandwidth), single_pass_reduce_expectations);

  std::cout << "OK" << std::endl;

  return 0; 
}

