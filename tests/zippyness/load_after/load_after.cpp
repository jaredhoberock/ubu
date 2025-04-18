// circle --verbose -O3 -std=c++20 -I../../../  -sm_60 block_load.cpp -lcudart -lfmt -o block_load
#include "measure_bandwidth_of_invocation.hpp"
#include "validate.hpp"
#include <algorithm>
#include <vector>
#include <ubu/ubu.hpp>

// postcondition: is_injective(result)
constexpr ubu::layout_of_rank<3> auto layout_for_load(std::size_t n)
{
  using namespace ubu;

  auto num_elements_per_thread = 15_c;
  auto num_threads_per_block = 128_c;
  auto tile_size = num_elements_per_thread * num_threads_per_block;
  auto num_blocks = ceil_div(n, tile_size);

  std::tuple shape(num_elements_per_thread, num_threads_per_block, num_blocks);

  return strided_layout(shape);
}

template<ubu::sized_vector_like I>
ubu::cuda::event load_after(ubu::cuda::device_executor gpu, const ubu::cuda::event& before, I input)
{
  using namespace ubu;
  using T = tensor_element_t<I>;

  layout_of_rank<3> auto layout = layout_for_load(std::size(input));
  tensor_of_rank<3> auto input_tiles = compose(input, layout);
  
  auto shape = layout.shape();

  constexpr auto num_elements_per_thread = get<0>(shape);
  constexpr auto block_size = get<1>(shape);
  std::size_t num_blocks = get<2>(shape);

  constexpr std::size_t max_tile_size = num_elements_per_thread * block_size;

  // kernel configuration
  std::pair kernel_shape(block_size, num_blocks);
  std::pair workspace_shape(constant<sizeof(T)>() * num_elements_per_thread * block_size, 0_c);

  // 32 registers / 409.21 GB/s ~ 93% peak bandwidth on RTX 3070
  return bulk_execute_with_workspace_after(gpu,
                                           cuda::device_allocator<std::byte>(),
                                           before,
                                           kernel_shape,
                                           workspace_shape,
                                           [=](ubu::int2 idx, auto ws)
  {
    basic_cooperator grid(idx, kernel_shape, ws);

    auto [block, block_idx] = subgroup_and_id(grid);

    // get this block's tile
    auto tile = slice(input_tiles, std::tuple(_, _, block_idx));

    // load each thread's slice into registers
    inplace_vector thread_data = coop_load<num_elements_per_thread>(block, tile.span());
  });
}

template<class T>
using device_vector = std::vector<T, ubu::cuda::managed_allocator<T>>;

double test_performance(std::size_t size, std::size_t num_trials)
{
  device_vector<int> input(size);
  std::span input_view(input);
  ubu::cuda::device_executor ex;
  ubu::cuda::event before = ubu::initial_happening(ex);

  // warmup
  load_after(ex, before, input_view);

  return measure_bandwidth_of_invocation_in_gigabytes_per_second(num_trials, input_view.size_bytes(), [&]
  {
    load_after(ex, before, input_view);
  });
}

// these expected performance intervals are in units of percent of theoretical peak bandwidth
performance_expectations_t load_after_expectations = {
  {"NVIDIA GeForce RTX 3070", {0.92, 0.97}},
  {"NVIDIA RTX A5000", {0.92, 0.96}}
};


int main(int argc, char** argv)
{
  std::size_t performance_size = choose_large_problem_size_using_heuristic<int>(1);
  std::size_t num_performance_trials = 1000;

  if(argc == 2)
  {
    std::string_view arg(argv[1]);
    if(arg != "quick")
    {
      std::cerr << "Unrecognized argument \"" << arg << "\"" << std::endl;
      return -1;
    }

    performance_size /= 10;
    num_performance_trials = 30;
  }

  std::cout << "Testing performance... " << std::flush;
  double bandwidth = test_performance(performance_size, num_performance_trials);
  std::cout << "Done." << std::endl;

  report_performance(bandwidth_to_performance(bandwidth), load_after_expectations);

  std::cout << "OK" << std::endl;

  return 0; 
}

