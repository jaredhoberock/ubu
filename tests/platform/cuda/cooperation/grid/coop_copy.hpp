#include <cassert>
#include <iostream>
#include <numeric>
#include <ubu/ubu.hpp>

template<class T>
using device_vector = std::vector<T, ubu::cuda::managed_allocator<T>>;

void test_coop_copy(int num_blocks_per_grid, int num_warps_per_block, int num_elements_per_thread)
{
#if defined(__CUDACC__)
  using namespace ubu;

  int block_size = num_elements_per_thread * cuda::warp_size;

  device_vector<int> input(num_blocks_per_grid * block_size * num_elements_per_thread);
  std::iota(input.begin(), input.end(), 0);
  std::span input_view(input.data(), input.size());

  device_vector<int> result(input.size());
  std::span result_view(result.begin(), result.size());

  ubu::int2 shape(block_size, num_blocks_per_grid);

  bulk_execute(cuda::coop_executor(), shape, [=](ubu::int2 coord)
  {
    basic_cooperator grid(coord, shape, cuda::coop_grid_workspace{});

    coop_copy(grid, input_view, result_view);
  });

  std::vector<int> expected(input.begin(), input.end());
  if(not std::equal(expected.begin(), expected.end(), result.begin()))
  {
    std::cerr << "test_coop_copy(" << num_blocks_per_grid << ", " << num_warps_per_block << ", " << num_elements_per_thread << ") failed" << std::endl;
    assert(false);
  }
#endif
}

void test_coop_copy()
{
  for(int num_blocks_per_grid = 1; num_blocks_per_grid < 4; ++num_blocks_per_grid)
  {
    for(int num_warps_per_block = 1; num_warps_per_block <= 4; ++num_warps_per_block)
    {
      for(int num_elements_per_thread = 1; num_elements_per_thread < 4; ++num_elements_per_thread)
      {
        test_coop_copy(num_blocks_per_grid, num_warps_per_block, num_elements_per_thread);
      }
    }
  }
}
