#include <cassert>
#include <optional>
#include <span>
#include <ubu/cooperation/algorithm/coop_reduce.hpp>
#include <ubu/cooperation/cooperator/basic_cooperator.hpp>
#include <ubu/cooperation/cooperator/is_leader.hpp>
#include <ubu/places/execution/executor/bulk_execute_with_workspace.hpp>
#include <ubu/platforms/cuda/device_allocator.hpp>
#include <ubu/platforms/cuda/coop_executor.hpp>
#include <ubu/platforms/cuda/managed_allocator.hpp>
#include <ubu/tensors/coordinates/point.hpp>
#include <vector>

namespace ns = ubu;

template<class T>
using device_vector = std::vector<T, ns::cuda::managed_allocator<T>>;

void test(int num_warps_per_block, int num_blocks)
{
#if defined(__CUDACC__)
  using namespace ns;

  int block_size = num_warps_per_block * cuda::warp_size;

  int n = block_size * num_blocks;

  device_vector<int> input(n, 1);
  std::span input_view(input.data(), input.size());

  device_vector<int> result(1, -1);
  std::span result_view(result.data(), result.size());

  ubu::int2 grid_shape(block_size, num_blocks);

  auto inner_workspace_size = sizeof(int) * num_warps_per_block;
  auto outer_workspace_size = sizeof(int) * num_blocks;
  ubu::int2 workspace_shape(inner_workspace_size, outer_workspace_size);

  bulk_execute_with_workspace(cuda::coop_executor(),
                              cuda::device_allocator<std::byte>(),
                              grid_shape,
                              workspace_shape,
                              [=](ubu::int2 coord, auto ws)
  {
    basic_cooperator grid(coord, grid_shape, ws);

    std::optional value = input_view[id(grid)];

    std::optional result = coop_reduce(grid, value, std::plus{});

    if(result)
    {
      assert(is_leader(grid));
      result_view[0] = *result;
    }
    else
    {
      assert(not is_leader(grid));
    }
  });

  int expected = n;

  assert(expected == result[0]);
#endif
}

void test_cooperative_grid_like_coop_reduce()
{
  int max_num_blocks_per_grid = 16; // this number was arbitrarily chosen
  int max_num_warps_per_block = 32;

  for(int num_blocks = 1; num_blocks < max_num_blocks_per_grid; ++num_blocks)
  {
    for(int num_warps = 1; num_warps < max_num_warps_per_block; ++num_warps)
    {
      test(num_warps, num_warps);
    }
  }
}


void test_coop_reduce()
{
  test_cooperative_grid_like_coop_reduce();
}

