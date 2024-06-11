#include <cassert>
#include <optional>
#include <span>
#include <ubu/cooperation/algorithms/coop_reduce.hpp>
#include <ubu/cooperation/cooperators/basic_cooperator.hpp>
#include <ubu/cooperation/cooperators/is_leader.hpp>
#include <ubu/places/execution/executors/bulk_execute_with_workspace.hpp>
#include <ubu/platforms/cuda/device_allocator.hpp>
#include <ubu/platforms/cuda/device_executor.hpp>
#include <ubu/platforms/cuda/managed_allocator.hpp>
#include <ubu/tensors/coordinates/point.hpp>
#include <vector>

namespace ns = ubu;

template<class T>
using device_vector = std::vector<T, ns::cuda::managed_allocator<T>>;

void test(int num_warps)
{
#if defined(__CUDACC__)
  using namespace ns;

  int block_size = num_warps * cuda::warp_size;

  device_vector<int> input(block_size, 1);
  std::span input_view(input.data(), input.size());

  device_vector<int> result(1, -1);
  std::span result_view(result.data(), result.size());

  auto workspace_size = sizeof(int) * num_warps;

  bulk_execute_with_workspace(cuda::device_executor(),
                              cuda::device_allocator<std::byte>(),
                              ubu::int2(block_size, 1),
                              std::pair(workspace_size, 0),
                              [=](ubu::int2 coord, auto ws)
  {
    basic_cooperator block(coord.x, block_size, get_local_workspace(ws));

    std::optional value = input_view[id(block)];

    std::optional result = coop_reduce(block, value, std::plus{});

    if(result)
    {
      assert(is_leader(block));
      result_view[0] = *result;
    }
    else
    {
      assert(not is_leader(block));
    }
  });

  int expected = block_size;

  assert(expected == result[0]);
#endif
}

void test_block_like_coop_reduce()
{
  int max_num_warps_per_block = 32;

  for(int num_warps = 1; num_warps < max_num_warps_per_block; ++num_warps)
  {
    test(num_warps);
  }
}


void test_coop_reduce()
{
  test_block_like_coop_reduce();
}

