#include <cassert>
#include <random>
#include <span>
#include <ubu/ubu.hpp>
#include <vector>

template<class T>
using device_vector = std::vector<T, ubu::cuda::managed_allocator<T>>;

void test(int block_size)
{
#if defined(__CUDACC__)
  using namespace ubu;

  device_vector<int> input(block_size);
  std::generate(input.begin(), input.end(), std::default_random_engine());
  std::span input_view(input.data(), input.size());

  device_vector<int> result(block_size);
  std::span result_view(result.data(), result.size());

  int num_warps = block_size / cuda::warp_size;
  int block_workspace_size = sizeof(int) * num_warps;

  bulk_execute_with_workspace(cuda::device_executor(),
                              cuda::device_allocator<std::byte>(),
                              ubu::int2(block_size, 1),
                              ubu::int2(block_workspace_size, 0),
                              [=](ubu::int2 coord, auto ws)
  {
    basic_cooperator block(coord.x, block_size, get_local_workspace(ws));

    int value = input_view[coord.x];

    int result = coop_inclusive_scan(block, value, std::plus{});

    result_view[coord.x] = result;
  });

  std::vector<int> expected(block_size);
  int cumulative_sum = 0;
  for(int i = 0; i != input.size(); ++i)
  {
    if(i == 0)
    {
      cumulative_sum = input[i];
    }
    else
    {
      cumulative_sum = cumulative_sum + input[i];
    }

    expected[i] = cumulative_sum;
  }

  if(not std::equal(expected.begin(), expected.end(), result.begin()))
  {
    std::cerr << "test(" << block_size << ") failed" << std::endl;
    assert(false);
  }
#endif
}

void test_coop_inclusive_scan()
{
  for(int num_warps = 1; num_warps <= 32; ++num_warps)
  {
    test(num_warps * ubu::cuda::warp_size);
  }
}

