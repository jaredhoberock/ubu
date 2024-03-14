#include <algorithm>
#include <cassert>
#include <iostream>
#include <optional>
#include <random>
#include <span>
#include <ubu/ubu.hpp>
#include <vector>

template<class T>
using device_vector = std::vector<T, ubu::cuda::managed_allocator<T>>;

void test_coop_exclusive_scan()
{
#if defined(__CUDACC__)
  using namespace ubu;

  device_vector<int> input(cuda::warp_size, 1);
  std::generate(input.begin(), input.end(), std::default_random_engine());
  std::span input_view(input.data(), input.size());

  device_vector<int> result(cuda::warp_size);
  std::span result_view(result.data(), result.size());

  int init = 13;

  bulk_execute(cuda::device_executor(), ubu::int2(cuda::warp_size, 1), [=](ubu::int2 coord)
  {
    basic_cooperator warp(coord.x, cuda::warp_size, cuda::warp_workspace{});

    int value = input_view[coord.x];
    int result = coop_exclusive_scan(warp, init, value, std::plus{});
    result_view[coord.x] = result;
  });

  std::vector<int> expected(cuda::warp_size);
  int cumulative_sum = init;
  for(int i = 0; i < input.size(); ++i)
  {
    expected[i] = cumulative_sum;
    cumulative_sum += input[i];
  }

  if(not std::equal(expected.begin(), expected.end(), result.begin()))
  {
    assert(false);
  }
#endif
}

