#include <cassert>
#include <iostream>
#include <numeric>
#include <ubu/ubu.hpp>

template<class T>
using device_vector = std::vector<T, ubu::cuda::managed_allocator<T>>;

void test_coop_copy(int num_elements_per_thread)
{
#if defined(__CUDACC__)
  using namespace ubu;

  device_vector<int> input(num_elements_per_thread * ubu::cuda::warp_size);
  std::iota(input.begin(), input.end(), 0);
  std::span input_view(input.data(), input.size());

  device_vector<int> result(input.size());
  std::span result_view(result.begin(), result.size());

  bulk_execute(cuda::device_executor(), ubu::int2(cuda::warp_size, 1), [=](ubu::int2 coord)
  {
    basic_cooperator warp(coord.x, cuda::warp_size, cuda::warp_workspace{});

    coop_copy(warp, input_view, result_view);
  });

  std::vector<int> expected(input.begin(), input.end());
  if(not std::equal(expected.begin(), expected.end(), result.begin()))
  {
    std::cerr << "test_coop_copy(" << num_elements_per_thread << ") failed" << std::endl;
    assert(false);
  }
#endif
}

void test_coop_copy()
{
  for(int num_elements_per_thread = 1; num_elements_per_thread < 32; ++num_elements_per_thread)
  {
    test_coop_copy(num_elements_per_thread);
  }
}

