#include <cassert>
#include <optional>
#include <span>
#include <ubu/cooperation/cooperator/basic_cooperator.hpp>
#include <ubu/places/execution/executor/bulk_execute.hpp>
#include <ubu/platforms/cuda/cooperation.hpp>
#include <ubu/platforms/cuda/device_executor.hpp>
#include <ubu/platforms/cuda/managed_allocator.hpp>
#include <ubu/tensor/coordinates/point.hpp>
#include <vector>

namespace ns = ubu;

template<class T>
using device_vector = std::vector<T, ns::cuda::managed_allocator<T>>;

void test_warp_like_coop_reduce()
{
#if defined(__CUDACC__)
  using namespace ns;

  int n = 32;
  int expected = n;

  device_vector<int> input(n, 1);
  std::span input_view(input.data(), input.size());

  device_vector<int> result(1, -1);
  std::span result_view(result.data(), result.size());

  bulk_execute(cuda::device_executor(), ubu::int2(n, 1), [=](ubu::int2 coord)
  {
    basic_cooperator warp(coord.x, n, cuda::warp_workspace{});

    std::optional value = input_view[coord.x];
    std::optional result = cuda::coop_reduce(warp, value, std::plus{});

    if(result)
    {
      result_view[0] = *result;
    }
  });

  assert(expected == result[0]);
#endif
}


void test_coop_reduce()
{
  test_warp_like_coop_reduce();
}

