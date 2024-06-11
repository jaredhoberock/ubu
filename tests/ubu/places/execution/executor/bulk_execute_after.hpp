#include <algorithm>
#include <ranges>
#include <ubu/places/causality/past_event.hpp>
#include <ubu/places/causality/wait.hpp>
#include <ubu/places/execution/executor/bulk_execute_after.hpp>
#include <ubu/places/memory/buffer/reinterpret_buffer.hpp>
#include <ubu/platform/cpp/inline_executor.hpp>
#include <ubu/tensor/coordinates/point.hpp>
#include <ubu/tensor/views/lattice.hpp>
#include <numeric>
#include <vector>

#undef NDEBUG
#include <cassert>

#ifdef __CUDACC__
#include <cuda_runtime_api.h>

template<class F>
__global__ void device_invoke(F f)
{
  f();
}
#endif

namespace ns = ubu;

struct has_bulk_execute_after_member
{
  template<class F>
  ns::past_event bulk_execute_after(ns::past_event before, int n, F&& f) const
  {
    before.wait();

    for(int i = 0; i < n; ++i)
    {
      f(i);
    }

    return {};
  }
};


struct has_bulk_execute_after_free_function {};

template<class F>
ns::past_event bulk_execute_after(const has_bulk_execute_after_free_function&, ns::past_event before, int n, F&& f)
{
  before.wait();

  for(int i = 0; i < n; ++i)
  {
    f(i);
  }

  return {};
}


template<std::ranges::view V, ns::coordinate S>
constexpr bool are_ascending_coordinates(V coords, S shape)
{
  ns::lattice expected(shape);

  for(int i = 0; i != coords.size(); ++i)
  {
    if(expected[i] != coords[i]) return false;
  }

  return true;
}


void test()
{
  {
    has_bulk_execute_after_member e;

    int n = 10;

    std::vector<int> expected(n);
    std::iota(expected.begin(), expected.end(), 0);
    std::vector<int> result(n,-1);

    ns::past_event before;

    auto done = ns::bulk_execute_after(e, before, n, [&](int coord)
    { 
      result[coord] = coord;
    });
    ns::wait(done);

    assert(expected == result);
  }

  {
    has_bulk_execute_after_free_function e;

    int n = 10;

    std::vector<int> expected(n);
    std::iota(expected.begin(), expected.end(), 0);
    std::vector<int> result(n,-1);

    ns::past_event before;

    auto done = ns::bulk_execute_after(e, before, n, [&](int coord)
    {
      result[coord] = coord;
    });
    ns::wait(done);

    assert(expected == result);
  }

  {
    // 1D grid shape with inline_executor

    ns::cpp::inline_executor e;

    int n = 10;

    std::vector<int> expected(n);
    std::iota(expected.begin(), expected.end(), 0);
    std::vector<int> result(n,-1);

    ns::past_event before;

    auto done = ns::bulk_execute_after(e, before, n, [&](int coord)
    {
      result[coord] = coord;
    });
    ns::wait(done);

    assert(expected == result);
  }

  {
    // 2D grid shape with inline_executor

    ns::cpp::inline_executor e;

    int counter = 0;

    ns::past_event before;

    ns::int2 grid_shape{2,5};

    auto done = ns::bulk_execute_after(e, before, grid_shape, [&](ns::int2 coord)
    { 
      ++counter;
    });
    ns::wait(done);

    assert(grid_shape.product() == counter);
  }

  {
    // 3D grid space with inline_executor

    ns::cpp::inline_executor e;

    int counter = 0;

    ns::past_event before;

    ns::int3 grid_shape{2,5,7};

    auto done = ns::bulk_execute_after(e, before, grid_shape, [&](ns::int3 coord)
    { 
      ++counter;
    });
    ns::wait(done);

    assert(grid_shape.product() == counter);
  }
}

void test_bulk_execute_after()
{
  test();

#ifdef __CUDACC__
  device_invoke<<<1,1>>>([] ()
  {
    test();
  });
  assert(cudaDeviceSynchronize() == cudaSuccess);
#endif
}

