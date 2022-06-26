#include <ubu/event/past_event.hpp>
#include <ubu/event/wait.hpp>
#include <ubu/execution/executor/bulk_execute_after.hpp>
#include <ubu/execution/executor/inline_executor.hpp>

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


void test()
{
  {
    has_bulk_execute_after_member e;

    int counter = 0;

    ns::past_event before;
    int expected = 10;

    auto done = ns::bulk_execute_after(e, before, expected, [&](int){ ++counter; });
    ns::wait(done);

    assert(expected == counter);
  }

  {
    has_bulk_execute_after_free_function e;

    int counter = 0;

    ns::past_event before;
    int expected = 10;

    auto done = ns::bulk_execute_after(e, before, expected, [&](int){ ++counter; });
    ns::wait(done);

    assert(expected == counter);
  }

  {
    // 1D grid space with inline_executor

    ns::inline_executor e;

    int counter = 0;

    ns::past_event before;
    int expected = 10;

    auto done = ns::bulk_execute_after(e, before, expected, [&](int){ ++counter; });
    ns::wait(done);

    assert(expected == counter);
  }

  {
    // 2D grid space with inline_executor

    ns::inline_executor e;

    int counter = 0;

    ns::past_event before;

    ns::int2 grid_shape{2,5};

    auto done = ns::bulk_execute_after(e, before, grid_shape, [&](ns::int2){ ++counter; });
    ns::wait(done);

    assert(grid_shape.product() == counter);
  }

  {
    // 3D grid space with inline_executor

    ns::inline_executor e;

    int counter = 0;

    ns::past_event before;

    ns::int3 grid_shape{2,5,7};

    auto done = ns::bulk_execute_after(e, before, grid_shape, [&](ns::int3){ ++counter; });
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

