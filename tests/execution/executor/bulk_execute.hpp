#include <aspera/event/always_complete_event.hpp>
#include <aspera/event/wait.hpp>
#include <aspera/execution/executor/bulk_execute.hpp>
#include <aspera/execution/executor/inline_executor.hpp>

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

namespace ns = aspera;

struct has_bulk_execute_member
{
  template<class F>
  ns::always_complete_event bulk_execute(ns::always_complete_event before, int n, F&& f) const
  {
    before.wait();

    for(int i = 0; i < n; ++i)
    {
      f(i);
    }

    return {};
  }
};


struct has_bulk_execute_free_function {};

template<class F>
ns::always_complete_event bulk_execute(const has_bulk_execute_free_function&, ns::always_complete_event before, int n, F&& f)
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
    has_bulk_execute_member e;

    int counter = 0;

    ns::always_complete_event before;
    int expected = 10;

    auto done = ns::bulk_execute(e, before, expected, [&](int){ ++counter; });
    ns::wait(done);

    assert(expected == counter);
  }

  {
    has_bulk_execute_free_function e;

    int counter = 0;

    ns::always_complete_event before;
    int expected = 10;

    auto done = ns::bulk_execute(e, before, expected, [&](int){ ++counter; });
    ns::wait(done);

    assert(expected == counter);
  }

  {
    // 1D grid space with inline_executor

    ns::inline_executor e;

    int counter = 0;

    ns::always_complete_event before;
    int expected = 10;

    auto done = ns::bulk_execute(e, before, expected, [&](int){ ++counter; });
    ns::wait(done);

    assert(expected == counter);
  }

  {
    // 2D grid space with inline_executor

    ns::inline_executor e;

    int counter = 0;

    ns::always_complete_event before;

    ns::int2 grid_shape{2,5};

    auto done = ns::bulk_execute(e, before, grid_shape, [&](ns::int2){ ++counter; });
    ns::wait(done);

    assert(grid_shape.product() == counter);
  }

  {
    // 3D grid space with inline_executor

    ns::inline_executor e;

    int counter = 0;

    ns::always_complete_event before;

    ns::int3 grid_shape{2,5,7};

    auto done = ns::bulk_execute(e, before, grid_shape, [&](ns::int3){ ++counter; });
    ns::wait(done);

    assert(grid_shape.product() == counter);
  }
}

void test_bulk_execute()
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

