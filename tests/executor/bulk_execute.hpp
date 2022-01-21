#include <cassert>
#include <aspera/event/complete_event.hpp>
#include <aspera/event/wait.hpp>
#include <aspera/executor/bulk_execute.hpp>

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
  ns::complete_event bulk_execute(ns::complete_event before, int n, F&& f) const
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
ns::complete_event bulk_execute(const has_bulk_execute_free_function&, ns::complete_event before, int n, F&& f)
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

    ns::complete_event before;
    int n = 10;

    auto done = ns::bulk_execute(e, before, n, [&](int){ ++counter; });
    ns::wait(done);

    assert(n == counter);
  }

  {
    has_bulk_execute_free_function e;

    int counter = 0;

    ns::complete_event before;
    int n = 10;

    auto done = ns::bulk_execute(e, before, n, [&](int){ ++counter; });
    ns::wait(done);

    assert(n == counter);
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


