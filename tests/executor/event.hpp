#include <aspera/executor/event.hpp>
#include <cassert>
#include <future>

#ifdef __CUDACC__
#include <cuda_runtime_api.h>

template<class F>
__global__ void device_invoke(F f)
{
  f();
}
#endif

namespace ns = aspera;


struct event_with_wait_member
{
  void wait() {}
};


struct event_with_wait_free_function {};

void wait(event_with_wait_free_function&) {}


void test()
{
  {
    static_assert(ns::event<event_with_wait_member>);
  }

  {
    static_assert(ns::event<event_with_wait_free_function>);
  }

  {
    static_assert(ns::event<std::future<void>>);
  }

  {
    static_assert(ns::event<std::future<int>>);
  }
}


void test_event()
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

