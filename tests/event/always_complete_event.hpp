#include <ubu/event/always_complete_event.hpp>
#include <ubu/event/event.hpp>
#include <ubu/event/wait.hpp>

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


void test()
{
  {
    static_assert(ns::event<ns::always_complete_event>);
    ns::always_complete_event e;
    ns::wait(e);
  }
}


void test_always_complete_event()
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

