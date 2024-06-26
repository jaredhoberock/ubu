#include <ubu/places/causality/happening.hpp>
#include <ubu/places/causality/past_event.hpp>
#include <ubu/places/causality/wait.hpp>

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
    static_assert(ns::happening<ns::past_event>);
    ns::past_event e;
    ns::wait(e);
  }
}


void test_past_event()
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

