#include <ubu/causality/happening.hpp>

#undef NDEBUG
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

namespace ns = ubu;


struct happening_with_static_member_function
{
  static happening_with_static_member_function initial_happening()
  {
    return {};
  }
};


struct happening_with_member_function
{
  happening_with_member_function initial_happening() const
  {
    return {};
  }
};


struct happening_with_free_function {};

happening_with_free_function initial_happening(happening_with_free_function)
{
  return {};
}


void test()
{
  static_assert(ns::happening<happening_with_static_member_function>);

  static_assert(ns::happening<happening_with_member_function>);

  static_assert(ns::happening<happening_with_free_function>);

  static_assert(ns::happening<std::future<void>>);
}


void test_happening()
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

