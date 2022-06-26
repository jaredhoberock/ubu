#include <ubu/event/happening.hpp>

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


struct happening_with_member_functions
{
  void wait() const {}

  happening_with_member_functions because_of(const happening_with_member_functions&) const
  {
    return {};
  }
};


struct happening_with_wait_member
{
  void wait() const {}
};

happening_with_wait_member because_of(const happening_with_wait_member&, const happening_with_wait_member&)
{
  return {};
}


struct happening_with_free_functions {};

void wait(const happening_with_free_functions&) {}

happening_with_free_functions because_of(const happening_with_free_functions&, const happening_with_free_functions&)
{
  return {};
}


struct happening_with_wait_free_function
{
  happening_with_wait_free_function because_of(const happening_with_wait_free_function&) const
  {
    return {};
  }
};

void wait(const happening_with_wait_free_function&) {}


void test()
{
  static_assert(ns::happening<happening_with_member_functions>);

  static_assert(ns::happening<happening_with_wait_member>);

  static_assert(ns::happening<happening_with_free_functions>);

  static_assert(ns::happening<happening_with_wait_free_function>);

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

