#include <ubu/event/event.hpp>

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


struct event_with_member_functions
{
  void wait() const {}

  event_with_member_functions because_of(const event_with_member_functions&) const
  {
    return {};
  }
};


struct event_with_wait_member
{
  void wait() const {}
};

event_with_wait_member because_of(const event_with_wait_member&, const event_with_wait_member&)
{
  return {};
}


struct event_with_free_functions {};

void wait(const event_with_free_functions&) {}

event_with_free_functions because_of(const event_with_free_functions&, const event_with_free_functions&)
{
  return {};
}


struct event_with_wait_free_function
{
  event_with_wait_free_function because_of(const event_with_wait_free_function&) const
  {
    return {};
  }
};

void wait(const event_with_wait_free_function&) {}


void test()
{
  static_assert(ns::event<event_with_member_functions>);

  static_assert(ns::event<event_with_wait_member>);

  static_assert(ns::event<event_with_free_functions>);

  static_assert(ns::event<event_with_wait_free_function>);

  static_assert(ns::event<std::future<void>>);
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

