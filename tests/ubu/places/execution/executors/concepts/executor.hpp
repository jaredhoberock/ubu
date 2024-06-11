#include <ubu/places/causality/past_event.hpp>
#include <ubu/places/execution/executors/concepts/executor.hpp>

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


struct executor_with_execute_after_member
{
  bool operator==(const executor_with_execute_after_member&) const { return true; }
  bool operator!=(const executor_with_execute_after_member&) const { return false; }

  using happening_type = ns::past_event;

  ns::past_event execute_after(ns::past_event, auto&& f) const
  {
    f();
    return {};
  }
};


struct executor_with_execute_after_free_function
{
  bool operator==(const executor_with_execute_after_free_function&) const { return true; }
  bool operator!=(const executor_with_execute_after_free_function&) const { return false; }

  using happening_type = ns::past_event;
};

ns::past_event execute_after(const executor_with_execute_after_free_function&, ns::past_event, auto&& f)
{
  f();
  return {};
}


void test()
{
  {
    static_assert(ns::executor<executor_with_execute_after_member>);
  }

  {
    static_assert(ns::executor<executor_with_execute_after_free_function>);
  }
}


void test_executor()
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

