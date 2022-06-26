#include <ubu/causality/past_event.hpp>
#include <ubu/execution/executor/first_execute.hpp>

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

struct has_first_execute_member_function
{
  template<class F>
  ns::past_event first_execute(F&& f) const
  {
    f();
    return {};
  }
};


struct has_first_execute_free_function {};

template<class F>
ns::past_event first_execute(const has_first_execute_free_function&, F&& f)
{
  f();
  return {};
}


void test()
{
  {
    auto lambda = []{};

    static_assert(std::is_same_v<ns::past_event, ns::first_execute_result_t<has_first_execute_member_function, decltype(lambda)>>, "Expected past_event.");
  }

  {
    auto lambda = []{};

    static_assert(std::is_same_v<ns::past_event, ns::first_execute_result_t<has_first_execute_free_function, decltype(lambda)>>, "Expected past_event.");
  }
}

void test_first_execute_result_t()
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

