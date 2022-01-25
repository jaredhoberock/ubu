#include <cassert>
#include <aspera/event/complete_event.hpp>
#include <aspera/executor/first_execute.hpp>

#ifdef __CUDACC__
#include <cuda_runtime_api.h>

template<class F>
__global__ void device_invoke(F f)
{
  f();
}
#endif

namespace ns = aspera;

struct has_first_execute_member_function
{
  template<class F>
  ns::complete_event first_execute(F&& f) const
  {
    f();
    return {};
  }
};


struct has_first_execute_free_function {};

template<class F>
ns::complete_event first_execute(const has_first_execute_free_function&, F&& f)
{
  f();
  return {};
}


void test()
{
  {
    auto lambda = []{};

    static_assert(std::is_same_v<ns::complete_event, ns::first_execute_result_t<has_first_execute_member_function, decltype(lambda)>>, "Expected complete_event.");
  }

  {
    auto lambda = []{};

    static_assert(std::is_same_v<ns::complete_event, ns::first_execute_result_t<has_first_execute_free_function, decltype(lambda)>>, "Expected complete_event.");
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

