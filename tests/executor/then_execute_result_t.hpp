#include <aspera/event/complete_event.hpp>
#include <aspera/executor/then_execute.hpp>

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

namespace ns = aspera;

struct has_then_execute_member_function
{
  template<ns::event E, class F>
  ns::complete_event then_execute(E&& before, F&& f) const
  {
    ns::wait(std::move(before));
    f();
    return {};
  }
};


struct has_then_execute_free_function {};

template<ns::event E, class F>
ns::complete_event then_execute(const has_then_execute_free_function&, E&& before, F&& f)
{
  ns::wait(std::move(before));
  f();
  return {};
}


void test()
{
  {
    auto lambda = []{};

    static_assert(std::is_same_v<ns::complete_event, ns::then_execute_result_t<has_then_execute_member_function, ns::complete_event, decltype(lambda)>>, "Expected complete_event.");
  }

  {
    auto lambda = []{};

    static_assert(std::is_same_v<ns::complete_event, ns::then_execute_result_t<has_then_execute_free_function, ns::complete_event, decltype(lambda)>>, "Expected complete_event.");
  }
}

void test_then_execute_result_t()
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

