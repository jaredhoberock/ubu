#include <aspera/event/always_complete_event.hpp>
#include <aspera/execution/executor/execute_after.hpp>

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

struct has_execute_after_member_function
{
  template<ns::event E, class F>
  ns::always_complete_event execute_after(E&& before, F&& f) const
  {
    ns::wait(std::move(before));
    f();
    return {};
  }
};


struct has_execute_after_free_function {};

template<ns::event E, class F>
ns::always_complete_event execute_after(const has_execute_after_free_function&, E&& before, F&& f)
{
  ns::wait(std::move(before));
  f();
  return {};
}


void test()
{
  {
    auto lambda = []{};

    static_assert(std::is_same_v<ns::always_complete_event, ns::execute_after_result_t<has_execute_after_member_function, ns::always_complete_event, decltype(lambda)>>, "Expected always_complete_event.");
  }

  {
    auto lambda = []{};

    static_assert(std::is_same_v<ns::always_complete_event, ns::execute_after_result_t<has_execute_after_free_function, ns::always_complete_event, decltype(lambda)>>, "Expected always_complete_event.");
  }
}

void test_execute_after_result_t()
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

