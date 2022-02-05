#include <aspera/event/always_complete_event.hpp>
#include <aspera/executor/bulk_execute.hpp>

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

struct has_bulk_execute_member
{
  template<class F>
  ns::always_complete_event bulk_execute(ns::always_complete_event before, int n, F&& f) const
  {
    before.wait();

    for(int i = 0; i < n; ++i)
    {
      f(i);
    }

    return {};
  }
};


struct has_bulk_execute_free_function {};

template<class F>
ns::always_complete_event bulk_execute(const has_bulk_execute_free_function&, ns::always_complete_event before, int n, F&& f)
{
  before.wait();

  for(int i = 0; i < n; ++i)
  {
    f(i);
  }

  return {};
}


void test()
{
  {
    auto lambda = [](int){};

    static_assert(std::is_same_v<ns::always_complete_event, ns::bulk_execute_result_t<has_bulk_execute_member, ns::always_complete_event, int, decltype(lambda)>>, "Expected always_complete_event.");
  }

  {
    auto lambda = [](int){};

    static_assert(std::is_same_v<ns::always_complete_event, ns::bulk_execute_result_t<has_bulk_execute_free_function, ns::always_complete_event, int, decltype(lambda)>>, "Expected always_complete_event.");
  }
}

void test_bulk_execute_result_t()
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


